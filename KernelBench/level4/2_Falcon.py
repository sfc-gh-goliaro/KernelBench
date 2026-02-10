"""
Falcon Dense Decoder Model

A decoder-only transformer implementing Falcon architecture with paged KV cache:
- LayerNorm normalization (with bias)
- Multi-Query/Grouped-Query Attention (MQA/GQA) with per-layer paged KV cache
- Rotary Position Embeddings (RoPE)
- GELU MLP
- Parallel attention and MLP (Falcon-style)

Implementation aligned with vLLM's paged attention design:
- Each attention layer owns its own KV cache
- Uses AttentionMetadata for slot_mapping, block_table, etc.
- Supports both prefill and decode phases

Variants from Table 5:
- Falcon-7B: hidden=4544, heads=71, kv_heads=1 (MQA), layers=32
- Falcon-40B: hidden=8192, heads=128, kv_heads=8 (GQA), layers=60
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

# Import level1 operators
from ..level1.normalization._6_LayerNorm import Model as LayerNorm
from ..level1.embeddings._1_RotaryEmbedding import Model as RotaryEmbedding
from ..level1.activations._8_GELU import Model as GELU
from ..level1.matmul._10_Linear import Model as Linear
from ..level1.attention._1_PagedKVCache import AttentionMetadata, create_attention_metadata
from ..level1.attention._2_Attention import MultiHeadAttention as MultiQueryAttention


# ============================================================================
# Model Variants
# ============================================================================

VARIANTS: Dict[str, str] = {
    "7B": "tiiuae/falcon-7b",
    "40B": "tiiuae/falcon-40b",
}


# ============================================================================
# Component Modules
# ============================================================================

class FalconAttention(nn.Module):
    """
    Falcon-style attention block with per-layer paged KV cache.
    
    Uses:
    - Fused QKV projection (query_key_value)
    - RotaryEmbedding for position encoding
    - MultiQueryAttention for attention with paged KV cache
    
    Supports both MQA (num_kv_heads=1, Falcon-7B) and GQA (num_kv_heads>1, Falcon-40B).
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        block_size: int = 16,
        num_blocks: int = 1024,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx

        # Falcon uses fused QKV projection
        # Output size: num_heads * head_dim (Q) + num_kv_heads * head_dim (K) + num_kv_heads * head_dim (V)
        # For Falcon-7B with MQA: 71 * 64 + 1 * 64 + 1 * 64 = 4544 + 128 = 4672
        # For Falcon-40B with GQA: 128 * 64 + 8 * 64 + 8 * 64 = 8192 + 1024 = 9216
        self.query_key_value = Linear(
            hidden_size, 
            num_heads * head_dim + 2 * num_kv_heads * head_dim,  # Q + K + V
            bias=False
        )
        self.dense = Linear(num_heads * head_dim, hidden_size, bias=False)

        # RoPE with bhsd layout
        self.rotary_emb = RotaryEmbedding(
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            base=rope_theta,
            layout="bhsd",
        )
        
        # Attention with paged KV cache (per-layer cache)
        self.attn = MultiQueryAttention(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            num_blocks=num_blocks,
            max_seq_len=max_seq_len,
        )

    def forward(
        self, 
        x: torch.Tensor, 
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        Forward pass with paged KV cache.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            attn_metadata: Attention metadata (slot_mapping, block_table, etc.)
            
        Returns:
            Output tensor (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Fused QKV projection
        qkv = self.query_key_value(x)
        
        # Split into Q, K, V
        # Q: num_heads * head_dim, K: num_kv_heads * head_dim, V: num_kv_heads * head_dim
        q_size = self.num_heads * self.head_dim
        k_size = self.num_kv_heads * self.head_dim
        v_size = self.num_kv_heads * self.head_dim
        
        q = qkv[..., :q_size]
        k = qkv[..., q_size:q_size + k_size]
        v = qkv[..., q_size + k_size:]
        
        # Reshape to (batch, heads, seq, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Compute position IDs accounting for context_lens
        # For prefill: positions are 0, 1, 2, ... seq_len-1
        # For decode: positions are context_lens, context_lens+1, ...
        if attn_metadata.is_prefill:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        else:
            # During decode, each sequence may have different context length
            position_ids = attn_metadata.context_lens.unsqueeze(1) + torch.arange(seq_len, device=device).unsqueeze(0)

        # Apply RoPE - K has 1 head, Q has num_heads
        # The RoPE implementation broadcasts over the heads dimension correctly
        q, k = self.rotary_emb(q, k, position_ids)

        # Attention with paged KV cache
        attn_output = self.attn(q, k, v, attn_metadata)

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.dense(attn_output)

    def reset_cache(self):
        """Reset this layer's KV cache."""
        self.attn.reset_cache()


class FalconMLP(nn.Module):
    """Falcon MLP with GELU activation using level1 operators."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense_h_to_4h = Linear(hidden_size, intermediate_size, bias=False)
        self.dense_4h_to_h = Linear(intermediate_size, hidden_size, bias=False)
        self.gelu = GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_4h_to_h(self.gelu(self.dense_h_to_4h(x)))


class FalconDecoderLayer(nn.Module):
    """
    Single Falcon decoder layer with paged attention.
    
    Falcon uses parallel attention and MLP (both computed on the same input,
    then summed with residual).
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        layer_norm_eps: float = 1e-5,
        block_size: int = 16,
        num_blocks: int = 1024,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        
        # LayerNorm with bias (Falcon uses standard LayerNorm)
        self.input_layernorm = LayerNorm(hidden_size, eps=layer_norm_eps)
        
        self.self_attention = FalconAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            block_size=block_size,
            num_blocks=num_blocks,
            layer_idx=layer_idx,
        )
        
        self.mlp = FalconMLP(hidden_size, intermediate_size)

    def forward(
        self, 
        x: torch.Tensor, 
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # Falcon uses parallel attention and MLP
        residual = x
        x = self.input_layernorm(x)
        
        # Parallel attention and MLP on the normalized input
        attn_output = self.self_attention(x, attn_metadata)
        mlp_output = self.mlp(x)
        
        # Sum with residual
        x = residual + attn_output + mlp_output
        return x

    def reset_cache(self):
        """Reset this layer's KV cache."""
        self.self_attention.reset_cache()


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Falcon decoder-only transformer with Paged KV Cache.
    
    Aligned with vLLM's design:
    - Each decoder layer has its own KV cache
    - Uses AttentionMetadata for cache management
    - Supports both prefill and decode phases
    
    Uses level1 operators:
    - LayerNorm (with learnable weight and bias)
    - RotaryEmbedding
    - MultiQueryAttention (with paged KV cache)
    - Linear
    - GELU
    """
    
    def __init__(
        self,
        vocab_size: int = 65024,
        hidden_size: int = 4544,
        num_layers: int = 32,
        num_heads: int = 71,
        num_kv_heads: int = 1,
        head_dim: Optional[int] = None,
        intermediate_size: int = 18176,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        layer_norm_eps: float = 1e-5,
        block_size: int = 16,
        num_blocks: int = 1024,
        **kwargs  # Accept and ignore extra kwargs for flexibility
    ):
        super().__init__()
        
        if head_dim is None:
            head_dim = hidden_size // num_heads
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.layer_norm_eps = layer_norm_eps
        self.block_size = block_size
        self.num_blocks = num_blocks
        
        # Token embedding
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # Decoder layers - each with its own KV cache
        self.h = nn.ModuleList([
            FalconDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
                layer_norm_eps=layer_norm_eps,
                block_size=block_size,
                num_blocks=num_blocks,
                layer_idx=i,
            )
            for i in range(num_layers)
        ])
        
        # Final norm and LM head
        self.ln_f = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.lm_head = Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self, 
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        Forward pass with paged KV cache (vLLM-style interface).
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attn_metadata: Attention metadata containing:
                - slot_mapping: Where to write new K,V in cache
                - block_table: How to read K,V from cache
                - context_lens: Tokens already in cache
                - is_prefill: Whether this is prefill or decode
            
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
        """
        x = self.word_embeddings(input_ids)

        for layer in self.h:
            x = layer(x, attn_metadata)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def reset_cache(self):
        """Reset all layer KV caches."""
        for layer in self.h:
            layer.reset_cache()

    def _prefill(
        self,
        input_ids: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prefill (prompt processing) - process all prompt tokens at once.
        
        Args:
            input_ids: (batch_size, seq_len) prompt token IDs
            block_table: (batch_size, max_blocks_per_seq) block assignments
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # All tokens are new (no context)
        context_lens = torch.zeros(batch_size, dtype=torch.long, device=device)
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        
        attn_metadata = create_attention_metadata(
            batch_size=batch_size,
            seq_lens=seq_lens,
            context_lens=context_lens,
            block_table=block_table,
            block_size=self.block_size,
            device=device,
        )
        
        return self.forward(input_ids, attn_metadata)

    def _decode(
        self,
        input_ids: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode (token generation) - process one token at a time.
        
        Args:
            input_ids: (batch_size, 1) new token IDs
            block_table: (batch_size, max_blocks_per_seq) block assignments
            context_lens: (batch_size,) tokens already processed
            
        Returns:
            logits: (batch_size, 1, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        assert seq_len == 1, "Decode should process one token at a time"
        
        seq_lens = context_lens + 1
        
        attn_metadata = create_attention_metadata(
            batch_size=batch_size,
            seq_lens=seq_lens,
            context_lens=context_lens,
            block_table=block_table,
            block_size=self.block_size,
            device=device,
        )
        
        return self.forward(input_ids, attn_metadata)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        block_table: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: (batch_size, prompt_len) prompt token IDs
            max_new_tokens: Maximum number of tokens to generate. If 0, only
                performs prefill and returns logits (requires return_logits=True).
            block_table: Optional pre-allocated block table
            return_logits: If True, also return logits at each generation step
            
        Returns:
            If return_logits=False:
                generated_ids: (batch_size, prompt_len + max_new_tokens)
            If return_logits=True:
                Tuple of (generated_ids, logits_list) where logits_list contains
                logits tensors for prefill and each decode step.
                For max_new_tokens=0: returns (input_ids, [prefill_logits])
        """
        batch_size, prompt_len = input_ids.shape
        device = input_ids.device
        
        # Reset caches
        self.reset_cache()
        
        # Allocate block table if not provided
        if block_table is None:
            max_seq_len = prompt_len + max(max_new_tokens, 1)
            max_blocks = (max_seq_len + self.block_size - 1) // self.block_size
            block_table = torch.arange(
                max_blocks, device=device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1).contiguous()
            # Simple contiguous allocation - in practice would use block allocator
            for i in range(batch_size):
                block_table[i] = torch.arange(
                    i * max_blocks, 
                    (i + 1) * max_blocks, 
                    device=device
                ) % self.num_blocks
        
        # Prefill
        prefill_logits = self._prefill(input_ids, block_table)
        
        # Handle max_new_tokens=0 case (prefill only)
        if max_new_tokens == 0:
            if return_logits:
                return input_ids, [prefill_logits]
            else:
                # No new tokens to generate, just return input
                return input_ids
        
        all_logits = [prefill_logits] if return_logits else None
        next_token = prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        generated = [input_ids, next_token]
        context_lens = torch.full((batch_size,), prompt_len, dtype=torch.long, device=device)
        
        # Decode loop
        for _ in range(max_new_tokens - 1):
            decode_logits = self._decode(next_token, block_table, context_lens)
            context_lens += 1  # Increment after decode (context grows after each step)
            if return_logits:
                all_logits.append(decode_logits)
            next_token = decode_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
        
        generated_ids = torch.cat(generated, dim=1)
        
        if return_logits:
            return generated_ids, all_logits
        return generated_ids
