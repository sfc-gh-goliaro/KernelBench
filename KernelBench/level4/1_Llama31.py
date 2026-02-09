"""
Llama-3.1 Dense Decoder Model

A decoder-only transformer implementing Llama-3.1 architecture with paged KV cache:
- RMSNorm normalization
- Grouped-Query Attention (GQA) with per-layer paged KV cache
- Rotary Position Embeddings (RoPE)
- SwiGLU MLP (gate/up projection fused)

Implementation aligned with vLLM's paged attention design:
- Each attention layer owns its own KV cache
- Uses AttentionMetadata for slot_mapping, block_table, etc.
- Supports both prefill and decode phases

Variants from Table 5:
- Llama-3.1-8B: hidden=4096, heads=32, kv_heads=8, layers=32
- Llama-3.1-70B: hidden=8192, heads=64, kv_heads=8, layers=80
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

# Import level1 operators
from ..level1.normalization._4_RMSNorm import Model as RMSNorm
from ..level1.embeddings._1_RotaryEmbedding import Model as RotaryEmbedding
from ..level1.activations._7_Swish import Model as Swish
from ..level1.matmul._10_Linear import Model as Linear
from ..level1.attention._1_PagedKVCache import AttentionMetadata, create_attention_metadata
from ..level1.attention._2_Attention import MultiHeadAttention as GroupedQueryAttention


# ============================================================================
# Model Variants
# ============================================================================

VARIANTS: Dict[str, str] = {
    "8B": "meta-llama/Llama-3.1-8B",
    "70B": "meta-llama/Llama-3.1-70B",
}


# ============================================================================
# Component Modules
# ============================================================================

class LlamaAttention(nn.Module):
    """
    Llama-style attention block with per-layer paged KV cache.
    
    Uses:
    - Linear for Q/K/V/O projections
    - RotaryEmbedding for position encoding (with optional Llama3 scaling)
    - GroupedQueryAttention for attention with paged KV cache
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
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

        # Q/K/V/O projections
        self.q_proj = Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = Linear(num_heads * head_dim, hidden_size, bias=False)

        # RoPE with bhsd layout and optional Llama3 scaling
        self.rotary_emb = RotaryEmbedding(
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            base=rope_theta,
            layout="bhsd",
            rope_scaling=rope_scaling,
        )
        
        # GQA with its own KV cache (per-layer cache)
        self.attn = GroupedQueryAttention(
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

        # Q/K/V projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Compute position IDs accounting for context_lens
        # For prefill: positions are 0, 1, 2, ... seq_len-1
        # For decode: positions are context_lens, context_lens+1, ...
        if attn_metadata.is_prefill:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        else:
            # During decode, each sequence may have different context length
            # position_ids shape: (batch_size, seq_len)
            position_ids = attn_metadata.context_lens.unsqueeze(1) + torch.arange(seq_len, device=device).unsqueeze(0)

        # Apply RoPE with position-aware embeddings
        q, k = self.rotary_emb(q, k, position_ids)

        # Attention with paged KV cache
        attn_output = self.attn(q, k, v, attn_metadata)

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)

    def reset_cache(self):
        """Reset this layer's KV cache."""
        self.attn.reset_cache()


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP using level1 operators."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.swish(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class LlamaDecoderLayer(nn.Module):
    """Single Llama decoder layer with paged attention."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rms_norm_eps: float = 1e-6,
        block_size: int = 16,
        num_blocks: int = 1024,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps, learnable_weight=True, dim=-1)
        self.self_attn = LlamaAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            block_size=block_size,
            num_blocks=num_blocks,
            layer_idx=layer_idx,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps, learnable_weight=True, dim=-1)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(
        self, 
        x: torch.Tensor, 
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attn_metadata)
        x = residual + x

        # MLP with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x

    def reset_cache(self):
        """Reset this layer's KV cache."""
        self.self_attn.reset_cache()


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Llama-3.1 style decoder-only transformer with Paged KV Cache.
    
    Aligned with vLLM's design:
    - Each decoder layer has its own KV cache
    - Uses AttentionMetadata for cache management
    - Supports both prefill and decode phases
    
    Uses level1 operators:
    - RMSNorm (with learnable_weight=True, dim=-1)
    - RotaryEmbedding (with layout="bhsd")
    - GroupedQueryAttention (with paged KV cache)
    - Linear
    - Swish
    """
    
    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: Optional[int] = None,
        intermediate_size: int = 14336,
        max_seq_len: int = 8192,
        rope_theta: float = 500000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rms_norm_eps: float = 1e-5,
        block_size: int = 16,
        num_blocks: int = 1024,
        **kwargs
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
        self.rope_scaling = rope_scaling
        self.rms_norm_eps = rms_norm_eps
        self.block_size = block_size
        self.num_blocks = num_blocks
        
        # Token embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Decoder layers - each with its own KV cache
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                rms_norm_eps=rms_norm_eps,
                block_size=block_size,
                num_blocks=num_blocks,
                layer_idx=i,
            )
            for i in range(num_layers)
        ])
        
        # Final norm and LM head
        self.norm = RMSNorm(hidden_size, rms_norm_eps, learnable_weight=True, dim=-1)
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
        x = self.embed_tokens(input_ids)

        # All layers share the same block_table but each has its own cache
        for layer in self.layers:
            x = layer(x, attn_metadata)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def reset_cache(self):
        """Reset all layer KV caches."""
        for layer in self.layers:
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
