"""
BLOOM Dense Decoder Model

A decoder-only transformer implementing BLOOM architecture with paged KV cache:
- LayerNorm normalization (including word embeddings LayerNorm)
- Multi-Head Attention with ALiBi positional encoding
- Per-layer paged KV cache
- GELU MLP

Implementation aligned with vLLM's paged attention design:
- Each attention layer owns its own KV cache
- Uses AttentionMetadata for slot_mapping, block_table, etc.
- Supports both prefill and decode phases

Variants:
- BLOOM-560M: hidden=1024, heads=16, layers=24
- BLOOM-1.1B: hidden=1536, heads=16, layers=24
- BLOOM-1.7B: hidden=2048, heads=16, layers=24
- BLOOM-3B: hidden=2560, heads=32, layers=30
- BLOOM-7.1B: hidden=4096, heads=32, layers=30
- BLOOM-176B: hidden=14336, heads=112, layers=70
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

# Import level1 operators
from ..level1.normalization._6_LayerNorm import Model as LayerNorm
from ..level1.activations._8_GELU import Model as GELU
from ..level1.matmul._10_Linear import Model as Linear
from ..level1.attention._3_GroupedQueryAttention import (
    AttentionMetadata,
    create_attention_metadata,
)


# ============================================================================
# Model Variants
# ============================================================================

VARIANTS: Dict[str, str] = {
    "560M": "bigscience/bloom-560m",
    "1.1B": "bigscience/bloom-1b1",
    "1.7B": "bigscience/bloom-1b7",
    "3B": "bigscience/bloom-3b",
    "7.1B": "bigscience/bloom-7b1",
    "176B": "bigscience/bloom",
}


# ============================================================================
# ALiBi Utilities
# ============================================================================

def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    Compute ALiBi slopes for each attention head.

    ALiBi (Attention with Linear Biases) uses head-specific slopes
    to apply linear position-based biases to attention scores.

    Args:
        num_heads: Number of attention heads

    Returns:
        slopes: (num_heads,) tensor of slope values
    """
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        num_remaining_heads = min(
            closest_power_of_2, num_heads - closest_power_of_2
        )
        extra_powers = torch.arange(
            start=1, end=1 + 2 * num_remaining_heads, step=2, dtype=torch.int32
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes


# ============================================================================
# Component Modules
# ============================================================================

class BloomAttention(nn.Module):
    """
    BLOOM-style attention block with ALiBi and per-layer paged KV cache.

    Uses:
    - Fused QKV projection (matching HuggingFace structure)
    - ALiBi for position encoding (no RoPE)
    - Paged KV cache for efficient memory management
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        block_size: int = 16,
        num_blocks: int = 1024,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.max_seq_len = max_seq_len

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Fused QKV projection (matches HuggingFace BLOOM structure)
        self.query_key_value = Linear(hidden_size, 3 * num_heads * head_dim, bias=True)
        # Output projection (named 'dense' to match HuggingFace)
        self.dense = Linear(num_heads * head_dim, hidden_size, bias=True)

        # ALiBi slopes
        slopes = get_alibi_slopes(num_heads)
        self.register_buffer('alibi_slopes', slopes)

        # Initialize paged KV cache as buffer (not parameter - no gradients)
        # Shape: (num_blocks, block_size, num_heads, head_dim, 2) where 2 = [K, V]
        self.register_buffer('kv_cache', torch.zeros(
            num_blocks, block_size, num_heads, head_dim, 2
        ))

    def reset_cache(self):
        """Reset this layer's KV cache."""
        self.kv_cache.zero_()

    def _write_to_cache(self, k: torch.Tensor, v: torch.Tensor,
                        slot_mapping: torch.Tensor) -> None:
        """
        Write new K,V to cache using slot_mapping.

        Args:
            k: (batch_size, num_heads, seq_len, head_dim)
            v: (batch_size, num_heads, seq_len, head_dim)
            slot_mapping: (num_tokens,) - absolute positions in cache
        """
        if slot_mapping.numel() == 0:
            return

        # Flatten K,V to (num_tokens, num_heads, head_dim)
        k_flat = k.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim)
        v_flat = v.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim)

        # Compute block indices and offsets from slot_mapping
        block_indices = slot_mapping // self.block_size
        block_offsets = slot_mapping % self.block_size

        # Write to cache using advanced indexing
        self.kv_cache[block_indices, block_offsets, :, :, 0] = k_flat
        self.kv_cache[block_indices, block_offsets, :, :, 1] = v_flat

    def _gather_kv_from_cache(self, block_table: torch.Tensor,
                               context_lens: torch.Tensor,
                               max_context_len: int) -> tuple:
        """
        Gather K and V tensors from paged cache.

        Args:
            block_table: (batch_size, max_blocks_per_seq)
            context_lens: (batch_size,)
            max_context_len: Maximum context length to gather

        Returns:
            k_cache: (batch_size, num_heads, max_context_len, head_dim)
            v_cache: (batch_size, num_heads, max_context_len, head_dim)
        """
        batch_size = block_table.shape[0]
        device = self.kv_cache.device

        # Calculate how many blocks we need
        num_blocks_needed = (max_context_len + self.block_size - 1) // self.block_size

        # Gather only the blocks we need
        block_table_truncated = block_table[:, :num_blocks_needed]

        gathered_blocks = self.kv_cache[block_table_truncated.flatten()]
        gathered_blocks = gathered_blocks.view(
            batch_size, num_blocks_needed, self.block_size, self.num_heads, self.head_dim, 2
        )

        # Reshape to (batch, num_blocks * block_size, heads, head_dim, 2)
        gathered = gathered_blocks.view(
            batch_size, num_blocks_needed * self.block_size, self.num_heads, self.head_dim, 2
        )

        # Truncate to exactly max_context_len
        gathered = gathered[:, :max_context_len, :, :, :]

        # Split K and V, transpose to (batch, heads, context, head_dim)
        k_cache = gathered[..., 0].transpose(1, 2)
        v_cache = gathered[..., 1].transpose(1, 2)

        return k_cache, v_cache

    def _build_alibi_tensor(self, seq_len: int, batch_size: int,
                             device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Build ALiBi tensor matching HuggingFace's implementation.
        
        Returns:
            alibi: (batch_size * num_heads, 1, seq_len) - added to attention scores
        """
        # Positions: [0, 1, 2, ..., seq_len-1]
        arange_tensor = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0)
        # Apply slopes: (num_heads, 1) * (1, seq_len) -> (num_heads, seq_len)
        alibi = self.alibi_slopes.to(device=device, dtype=dtype).unsqueeze(1) * arange_tensor
        # Expand for batch and reshape: (batch * num_heads, 1, seq_len)
        alibi = alibi.unsqueeze(0).expand(batch_size, -1, -1)
        alibi = alibi.reshape(batch_size * self.num_heads, 1, seq_len)
        return alibi

    def _prefill_attention(self, q: torch.Tensor, k: torch.Tensor,
                           v: torch.Tensor) -> torch.Tensor:
        """
        Compute attention during prefill phase (no cache read needed).
        Matches HuggingFace BLOOM's attention implementation.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        dtype = q.dtype

        # Reshape for batched matmul: (batch * heads, seq, head_dim)
        q_reshaped = q.reshape(batch_size * num_heads, seq_len, head_dim)
        k_reshaped = k.reshape(batch_size * num_heads, seq_len, head_dim).transpose(-1, -2)
        v_reshaped = v.reshape(batch_size * num_heads, seq_len, head_dim)

        # Build ALiBi tensor: (batch * heads, 1, seq_len)
        alibi = self._build_alibi_tensor(seq_len, batch_size, device, dtype)

        # Compute attention scores with ALiBi: alibi + scale * (q @ k.T)
        # Using baddbmm pattern: beta * alibi + alpha * (q @ k.T)
        scores = torch.baddbmm(
            alibi,
            q_reshaped,
            k_reshaped,
            beta=1.0,
            alpha=self.scale,
        )
        # scores shape: (batch * heads, seq, seq)

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

        # Softmax and value projection
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
        attn_output = torch.bmm(attn_weights, v_reshaped)

        # Reshape back: (batch, heads, seq, head_dim)
        attn_output = attn_output.view(batch_size, num_heads, seq_len, head_dim)
        return attn_output

    def _decode_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          block_table: torch.Tensor,
                          context_lens: torch.Tensor) -> torch.Tensor:
        """
        Compute attention during decode phase (with cache read).
        Matches HuggingFace BLOOM's attention implementation.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        dtype = q.dtype

        max_context_len = context_lens.max().item()

        if max_context_len > 0:
            # Gather K, V from cache
            k_cache, v_cache = self._gather_kv_from_cache(
                block_table, context_lens, max_context_len
            )
            # Concatenate cached and new K, V
            k_full = torch.cat([k_cache, k], dim=2)
            v_full = torch.cat([v_cache, v], dim=2)
        else:
            k_full = k
            v_full = v

        total_len = k_full.shape[2]

        # Reshape for batched matmul: (batch * heads, seq, head_dim)
        q_reshaped = q.reshape(batch_size * num_heads, seq_len, head_dim)
        k_reshaped = k_full.reshape(batch_size * num_heads, total_len, head_dim).transpose(-1, -2)
        v_reshaped = v_full.reshape(batch_size * num_heads, total_len, head_dim)

        # Build ALiBi tensor for full sequence: (batch * heads, 1, total_len)
        alibi = self._build_alibi_tensor(total_len, batch_size, device, dtype)

        # Compute attention scores with ALiBi: alibi + scale * (q @ k.T)
        scores = torch.baddbmm(
            alibi,
            q_reshaped,
            k_reshaped,
            beta=1.0,
            alpha=self.scale,
        )
        # scores shape: (batch * heads, seq_len, total_len)

        # Reshape for mask application: (batch, heads, seq_len, total_len)
        scores = scores.view(batch_size, num_heads, seq_len, total_len)

        # Causal mask: query at position i can only attend to keys at positions <= i
        # Query positions are context_lens, context_lens+1, ..., context_lens+seq_len-1
        query_positions = context_lens.view(batch_size, 1, 1) + torch.arange(seq_len, device=device).view(1, seq_len, 1)
        key_positions = torch.arange(total_len, device=device).view(1, 1, total_len)
        causal_mask = key_positions > query_positions  # (batch, seq_len, total_len)

        # Padding mask for cached K/V
        if max_context_len > 0:
            cache_positions = torch.arange(max_context_len, device=device).unsqueeze(0)
            padding_mask = cache_positions >= context_lens.unsqueeze(1)
            padding_mask = torch.cat([
                padding_mask,
                torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            ], dim=1)
        else:
            padding_mask = torch.zeros(batch_size, total_len, dtype=torch.bool, device=device)

        # Combine masks: (batch, 1, seq_len, total_len)
        attn_mask = causal_mask | padding_mask.unsqueeze(1)
        attn_mask = attn_mask.unsqueeze(1)  # (batch, 1, seq_len, total_len)
        scores = scores.masked_fill(attn_mask, float('-inf'))

        # Reshape back and apply softmax
        scores = scores.view(batch_size * num_heads, seq_len, total_len)
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
        attn_output = torch.bmm(attn_weights, v_reshaped)

        # Reshape back: (batch, heads, seq, head_dim)
        attn_output = attn_output.view(batch_size, num_heads, seq_len, head_dim)
        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        Forward pass with paged KV cache and ALiBi attention.

        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            attn_metadata: Attention metadata (slot_mapping, block_table, etc.)

        Returns:
            Output tensor (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Ensure cache is on same device/dtype as inputs
        if self.kv_cache.device != x.device or self.kv_cache.dtype != x.dtype:
            self.kv_cache = self.kv_cache.to(device=x.device, dtype=x.dtype)
        if self.alibi_slopes.device != x.device:
            self.alibi_slopes = self.alibi_slopes.to(device=x.device)

        # Fused QKV projection
        qkv = self.query_key_value(x)
        
        # Split into Q, K, V using HuggingFace's interleaved format:
        # fused_qkv is (batch, seq, num_heads * 3 * head_dim)
        # Reshape to (batch, seq, num_heads, 3, head_dim) where dim 3 indexes Q/K/V per head
        qkv = qkv.view(batch_size, seq_len, self.num_heads, 3, self.head_dim)
        q = qkv[..., 0, :].transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = qkv[..., 1, :].transpose(1, 2)
        v = qkv[..., 2, :].transpose(1, 2)

        # Write new K,V to cache
        self._write_to_cache(k, v, attn_metadata.slot_mapping)

        # Choose prefill or decode path
        if attn_metadata.is_prefill:
            attn_output = self._prefill_attention(q, k, v)
        else:
            attn_output = self._decode_attention(
                q, k, v,
                attn_metadata.block_table,
                attn_metadata.context_lens
            )

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.dense(attn_output)


class BloomMLP(nn.Module):
    """BLOOM MLP using level1 operators (GELU activation with BLOOM's exact formula)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense_h_to_4h = Linear(hidden_size, intermediate_size, bias=True)
        self.dense_4h_to_h = Linear(intermediate_size, hidden_size, bias=True)
        # Use HuggingFace BLOOM's exact GELU formula for perfect alignment
        self.gelu = GELU(approximate='tanh')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_h_to_4h(x)
        x = self.gelu(x)
        x = self.dense_4h_to_h(x)
        return x


class BloomDecoderLayer(nn.Module):
    """Single BLOOM decoder layer with paged attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        intermediate_size: int,
        max_seq_len: int = 2048,
        layer_norm_eps: float = 1e-5,
        apply_residual_connection_post_layernorm: bool = False,
        block_size: int = 16,
        num_blocks: int = 1024,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm

        self.input_layernorm = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self_attention = BloomAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            block_size=block_size,
            num_blocks=num_blocks,
            layer_idx=layer_idx,
        )
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = BloomMLP(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # Layer norm at the beginning of the transformer layer
        layernorm_output = self.input_layernorm(x)

        # Residual connection point depends on config
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = x

        # Self attention
        attention_output = self.self_attention(layernorm_output, attn_metadata)
        attention_output = attention_output + residual

        # Post-attention layer norm
        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual for MLP
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP with residual
        output = self.mlp(layernorm_output) + residual

        return output

    def reset_cache(self):
        """Reset this layer's KV cache."""
        self.self_attention.reset_cache()


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    BLOOM-style decoder-only transformer with Paged KV Cache.

    Aligned with vLLM's design:
    - Each decoder layer has its own KV cache
    - Uses AttentionMetadata for cache management
    - Supports both prefill and decode phases

    Uses level1 operators:
    - LayerNorm
    - ALiBi attention (via BloomAttention with built-in ALiBi)
    - Linear
    - GELU
    """

    def __init__(
        self,
        vocab_size: int = 250880,
        hidden_size: int = 4096,
        num_layers: int = 30,
        num_heads: int = 32,
        head_dim: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        max_seq_len: int = 2048,
        layer_norm_eps: float = 1e-5,
        apply_residual_connection_post_layernorm: bool = False,
        block_size: int = 16,
        num_blocks: int = 1024,
        **kwargs
    ):
        super().__init__()

        if head_dim is None:
            head_dim = hidden_size // num_heads

        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.layer_norm_eps = layer_norm_eps
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Token embedding
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)

        # Word embeddings LayerNorm (BLOOM-specific)
        self.word_embeddings_layernorm = LayerNorm(hidden_size, eps=layer_norm_eps)

        # Decoder layers - each with its own KV cache
        # Named 'h' to match HuggingFace BLOOM structure
        self.h = nn.ModuleList([
            BloomDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                max_seq_len=max_seq_len,
                layer_norm_eps=layer_norm_eps,
                apply_residual_connection_post_layernorm=apply_residual_connection_post_layernorm,
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
        # Embed tokens
        x = self.word_embeddings(input_ids)

        # BLOOM applies LayerNorm to embeddings
        x = self.word_embeddings_layernorm(x)

        # All layers share the same block_table but each has its own cache
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
