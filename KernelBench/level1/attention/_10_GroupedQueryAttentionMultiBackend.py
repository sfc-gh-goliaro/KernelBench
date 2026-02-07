"""
Grouped-Query Attention (GQA) with Multiple Attention Backend Support

This module provides a flexible GQA implementation that supports multiple
attention computation backends:
- sdpa: PyTorch's scaled_dot_product_attention (default)
- flash: Flash Attention 2 (requires flash_attn package)
- flashinfer: FlashInfer (requires flashinfer package)
- eager: Manual eager implementation (no fused kernels)

Used by: Llama-2 70B+, Llama-3, Mistral, Qwen-2

The backend can be configured at initialization time or changed dynamically.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Literal, List
from enum import Enum


# ============================================================================
# Attention Backend Enum
# ============================================================================

class AttentionBackend(Enum):
    """Supported attention backends."""
    SDPA = "sdpa"
    FLASH = "flash"
    FLASHINFER = "flashinfer"
    EAGER = "eager"


# ============================================================================
# Backend Availability Detection
# ============================================================================

def is_flash_attn_available() -> bool:
    """Check if Flash Attention 2 is available."""
    try:
        from flash_attn import flash_attn_func
        return True
    except ImportError:
        return False


def is_flashinfer_available() -> bool:
    """Check if FlashInfer is available."""
    try:
        import flashinfer
        return True
    except ImportError:
        return False


def get_available_backends() -> List[str]:
    """Return list of available attention backends."""
    available = ["sdpa", "eager"]
    if is_flash_attn_available():
        available.append("flash")
    if is_flashinfer_available():
        available.append("flashinfer")
    return available


# ============================================================================
# Attention Metadata (vLLM-style)
# ============================================================================

@dataclass
class AttentionMetadata:
    """
    Metadata for paged attention operations.
    
    Aligned with vLLM's CommonAttentionMetadata pattern.
    """
    slot_mapping: torch.Tensor      # (num_tokens,) - where to write new K,V
    block_table: torch.Tensor       # (batch_size, max_blocks_per_seq) - block assignments
    context_lens: torch.Tensor      # (batch_size,) - tokens already in cache
    seq_lens: torch.Tensor          # (batch_size,) - total sequence lengths (context + new)
    is_prefill: bool                # True if this is prefill phase


def create_attention_metadata(
    batch_size: int,
    seq_lens: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> AttentionMetadata:
    """
    Create attention metadata for paged attention.
    
    Args:
        batch_size: Number of sequences in batch
        seq_lens: (batch_size,) total sequence lengths (context + new tokens)
        context_lens: (batch_size,) tokens already in cache
        block_table: (batch_size, max_blocks_per_seq) block assignments
        block_size: Number of tokens per block
        device: Device for tensors
        
    Returns:
        AttentionMetadata with computed slot_mapping
    """
    # Compute slot_mapping: for each new token, where in the cache to write it
    # New tokens are at positions [context_len, context_len + 1, ..., seq_len - 1]
    slot_mappings = []
    
    for batch_idx in range(batch_size):
        context_len = context_lens[batch_idx].item()
        seq_len = seq_lens[batch_idx].item()
        
        for pos in range(context_len, seq_len):
            block_idx = pos // block_size
            block_offset = pos % block_size
            physical_block = block_table[batch_idx, block_idx].item()
            slot = physical_block * block_size + block_offset
            slot_mappings.append(slot)
    
    slot_mapping = torch.tensor(slot_mappings, dtype=torch.long, device=device)
    
    # Determine if this is prefill (all context_lens are 0 and seq_len > 1)
    # or decode (context_lens > 0 or seq_len == 1)
    is_prefill = (context_lens == 0).all().item() and (seq_lens > 1).any().item()
    
    return AttentionMetadata(
        slot_mapping=slot_mapping,
        block_table=block_table,
        context_lens=context_lens,
        seq_lens=seq_lens,
        is_prefill=is_prefill,
    )


# ============================================================================
# Backend Implementations
# ============================================================================

def _eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Eager (manual) attention implementation without fused kernels.
    
    Args:
        q: (batch, num_heads, seq_len, head_dim)
        k: (batch, num_heads, key_len, head_dim)
        v: (batch, num_heads, key_len, head_dim)
        scale: Attention scale factor (typically 1/sqrt(head_dim))
        is_causal: Whether to apply causal masking
        attn_mask: Optional attention mask (batch, 1, seq_len, key_len) or similar
        
    Returns:
        Output tensor (batch, num_heads, seq_len, head_dim)
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    key_len = k.shape[2]
    
    # Compute attention scores: Q @ K^T * scale
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Apply causal mask if needed
    if is_causal:
        # Create causal mask: positions can only attend to <= their position
        causal_mask = torch.triu(
            torch.ones(seq_len, key_len, dtype=torch.bool, device=q.device),
            diagonal=key_len - seq_len + 1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
    
    # Apply additional attention mask if provided
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    
    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    
    # Weighted sum: attn_weights @ V
    output = torch.matmul(attn_weights, v)
    
    return output


def _flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Flash Attention 2 backend.
    
    Args:
        q: (batch, num_heads, seq_len, head_dim)
        k: (batch, num_heads, key_len, head_dim)
        v: (batch, num_heads, key_len, head_dim)
        scale: Attention scale factor
        is_causal: Whether to apply causal masking
        
    Returns:
        Output tensor (batch, num_heads, seq_len, head_dim)
    """
    from flash_attn import flash_attn_func
    
    # Flash attention expects (batch, seq_len, num_heads, head_dim)
    q = q.transpose(1, 2)  # (batch, seq_len, num_heads, head_dim)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Call flash attention
    output = flash_attn_func(
        q, k, v,
        softmax_scale=scale,
        causal=is_causal,
    )
    
    # Transpose back to (batch, num_heads, seq_len, head_dim)
    output = output.transpose(1, 2)
    
    return output


def _sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    PyTorch SDPA (Scaled Dot-Product Attention) backend.
    
    Args:
        q: (batch, num_heads, seq_len, head_dim)
        k: (batch, num_heads, key_len, head_dim)
        v: (batch, num_heads, key_len, head_dim)
        scale: Attention scale factor
        is_causal: Whether to apply causal masking
        attn_mask: Optional attention mask
        
    Returns:
        Output tensor (batch, num_heads, seq_len, head_dim)
    """
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal if attn_mask is None else False,
        scale=scale,
    )


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Grouped-Query Attention (GQA) with Multiple Attention Backends and Paged KV Cache

    Used by: Llama-2 70B+, Llama-3, Mistral, Qwen-2

    GQA reduces memory and compute by having multiple query heads share
    the same key/value heads. Uses paged KV cache where cache entries
    are stored in non-contiguous blocks accessed via a page table.
    
    This implementation owns its KV cache (per-layer cache pattern from vLLM).
    
    Supports multiple attention backends:
    - sdpa: PyTorch's scaled_dot_product_attention (default, good balance)
    - flash: Flash Attention 2 (fastest for long sequences)
    - flashinfer: FlashInfer (optimized for LLM serving)
    - eager: Manual implementation (for debugging/baseline)

    Shapes:
        q: (batch_size, num_heads, seq_len, head_dim) - projected query
        k: (batch_size, num_kv_heads, seq_len, head_dim) - projected key (new tokens)
        v: (batch_size, num_kv_heads, seq_len, head_dim) - projected value (new tokens)
        Output: (batch_size, num_heads, seq_len, head_dim)
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 1024,
        max_seq_len: int = 8192,
        dropout: float = 0.0,
        attn_backend: str = "sdpa",
    ):
        """
        Initialize GQA with paged KV cache and configurable attention backend.

        Args:
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (must divide num_heads)
            head_dim: Dimension of each attention head
            block_size: Number of tokens per cache block/page
            num_blocks: Total number of blocks in the cache pool
            max_seq_len: Maximum sequence length
            dropout: Attention dropout probability
            attn_backend: Attention backend to use ("sdpa", "flash", "flashinfer", "eager")
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Set attention backend
        self.set_attn_backend(attn_backend)
        
        # Initialize paged KV cache as buffer (not parameter - no gradients)
        # Shape: (num_blocks, block_size, num_kv_heads, head_dim, 2) where 2 = [K, V]
        self.register_buffer('kv_cache', torch.zeros(
            num_blocks, block_size, num_kv_heads, head_dim, 2
        ))
        
        # FlashInfer state (lazily initialized)
        self._flashinfer_prefill_wrapper = None
        self._flashinfer_decode_wrapper = None
        self._flashinfer_workspace = None

    def set_attn_backend(self, backend: str) -> None:
        """
        Set the attention backend to use.
        
        Args:
            backend: One of "sdpa", "flash", "flashinfer", "eager"
            
        Raises:
            ValueError: If backend is not available
        """
        available = get_available_backends()
        if backend not in available:
            raise ValueError(
                f"Attention backend '{backend}' is not available. "
                f"Available backends: {available}"
            )
        self.attn_backend = backend

    def get_attn_backend(self) -> str:
        """Get the current attention backend."""
        return self.attn_backend

    def reset_cache(self):
        """Reset the KV cache to zeros."""
        self.kv_cache.zero_()
        # Reset FlashInfer state
        self._flashinfer_prefill_wrapper = None
        self._flashinfer_decode_wrapper = None

    def _write_to_cache(self, k: torch.Tensor, v: torch.Tensor,
                        slot_mapping: torch.Tensor) -> None:
        """
        Write new K,V to cache using slot_mapping.
        
        Args:
            k: (batch_size, num_kv_heads, seq_len, head_dim)
            v: (batch_size, num_kv_heads, seq_len, head_dim)
            slot_mapping: (num_tokens,) - absolute positions in cache
        """
        if slot_mapping.numel() == 0:
            return
            
        # Flatten K,V to (num_tokens, num_kv_heads, head_dim)
        k_flat = k.transpose(1, 2).reshape(-1, self.num_kv_heads, self.head_dim)
        v_flat = v.transpose(1, 2).reshape(-1, self.num_kv_heads, self.head_dim)
        
        # Compute block indices and offsets from slot_mapping
        block_indices = slot_mapping // self.block_size
        block_offsets = slot_mapping % self.block_size
        
        # Write to cache using advanced indexing
        self.kv_cache[block_indices, block_offsets, :, :, 0] = k_flat
        self.kv_cache[block_indices, block_offsets, :, :, 1] = v_flat

    def _gather_kv_from_cache_truncated(self, block_table: torch.Tensor,
                                         context_lens: torch.Tensor,
                                         max_context_len: int) -> tuple:
        """
        Gather K and V tensors from paged cache, truncated to max_context_len.

        Args:
            block_table: (batch_size, max_blocks_per_seq)
            context_lens: (batch_size,)
            max_context_len: Maximum context length to gather

        Returns:
            k_cache: (batch_size, num_kv_heads, max_context_len, head_dim)
            v_cache: (batch_size, num_kv_heads, max_context_len, head_dim)
        """
        batch_size = block_table.shape[0]
        device = self.kv_cache.device
        
        # Calculate how many blocks we need
        num_blocks_needed = (max_context_len + self.block_size - 1) // self.block_size
        
        # Gather only the blocks we need
        block_table_truncated = block_table[:, :num_blocks_needed]
        
        gathered_blocks = self.kv_cache[block_table_truncated.flatten()]
        gathered_blocks = gathered_blocks.view(
            batch_size, num_blocks_needed, self.block_size, self.num_kv_heads, self.head_dim, 2
        )
        
        # Reshape to (batch, num_blocks * block_size, kv_heads, head_dim, 2)
        gathered = gathered_blocks.view(
            batch_size, num_blocks_needed * self.block_size, self.num_kv_heads, self.head_dim, 2
        )
        
        # Truncate to exactly max_context_len
        gathered = gathered[:, :max_context_len, :, :, :]

        # Split K and V, transpose to (batch, kv_heads, context, head_dim)
        k_cache = gathered[..., 0].transpose(1, 2)
        v_cache = gathered[..., 1].transpose(1, 2)

        return k_cache, v_cache

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention using the configured backend.
        
        Args:
            q: (batch, num_heads, seq_len, head_dim)
            k: (batch, num_heads, key_len, head_dim) - already expanded for GQA
            v: (batch, num_heads, key_len, head_dim) - already expanded for GQA
            is_causal: Whether to apply causal masking
            attn_mask: Optional attention mask
            
        Returns:
            Output tensor (batch, num_heads, seq_len, head_dim)
        """
        if self.attn_backend == "sdpa":
            return _sdpa_attention(q, k, v, self.scale, is_causal, attn_mask)
        elif self.attn_backend == "flash":
            if attn_mask is not None:
                # Flash attention doesn't support arbitrary masks well,
                # fall back to SDPA for masked attention
                return _sdpa_attention(q, k, v, self.scale, is_causal, attn_mask)
            return _flash_attention(q, k, v, self.scale, is_causal)
        elif self.attn_backend == "eager":
            return _eager_attention(q, k, v, self.scale, is_causal, attn_mask)
        elif self.attn_backend == "flashinfer":
            # FlashInfer for simple cases, fall back to SDPA for complex masks
            if attn_mask is not None:
                return _sdpa_attention(q, k, v, self.scale, is_causal, attn_mask)
            return self._flashinfer_attention(q, k, v, is_causal)
        else:
            raise ValueError(f"Unknown attention backend: {self.attn_backend}")

    def _flashinfer_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        FlashInfer attention backend.
        
        Uses single_prefill_with_kv_cache for simplicity.
        For production, use BatchPrefillWithPagedKVCacheWrapper.
        """
        import flashinfer
        
        batch_size, num_heads, seq_len, head_dim = q.shape
        key_len = k.shape[2]
        
        # FlashInfer expects (batch * seq_len, num_heads, head_dim) format
        # For simple cases, use single_prefill_with_kv_cache
        outputs = []
        for b in range(batch_size):
            q_b = q[b].transpose(0, 1).contiguous()  # (seq_len, num_heads, head_dim)
            k_b = k[b].transpose(0, 1).contiguous()  # (key_len, num_heads, head_dim)
            v_b = v[b].transpose(0, 1).contiguous()
            
            out_b = flashinfer.single_prefill_with_kv_cache(
                q_b, k_b, v_b,
                causal=is_causal,
                sm_scale=self.scale,
            )
            outputs.append(out_b.transpose(0, 1))  # Back to (num_heads, seq_len, head_dim)
        
        return torch.stack(outputs, dim=0)  # (batch, num_heads, seq_len, head_dim)

    def _prefill_attention(self, q: torch.Tensor, k: torch.Tensor, 
                           v: torch.Tensor) -> torch.Tensor:
        """
        Compute attention during prefill phase (no cache read needed).
        
        All tokens are new, so we only compute attention on the input K,V.
        
        Args:
            q: (batch_size, num_heads, seq_len, head_dim)
            k: (batch_size, num_kv_heads, seq_len, head_dim)
            v: (batch_size, num_kv_heads, seq_len, head_dim)
            
        Returns:
            Output tensor (batch_size, num_heads, seq_len, head_dim)
        """
        # Repeat K, V to match number of query heads for attention
        k_expanded = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_expanded = v.repeat_interleave(self.num_key_value_groups, dim=1)

        return self._compute_attention(q, k_expanded, v_expanded, is_causal=True)

    def _decode_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          block_table: torch.Tensor, 
                          context_lens: torch.Tensor) -> torch.Tensor:
        """
        Compute attention during decode phase (with cache read).
        
        Args:
            q: (batch_size, num_heads, seq_len, head_dim) - typically seq_len=1
            k: (batch_size, num_kv_heads, seq_len, head_dim) - new tokens
            v: (batch_size, num_kv_heads, seq_len, head_dim) - new tokens
            block_table: (batch_size, max_blocks_per_seq)
            context_lens: (batch_size,) - tokens already in cache (before new tokens)
            
        Returns:
            Output tensor (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        dtype = q.dtype

        # For decode, we need to handle variable context lengths per sequence
        max_context_len = context_lens.max().item()
        
        if max_context_len > 0:
            # Gather K, V from cache
            k_cache, v_cache = self._gather_kv_from_cache_truncated(
                block_table, context_lens, max_context_len
            )
            
            # Concatenate cached and new K, V
            k_full = torch.cat([k_cache, k], dim=2)
            v_full = torch.cat([v_cache, v], dim=2)
        else:
            k_full = k
            v_full = v

        # Repeat K, V to match number of query heads
        k_full = k_full.repeat_interleave(self.num_key_value_groups, dim=1)
        v_full = v_full.repeat_interleave(self.num_key_value_groups, dim=1)

        total_len = k_full.shape[2]

        # Create attention mask for decode
        query_positions = torch.arange(seq_len, device=device).unsqueeze(0) + context_lens.unsqueeze(1)
        key_positions = torch.arange(total_len, device=device).unsqueeze(0)
        
        # Causal mask
        causal_mask = key_positions > query_positions.unsqueeze(-1)
        
        # Padding mask for cached K/V
        if max_context_len > 0:
            cache_positions = torch.arange(max_context_len, device=device).unsqueeze(0)
            padding_mask = cache_positions >= context_lens.unsqueeze(1)
            padding_mask = torch.cat([
                padding_mask,
                torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            ], dim=1)
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
        else:
            padding_mask = torch.zeros(batch_size, 1, 1, total_len, dtype=torch.bool, device=device)
        
        # Combine masks
        causal_mask = causal_mask.unsqueeze(1)
        bool_mask = causal_mask | padding_mask

        # Convert to float mask
        attn_mask = torch.zeros(batch_size, 1, seq_len, total_len, dtype=dtype, device=device)
        attn_mask = attn_mask.masked_fill(bool_mask, float('-inf'))

        return self._compute_attention(q, k_full, v_full, is_causal=False, attn_mask=attn_mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attn_metadata: AttentionMetadata) -> torch.Tensor:
        """
        Forward pass with paged KV cache and grouped-query attention.

        Args:
            q: Projected query (batch_size, num_heads, seq_len, head_dim)
            k: Projected key for new tokens (batch_size, num_kv_heads, seq_len, head_dim)
            v: Projected value for new tokens (batch_size, num_kv_heads, seq_len, head_dim)
            attn_metadata: Attention metadata containing slot_mapping, block_table, etc.

        Returns:
            Output tensor (batch_size, num_heads, seq_len, head_dim)
        """
        # Ensure cache is on same device/dtype as inputs
        if self.kv_cache.device != q.device or self.kv_cache.dtype != q.dtype:
            self.kv_cache = self.kv_cache.to(device=q.device, dtype=q.dtype)
        
        # Write new K,V to cache
        self._write_to_cache(k, v, attn_metadata.slot_mapping)
        
        # Choose prefill or decode path
        if attn_metadata.is_prefill:
            return self._prefill_attention(q, k, v)
        else:
            return self._decode_attention(
                q, k, v, 
                attn_metadata.block_table, 
                attn_metadata.context_lens
            )


# ============================================================================
# Convenience Functions
# ============================================================================

def list_available_backends() -> List[str]:
    """
    List all available attention backends on the current system.
    
    Returns:
        List of available backend names
    """
    return get_available_backends()


def get_backend_info() -> dict:
    """
    Get detailed information about available backends.
    
    Returns:
        Dict with backend availability and version info
    """
    info = {
        "sdpa": {
            "available": True,
            "version": torch.__version__,
            "description": "PyTorch Scaled Dot-Product Attention",
        },
        "eager": {
            "available": True,
            "version": "N/A",
            "description": "Manual eager implementation (no fused kernels)",
        },
        "flash": {
            "available": is_flash_attn_available(),
            "version": None,
            "description": "Flash Attention 2",
        },
        "flashinfer": {
            "available": is_flashinfer_available(),
            "version": None,
            "description": "FlashInfer",
        },
    }
    
    if info["flash"]["available"]:
        try:
            import flash_attn
            info["flash"]["version"] = getattr(flash_attn, "__version__", "unknown")
        except:
            pass
    
    if info["flashinfer"]["available"]:
        try:
            import flashinfer
            info["flashinfer"]["version"] = getattr(flashinfer, "__version__", "unknown")
        except:
            pass
    
    return info


# ============================================================================
# Benchmark Configuration
# ============================================================================
