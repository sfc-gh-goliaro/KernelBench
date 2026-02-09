"""
Multi-Head Latent Attention (MLA)

Used by: DeepSeek-V2, DeepSeek-V2-Lite, DeepSeek-V3

MLA compresses KV into a low-rank latent space before caching,
reducing KV cache memory while maintaining model quality.

Key insight: Instead of caching full K,V tensors, we cache the compressed
latent representation. The K,V are recomputed on-the-fly via up-projections.

This implementation uses SDPA (Scaled Dot-Product Attention) for numerical
alignment with HuggingFace implementations.

Architecture:
    - Q projection (either direct or low-rank via q_lora_rank)
    - KV compression (kv_a_proj_with_mqa) to latent space
    - KV up-projection (kv_b_proj) to full K_nope and V
    - RoPE applied to rope portions of Q and K
    - SDPA for attention computation
    - Output projection

Shapes:
    Input: (batch_size, seq_len, hidden_size)
    Output: (batch_size, seq_len, hidden_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


# ============================================================================
# Attention Metadata (vLLM-style)
# ============================================================================

@dataclass
class AttentionMetadata:
    """
    Metadata for attention operations.
    
    Aligned with vLLM's CommonAttentionMetadata pattern.
    """
    slot_mapping: torch.Tensor      # (num_tokens,) - where to write new cache entries
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
    """
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
    
    is_prefill = (context_lens == 0).all().item() and (seq_lens > 1).any().item()
    
    return AttentionMetadata(
        slot_mapping=slot_mapping,
        block_table=block_table,
        context_lens=context_lens,
        seq_lens=seq_lens,
        is_prefill=is_prefill,
    )


# ============================================================================
# RoPE Functions (HuggingFace-compatible complex number approach)
# ============================================================================

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings using complex number multiplication.
    
    This matches HuggingFace's exact implementation for numerical alignment.
    
    Args:
        xq: Query tensor (batch, heads, seq, head_dim)
        xk: Key tensor (batch, heads, seq, head_dim)
        freqs_cis: Complex frequencies (batch, seq, head_dim//2)
        
    Returns:
        Rotated (xq, xk) with same shapes
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast to [batch, 1, seq_len, dim // 2]
    freqs_cis = freqs_cis.unsqueeze(1).to(xq_.device)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)
    return xq_out, xk_out


class RotaryEmbedding(nn.Module):
    """
    Rotary Embedding with YARN scaling.
    
    Matches HuggingFace's exact implementation using complex polar representation.
    
    Note: inv_freq must stay in float32 for numerical precision, even when the
    model is converted to bfloat16. This matches HuggingFace's behavior.
    """
    
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 163840,
        base: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.rope_scaling = rope_scaling
        
        # Compute inverse frequencies (keep in float32 for precision)
        # Store as a regular attribute to avoid dtype conversion when model.to() is called
        self._inv_freq_float32 = self._compute_inv_freq()
        # Register a placeholder buffer for device tracking
        self.register_buffer("inv_freq", self._inv_freq_float32.clone(), persistent=False)
        
        # Compute attention scaling (for YARN)
        self.attention_scaling = self._compute_attention_scaling()
    
    def _float_dict_key(self, d, key, default):
        """Get a key from dict, converting to float if needed."""
        val = d.get(key, default)
        return float(val) if val is not None else default
    
    def _apply(self, fn):
        """Override to keep inv_freq in float32 when model dtype changes."""
        # Apply function to all parameters and buffers
        super()._apply(fn)
        
        # Restore inv_freq from the original float32 values
        # This preserves precision that would be lost during bfloat16 conversion
        if hasattr(self, '_inv_freq_float32'):
            # Get target device from the converted buffer
            target_device = self.inv_freq.device
            # Copy the original float32 values to the target device
            self.inv_freq = self._inv_freq_float32.to(device=target_device)
            self._inv_freq_float32 = self._inv_freq_float32.to(device=target_device)
        return self
    
    def _compute_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequencies, with optional YARN scaling."""
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim)
        )
        
        if self.rope_scaling is not None:
            rope_type = self.rope_scaling.get("rope_type", self.rope_scaling.get("type", "default"))
            if rope_type == "yarn":
                # YARN scaling
                factor = float(self.rope_scaling.get("factor", 1.0))
                original_max_pos = self.rope_scaling.get("original_max_position_embeddings", 4096)
                beta_fast = float(self.rope_scaling.get("beta_fast", 32))
                beta_slow = float(self.rope_scaling.get("beta_slow", 1))
                
                # Compute YARN-scaled frequencies
                dim = self.head_dim
                
                def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
                    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))
                
                def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
                    low = max(math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings)), 0)
                    high = min(math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings)), dim - 1)
                    return low, high
                
                def linear_ramp_mask(min_val, max_val, dim):
                    if min_val == max_val:
                        max_val += 0.001
                    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
                    return torch.clamp(linear_func, 0, 1)
                
                low, high = find_correction_range(beta_fast, beta_slow, dim, self.base, original_max_pos)
                inv_freq_mask = 1.0 - linear_ramp_mask(low, high, dim // 2)
                inv_freq = inv_freq / factor * (1 - inv_freq_mask) + inv_freq * inv_freq_mask
        
        return inv_freq
    
    def _compute_attention_scaling(self) -> float:
        """Compute attention scaling factor for YARN.
        
        Matches HuggingFace's implementation in modeling_rope_utils.py.
        """
        if self.rope_scaling is None:
            return 1.0
            
        rope_type = self.rope_scaling.get("rope_type", self.rope_scaling.get("type", "default"))
        if rope_type != "yarn":
            return 1.0
        
        # Get attention_factor if explicitly provided
        attention_factor = self.rope_scaling.get("attention_factor")
        if attention_factor is not None:
            return float(attention_factor)
            
        # Otherwise compute from mscale/mscale_all_dim
        mscale = self._float_dict_key(self.rope_scaling, "mscale", None)
        mscale_all_dim = self._float_dict_key(self.rope_scaling, "mscale_all_dim", None)
        factor = self._float_dict_key(self.rope_scaling, "factor", 1.0)
        
        def get_mscale(scale, mscale_val=1.0):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale_val * math.log(scale) + 1.0
        
        # Compute attention_factor as in HuggingFace
        if mscale is not None and mscale_all_dim is not None:
            return float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
        elif mscale is not None:
            return get_mscale(factor, mscale)
        else:
            return get_mscale(factor)
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute rotary embeddings for given positions.
        
        Args:
            x: Input tensor (batch, seq, dim) - only used for device/dtype
            position_ids: Position indices (batch, seq)
            
        Returns:
            Complex frequency tensor for rotary embedding
        """
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        freqs_cis = freqs_cis * self.attention_scaling
        
        return freqs_cis


# ============================================================================
# RMSNorm
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Matches HuggingFace's DeepseekV2RMSNorm implementation exactly.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ============================================================================
# Multi-Head Latent Attention
# ============================================================================

class Model(nn.Module):
    """
    Multi-Head Latent Attention (MLA).
    
    Complete attention module with:
    - Q projection (with optional low-rank)
    - KV compression and up-projection
    - RoPE (with YARN scaling support)
    - SDPA for attention computation
    - Simple KV caching for decode phase
    
    Uses SDPA for numerical alignment with HuggingFace implementations.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        q_lora_rank: Optional[int] = None,
        max_seq_len: int = 163840,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        block_size: int = 16,
        num_blocks: int = 1024,
        layer_idx: int = 0,
    ):
        """
        Initialize Multi-Head Latent Attention.

        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            qk_nope_head_dim: Non-RoPE dimension for Q/K
            qk_rope_head_dim: RoPE dimension for Q/K
            v_head_dim: Value head dimension
            kv_lora_rank: Rank for KV compression (latent dimension)
            q_lora_rank: Optional rank for Q low-rank projection
            max_seq_len: Maximum sequence length
            rope_theta: RoPE theta parameter
            rope_scaling: Optional YARN scaling config
            block_size: Block size for paged cache (unused currently)
            num_blocks: Number of cache blocks (unused currently)
            layer_idx: Layer index (for debugging)
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.layer_idx = layer_idx
        self.scaling = self.qk_head_dim ** (-0.5)

        # Q projection (either direct or low-rank)
        if q_lora_rank is None:
            self.q_proj = nn.Linear(hidden_size, num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(q_lora_rank, eps=1e-6)
            self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)

        # KV projection with MQA-style fusion (compressed + rope)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            bias=False
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=1e-6)
        
        # KV up-projection: from compressed to full K_nope and V
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False
        )

        # Output projection
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)

        # RoPE with complex representation
        self.rotary_emb = RotaryEmbedding(
            head_dim=qk_rope_head_dim,
            max_seq_len=max_seq_len,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        
        # Simple KV cache for decode phase
        self.k_cache = None
        self.v_cache = None

    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        Forward pass with KV caching support.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            position_ids: Position indices (batch_size, seq_len)
            attn_metadata: Attention metadata
            
        Returns:
            Output tensor (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Q projection (with optional low-rank)
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV projection
        compressed_kv = self.kv_a_proj_with_mqa(x)
        k_nope_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # Up-project compressed KV to get k_nope and v
        kv_proj = self.kv_b_proj(self.kv_a_layernorm(k_nope_compressed))
        kv_proj = kv_proj.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value_states = torch.split(kv_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Reshape k_pe for RoPE: (batch, 1, seq, rope_dim)
        k_pe = k_pe.view(batch_size, 1, seq_len, self.qk_rope_head_dim)
        
        # Compute rotary embeddings (complex polar representation)
        freqs_cis = self.rotary_emb(x, position_ids)
        
        # Apply RoPE using complex multiplication
        q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, freqs_cis)
        
        # Expand k_pe to all heads
        k_pe = k_pe.expand(*k_nope.shape[:-1], -1)
        
        # Concatenate nope and pe components
        query_states = torch.cat((q_nope, q_pe), dim=-1)
        key_states = torch.cat((k_nope, k_pe), dim=-1)

        # Handle KV caching for decode phase
        if attn_metadata.is_prefill:
            # During prefill, store KV in cache
            self.k_cache = key_states
            self.v_cache = value_states
        else:
            # During decode, concatenate with cached KV
            if self.k_cache is not None:
                key_states = torch.cat([self.k_cache, key_states], dim=2)
                value_states = torch.cat([self.v_cache, value_states], dim=2)
            # Update cache with new KV
            self.k_cache = key_states
            self.v_cache = value_states

        # Scaled dot-product attention using SDPA for numerical alignment with HuggingFace
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            scale=self.scaling,
            is_causal=attn_metadata.is_prefill,  # Causal only during prefill
        )

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)

    def reset_cache(self):
        """Reset this layer's KV cache."""
        self.k_cache = None
        self.v_cache = None


# ============================================================================
# Benchmark Configuration
# ============================================================================

def get_inputs():
    """Generate inputs for benchmarking."""
    batch_size = 1
    seq_len = 512
    hidden_size = 2048
    num_heads = 16
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    kv_lora_rank = 512
    
    device = "cuda"
    dtype = torch.bfloat16
    
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    
    block_table = torch.arange(64, device=device, dtype=torch.long).unsqueeze(0)
    context_lens = torch.zeros(batch_size, dtype=torch.long, device=device)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
    
    attn_metadata = create_attention_metadata(
        batch_size=batch_size,
        seq_lens=seq_lens,
        context_lens=context_lens,
        block_table=block_table,
        block_size=16,
        device=device,
    )
    
    return [x, position_ids, attn_metadata]


def get_init_inputs():
    """Get initialization parameters for MLA."""
    return {
        "hidden_size": 2048,
        "num_heads": 16,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "kv_lora_rank": 512,
        "q_lora_rank": None,
        "max_seq_len": 8192,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "block_size": 16,
        "num_blocks": 1024,
        "layer_idx": 0,
    }
