"""
Rotary Position Embedding (RoPE)

Used by: Llama, Qwen, Mistral, Gemma, Yi, DeepSeek, Phi

Applies rotary position embeddings to query and key tensors by rotating
pairs of dimensions using precomputed cos/sin values based on position.

Supports multiple RoPE types:
- "default": Standard RoPE
- "llama3": Llama-3.1 style with piecewise frequency scaling
- "yarn": YaRN (Yet another RoPE extensioN) for extended context

Shapes (depends on layout parameter):
    layout="bshd": (batch_size, seq_len, num_heads, head_dim) - default
    layout="bhsd": (batch_size, num_heads, seq_len, head_dim) - Llama attention style
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import math
import torch
import torch.nn as nn
from typing import Optional, Literal, Dict, Any


def compute_default_inv_freq(
    head_dim: int,
    base: float,
    device: torch.device = None,
) -> torch.Tensor:
    """Compute standard RoPE inverse frequencies."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim)
    )
    return inv_freq


def compute_llama3_inv_freq(
    head_dim: int,
    base: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute Llama-3.1 style inverse frequencies with piecewise scaling.
    
    This implements the llama3 RoPE scaling from the transformers library:
    - For wavelengths < high_freq_wavelen: keep inv_freq unchanged
    - For wavelengths > low_freq_wavelen: divide inv_freq by factor
    - For wavelengths in between: smoothly interpolate
    
    Args:
        head_dim: Dimension of each attention head
        base: Base frequency (rope_theta)
        factor: Scaling factor (typically 8.0 for Llama-3.1)
        low_freq_factor: Low frequency factor (typically 1.0)
        high_freq_factor: High frequency factor (typically 4.0)
        original_max_position_embeddings: Original context length (typically 8192)
        device: Device for tensor creation
        
    Returns:
        Scaled inverse frequencies
    """
    # Compute base inverse frequencies
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim)
    )
    
    old_context_len = original_max_position_embeddings
    
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    
    wavelen = 2 * math.pi / inv_freq
    
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    
    is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    
    return inv_freq_llama


def yarn_find_correction_dim(
    num_rotations: float, 
    dim: int, 
    base: float = 10000, 
    max_position_embeddings: int = 2048
) -> float:
    """Find the dimension for YaRN correction based on number of rotations."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def yarn_find_correction_range(
    low_rot: float, 
    high_rot: float, 
    dim: int, 
    base: float = 10000, 
    max_position_embeddings: int = 2048
) -> tuple:
    """Find the dimension range for YaRN correction."""
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
    """Create a linear ramp mask for YaRN interpolation."""
    if min_val == max_val:
        max_val += 0.001  # Prevent singularity
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """Compute the mscale factor for YaRN attention scaling."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def compute_yarn_inv_freq(
    head_dim: int,
    base: float,
    factor: float,
    original_max_position_embeddings: int,
    beta_fast: float = 32,
    beta_slow: float = 1,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute YaRN (Yet another RoPE extensioN) inverse frequencies.
    
    YaRN uses a combination of extrapolation and interpolation for different
    frequency dimensions, blended using a linear ramp.
    
    Args:
        head_dim: Dimension of each attention head
        base: Base frequency (rope_theta)
        factor: Scaling factor for context extension
        original_max_position_embeddings: Original max position embeddings
        beta_fast: Fast beta for correction range
        beta_slow: Slow beta for correction range
        device: Device for tensor creation
        
    Returns:
        Scaled inverse frequencies
    """
    dim = head_dim
    
    # Extrapolation frequencies (no scaling)
    freq_extra = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    
    # Interpolation frequencies (scaled by factor)
    freq_inter = 1.0 / (
        factor * base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    
    # Find correction range
    low, high = yarn_find_correction_range(
        beta_fast, beta_slow, dim, base, original_max_position_embeddings
    )
    
    # Create linear ramp mask
    inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(device)
    
    # Blend extra and inter frequencies
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
    
    return inv_freq


class Model(nn.Module):
    """
    Rotary Position Embedding (RoPE) with support for various scaling methods.
    
    Supports:
    - Standard RoPE (rope_type="default")
    - Llama-3.1 RoPE scaling (rope_type="llama3")
    - YaRN RoPE scaling (rope_type="yarn")
    """
    
    def __init__(
        self, 
        head_dim: int, 
        max_seq_len: int = 8192, 
        base: float = 10000.0,
        layout: Literal["bshd", "bhsd"] = "bshd",
        rope_scaling: Optional[Dict[str, Any]] = None,
        interleaved: bool = False,
    ):
        """
        Initialize RoPE.
        
        Args:
            head_dim: Dimension of each attention head (must be even)
            max_seq_len: Maximum sequence length for precomputed embeddings
            base: Base for the frequency computation (rope_theta)
            layout: Input tensor layout. 
                    "bshd" = (batch, seq, heads, head_dim) - default
                    "bhsd" = (batch, heads, seq, head_dim) - Llama attention style
            rope_scaling: Optional RoPE scaling config dict with keys:
                    - rope_type/type: "default", "llama3", or "yarn"
                    - factor: Scaling factor
                    - For llama3: low_freq_factor, high_freq_factor, original_max_position_embeddings
                    - For yarn: beta_fast, beta_slow, original_max_position_embeddings
            interleaved: If True, apply interleaving transformation before RoPE (used by DeepSeek).
                        This transforms [x0,x1,x2,x3,...] to [x0,x2,...,x1,x3,...] format.
        """
        super(Model, self).__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.layout = layout
        self.rope_scaling = rope_scaling
        self.interleaved = interleaved
        
        # Determine rope type - support both "rope_type" and "type" keys
        rope_type = "default"
        if rope_scaling is not None:
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        
        self.rope_type = rope_type
        
        # Compute inverse frequencies based on rope type
        if rope_type == "llama3":
            inv_freq = compute_llama3_inv_freq(
                head_dim=head_dim,
                base=base,
                factor=rope_scaling["factor"],
                low_freq_factor=rope_scaling["low_freq_factor"],
                high_freq_factor=rope_scaling["high_freq_factor"],
                original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
            )
        elif rope_type == "yarn":
            inv_freq = compute_yarn_inv_freq(
                head_dim=head_dim,
                base=base,
                factor=rope_scaling["factor"],
                original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
                beta_fast=rope_scaling.get("beta_fast", 32),
                beta_slow=rope_scaling.get("beta_slow", 1),
            )
            # Store YaRN mscale parameters
            self.yarn_factor = rope_scaling["factor"]
            self.yarn_mscale = rope_scaling.get("mscale", 1.0)
            self.yarn_mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
        else:
            inv_freq = compute_default_inv_freq(head_dim, base)
        
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for all positions
        self._update_cos_sin_cache(max_seq_len)
    
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device = None):
        """Update the cos/sin cache for the given sequence length."""
        if device is None:
            device = self.inv_freq.device
            
        t = torch.arange(seq_len, device=device, dtype=torch.float)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        
        # Apply YaRN mscale if applicable
        if self.rope_type == "yarn" and hasattr(self, 'yarn_mscale'):
            mscale = yarn_get_mscale(self.yarn_factor, self.yarn_mscale)
            mscale_all_dim = yarn_get_mscale(self.yarn_factor, self.yarn_mscale_all_dim)
            if mscale_all_dim > 0:
                _mscale = mscale / mscale_all_dim
            else:
                _mscale = mscale
            cos = cos * _mscale
            sin = sin * _mscale
        
        self.register_buffer('cos_cached', cos, persistent=False)
        self.register_buffer('sin_cached', sin, persistent=False)
        self.max_seq_len = seq_len
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> tuple:
        """
        Apply rotary embeddings to q and k.
        
        Args:
            q: Query tensor. Shape depends on layout:
               - "bshd": (batch_size, seq_len, num_heads, head_dim)
               - "bhsd": (batch_size, num_heads, seq_len, head_dim)
            k: Key tensor with same layout as q
            position_ids: Optional position indices of shape (batch_size, seq_len)
            
        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as inputs
        """
        # Get seq_len based on layout
        if self.layout == "bhsd":
            seq_len = q.shape[2]  # (batch, heads, seq, head_dim)
        else:
            seq_len = q.shape[1]  # (batch, seq, heads, head_dim)
        
        # Extend cache if needed
        if position_ids is not None:
            max_pos = position_ids.max().item() + 1
            if max_pos > self.max_seq_len:
                self._update_cos_sin_cache(max_pos, q.device)
        elif seq_len > self.max_seq_len:
            self._update_cos_sin_cache(seq_len, q.device)
        
        if position_ids is None:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        
        # Reshape for broadcasting based on layout
        if self.layout == "bhsd":
            # For (batch, heads, seq, head_dim): broadcast shape is (1, 1, seq, head_dim)
            if position_ids is None:
                cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
                sin = sin.unsqueeze(0).unsqueeze(0)
            else:
                cos = cos.unsqueeze(1)  # (batch, 1, seq, head_dim)
                sin = sin.unsqueeze(1)
        else:
            # For (batch, seq, heads, head_dim): broadcast shape is (1, seq, 1, head_dim)
            if position_ids is None:
                cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim)
                sin = sin.unsqueeze(0).unsqueeze(2)
            else:
                cos = cos.unsqueeze(2)  # (batch, seq, 1, head_dim)
                sin = sin.unsqueeze(2)
        
        # Convert cos/sin to input dtype to preserve precision
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        
        return q_rotated, k_rotated
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to a single tensor."""
        # Apply interleaving transformation if needed (used by DeepSeek)
        # This transforms [x0, x1, x2, x3, ...] to [x0, x2, ..., x1, x3, ...]
        if self.interleaved:
            d = x.shape[-1]
            x = x.view(*x.shape[:-1], d // 2, 2).transpose(-1, -2).reshape(*x.shape[:-1], d)
        
        # Split into two halves
        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]
        
        # Rotate (matches HuggingFace's rotate_half)
        rotated = torch.cat((-x2, x1), dim=-1)
        
        return x * cos + rotated * sin


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "num_heads": 32, "head_dim": 128},
    # Llama-3.1-8B: hidden_size=4096, num_heads=32, head_dim=128
    {"batch_size": 8, "seq_length": 4096, "num_heads": 32, "head_dim": 128},
    # Llama-3.1-70B: hidden_size=8192, num_heads=64, head_dim=128
    {"batch_size": 4, "seq_length": 4096, "num_heads": 64, "head_dim": 128},
    # Mistral-7B-v0.3: hidden_size=4096, num_heads=32, head_dim=128
    {"batch_size": 8, "seq_length": 4096, "num_heads": 32, "head_dim": 128},
    # Qwen2-VL-7B: num_heads=28, head_dim=128
    {"batch_size": 8, "seq_length": 2048, "num_heads": 28, "head_dim": 128},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("embeddings", "1_RotaryEmbedding")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    q = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["num_heads"], p["head_dim"]), dtype=dtype, device=device)
    k = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["num_heads"], p["head_dim"]), dtype=dtype, device=device)
    return [q, k]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["head_dim"]]
