"""
Rotary Position Embedding (RoPE)

Used by: Llama, Qwen, Mistral, Gemma, Yi, DeepSeek, Phi

Applies rotary position embeddings to query and key tensors by rotating
pairs of dimensions using precomputed cos/sin values based on position.

Supports multiple RoPE types:
- "default": Standard RoPE
- "llama3": Llama-3.1 style with piecewise frequency scaling
- "yarn": YaRN (Yet another RoPE extensioN) for extended context (DeepSeek-V2)

Supports two application modes:
- "half_rotate": cos/sin rotate_half approach (Llama, Falcon, Mixtral, etc.)
    Pairs dimensions d/2 apart: (x_0, x_{d/2}), (x_1, x_{d/2+1}), ...
- "complex": complex-polar multiplication approach (DeepSeek-V2)
    Pairs adjacent dimensions: (x_0, x_1), (x_2, x_3), ...

Shapes (depends on layout parameter):
    layout="bshd": (batch_size, seq_len, num_heads, head_dim) - default
    layout="bhsd": (batch_size, num_heads, seq_len, head_dim) - Llama attention style
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Literal, Dict, Any, Tuple


# ============================================================================
# Inverse frequency computation helpers
# ============================================================================

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
    Compute YaRN inverse frequencies (DeepSeek-V2 style).

    Matches HuggingFace's DeepSeek-V2 implementation exactly.

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
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
    )

    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
        low = max(math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings)), 0)
        high = min(math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings)), dim - 1)
        return low, high

    def linear_ramp_mask(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32, device=device) - min_val) / (max_val - min_val)
        return torch.clamp(linear_func, 0, 1)

    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings)
    inv_freq_mask = 1.0 - linear_ramp_mask(low, high, dim // 2)
    inv_freq = inv_freq / factor * (1 - inv_freq_mask) + inv_freq * inv_freq_mask

    return inv_freq


def compute_yarn_attention_scaling(rope_scaling: Dict[str, Any]) -> float:
    """
    Compute attention scaling factor for YARN.

    Matches HuggingFace's implementation in modeling_rope_utils.py.

    Args:
        rope_scaling: YARN scaling config dict

    Returns:
        Attention scaling factor (float)
    """
    def _float_key(d, key, default):
        val = d.get(key, default)
        return float(val) if val is not None else default

    def get_mscale(scale, mscale_val=1.0):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale_val * math.log(scale) + 1.0

    # Get attention_factor if explicitly provided
    attention_factor = rope_scaling.get("attention_factor")
    if attention_factor is not None:
        return float(attention_factor)

    # Otherwise compute from mscale/mscale_all_dim
    mscale = _float_key(rope_scaling, "mscale", None)
    mscale_all_dim = _float_key(rope_scaling, "mscale_all_dim", None)
    factor = _float_key(rope_scaling, "factor", 1.0)

    if mscale is not None and mscale_all_dim is not None:
        return float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
    elif mscale is not None:
        return get_mscale(factor, mscale)
    else:
        return get_mscale(factor)


# ============================================================================
# RoPE application helpers
# ============================================================================

def apply_rotary_complex(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings using complex number multiplication.

    Pairs adjacent dimensions: (x_0, x_1), (x_2, x_3), ...
    Used by DeepSeek-V2 for exact numerical alignment.

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


# ============================================================================
# Main Model
# ============================================================================

class Model(nn.Module):
    """
    Rotary Position Embedding (RoPE) with support for various scaling methods.

    Supports:
    - Standard RoPE (rope_type="default")
    - Llama-3.1 RoPE scaling (rope_type="llama3")
    - YaRN RoPE scaling (rope_type="yarn", used by DeepSeek-V2)

    Application modes:
    - "half_rotate": cos/sin rotate_half (Llama, Falcon, Mixtral, etc.)
    - "complex": complex-polar multiplication (DeepSeek-V2)
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        layout: Literal["bshd", "bhsd"] = "bshd",
        rope_scaling: Optional[Dict[str, Any]] = None,
        interleaved: bool = False,
        mode: Literal["half_rotate", "complex"] = "half_rotate",
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
                    - For yarn: beta_fast, beta_slow, original_max_position_embeddings,
                                mscale, mscale_all_dim (for attention scaling)
            interleaved: If True, apply interleaving transformation before RoPE.
                        This transforms [x0,x1,x2,x3,...] to [x0,x2,...,x1,x3,...] format.
                        Only used with mode="half_rotate".
            mode: Application mode.
                  "half_rotate" - cos/sin with rotate_half (pairs dims d/2 apart)
                  "complex" - complex polar multiplication (pairs adjacent dims)
        """
        super(Model, self).__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.layout = layout
        self.rope_scaling = rope_scaling
        self.interleaved = interleaved
        self.mode = mode

        # Determine rope type
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
                factor=float(rope_scaling.get("factor", 1.0)),
                original_max_position_embeddings=rope_scaling.get("original_max_position_embeddings", 4096),
                beta_fast=float(rope_scaling.get("beta_fast", 32)),
                beta_slow=float(rope_scaling.get("beta_slow", 1)),
            )
            self.attention_scaling = compute_yarn_attention_scaling(rope_scaling)
        else:
            inv_freq = compute_default_inv_freq(head_dim, base)

        # Store inv_freq in float32 as a non-buffer attribute to survive
        # model.to(bfloat16) conversions. The buffer is for device tracking only.
        self._inv_freq_float32 = inv_freq
        self.register_buffer("inv_freq", inv_freq.clone(), persistent=False)

        if mode != "complex":
            # Precompute cos and sin for all positions
            self._update_cos_sin_cache(max_seq_len)

    def _apply(self, fn):
        """Override to keep inv_freq in float32 when model dtype changes."""
        super()._apply(fn)
        if hasattr(self, '_inv_freq_float32'):
            target_device = self.inv_freq.device
            self.inv_freq = self._inv_freq_float32.to(device=target_device)
            self._inv_freq_float32 = self._inv_freq_float32.to(device=target_device)
        return self

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device = None):
        """Update the cos/sin cache for the given sequence length (half_rotate mode only)."""
        if device is None:
            device = self.inv_freq.device

        t = torch.arange(seq_len, device=device, dtype=torch.float)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        self.register_buffer('cos_cached', cos, persistent=False)
        self.register_buffer('sin_cached', sin, persistent=False)
        self.max_seq_len = seq_len

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        if self.mode == "complex":
            return self._forward_complex(q, k, position_ids)
        else:
            return self._forward_half_rotate(q, k, position_ids)

    def _forward_complex(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE using complex-polar multiplication (DeepSeek-V2 style)."""
        # Determine reference tensor for device (q is always available)
        device = q.device

        if position_ids is None:
            # Infer from layout
            if self.layout == "bhsd":
                seq_len = q.shape[2]
            else:
                seq_len = q.shape[1]
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # Compute complex frequencies from inv_freq and position_ids
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.to(device) @ position_ids_expanded).transpose(1, 2)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        # Apply YARN attention scaling if applicable
        if hasattr(self, 'attention_scaling'):
            freqs_cis = freqs_cis * self.attention_scaling

        # Ensure q and k are in bhsd layout for complex application
        if self.layout == "bshd":
            # (batch, seq, heads, dim) -> (batch, heads, seq, dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

        q_rot, k_rot = apply_rotary_complex(q, k, freqs_cis)

        if self.layout == "bshd":
            q_rot = q_rot.transpose(1, 2)
            k_rot = k_rot.transpose(1, 2)

        return q_rot, k_rot

    def _forward_half_rotate(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE using cos/sin rotate_half approach."""
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
            if position_ids is None:
                cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
                sin = sin.unsqueeze(0).unsqueeze(0)
            else:
                cos = cos.unsqueeze(1)  # (batch, 1, seq, head_dim)
                sin = sin.unsqueeze(1)
        else:
            if position_ids is None:
                cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim)
                sin = sin.unsqueeze(0).unsqueeze(2)
            else:
                cos = cos.unsqueeze(2)  # (batch, seq, 1, head_dim)
                sin = sin.unsqueeze(2)

        # Convert cos/sin to input dtype to preserve precision
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)

        q_rotated = self._apply_rotary_half(q, cos, sin)
        k_rotated = self._apply_rotary_half(k, cos, sin)

        return q_rotated, k_rotated

    def _apply_rotary_half(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding using rotate_half to a single tensor."""
        # Apply interleaving transformation if needed
        if self.interleaved:
            d = x.shape[-1]
            x = x.view(*x.shape[:-1], d // 2, 2).transpose(-1, -2).reshape(*x.shape[:-1], d)

        # Split into two halves
        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]

        # Rotate (matches HuggingFace's rotate_half)
        rotated = torch.cat((-x2, x1), dim=-1)

        return x * cos + rotated * sin

    def apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to q and k using pre-computed cos/sin.

        This is useful for models that compute their own cos/sin externally
        (e.g. Qwen2-VL's M-RoPE or vision RoPE) but want to reuse the
        core rotate_half logic.

        Only supported for mode="half_rotate".

        Args:
            q: Query tensor (arbitrary shape, rotation applied along last dim)
            k: Key tensor (same shape convention as q)
            cos: Cosine values, broadcastable to q/k shape
            sin: Sine values, broadcastable to q/k shape

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        q_rotated = self._apply_rotary_half(q, cos, sin)
        k_rotated = self._apply_rotary_half(k, cos, sin)
        return q_rotated, k_rotated


# ============================================================================
# Benchmark Configuration
# ============================================================================
