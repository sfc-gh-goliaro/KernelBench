"""
Multimodal Rotary Position Embedding (M-RoPE)

Used by: Qwen2-VL, Qwen3-VL, Qwen3-Omni (language model decoders)

Computes 3D multimodal rotary position embeddings from 3D position IDs
(temporal, height, width). For text tokens the three dimensions are identical,
while for vision/video tokens they encode the spatial and temporal structure.

Supports two section assembly modes:
- "chunked": Qwen2-VL style. Splits head_dim into chunks per mrope_section,
  picks dimension [i % 3] from each chunk, and concatenates.
- "interleaved": Qwen3-VL / Qwen3-Omni style. Starts from temporal frequencies
  and overwrites height/width frequencies at stride-3 positions.

Shapes:
    Input: position_ids (3, batch_size, seq_len)
    Output: (cos, sin) each of shape (batch_size, seq_len, head_dim)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Literal


class Model(nn.Module):
    """
    Multimodal Rotary Position Embedding (M-RoPE).

    Owns inv_freq and computes cos/sin from 3D position IDs with configurable
    section assembly strategy.

    Args:
        head_dim: Full dimension of each attention head (must be even).
        base: Base for the frequency computation (rope_theta).
        mrope_section: List of 3 ints giving the number of frequency pairs
                       assigned to each of the 3 position dimensions
                       (temporal, height, width). Must sum to head_dim // 2.
        mode: Section assembly mode.
              "chunked" - Qwen2-VL: splits cos/sin by doubled sections,
                          picks dim [i%3] from each chunk, concatenates.
              "interleaved" - Qwen3-VL/Omni: starts from temporal, overwrites
                              height/width at stride-3 positions.
        attention_scaling: Multiplicative scaling applied to cos/sin
                          (default 1.0, used by some YARN configs).
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 1000000.0,
        mrope_section: Optional[List[int]] = None,
        mode: Literal["chunked", "interleaved"] = "chunked",
        attention_scaling: float = 1.0,
    ):
        super(Model, self).__init__()
        if mrope_section is None:
            mrope_section = [16, 24, 24]
        self.head_dim = head_dim
        self.mrope_section = mrope_section
        self.mode = mode
        self.attention_scaling = attention_scaling

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
        # Store float32 copy that survives .to(bfloat16)
        self._inv_freq_float32 = inv_freq
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply(self, fn):
        """Override to keep inv_freq in float32 when model dtype changes."""
        super()._apply(fn)
        if hasattr(self, '_inv_freq_float32'):
            target_device = self.inv_freq.device
            self._inv_freq_float32 = self._inv_freq_float32.to(device=target_device)
            self.inv_freq = self._inv_freq_float32.clone()
        return self

    def _get_inv_freq(self) -> torch.Tensor:
        """Return inv_freq in float32, updating device if needed."""
        if self._inv_freq_float32.device != self.inv_freq.device:
            self._inv_freq_float32 = self._inv_freq_float32.to(device=self.inv_freq.device)
        return self._inv_freq_float32

    def _assemble_chunked(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Qwen2-VL chunked M-RoPE assembly.

        Splits (3, batch, seq, head_dim) cos/sin along head_dim into sections
        matching mrope_section * 2 (list duplication, NOT element doubling),
        picks dimension [i % 3] from each section, and concatenates back to
        (batch, seq, head_dim).

        For mrope_section = [16, 24, 24] and head_dim = 128:
          sections = [16, 24, 24, 16, 24, 24]  (sums to 128)
          chunk 0 (16): dim 0 (temporal)
          chunk 1 (24): dim 1 (height)
          chunk 2 (24): dim 2 (width)
          chunk 3 (16): dim 0 (temporal)
          chunk 4 (24): dim 1 (height)
          chunk 5 (24): dim 2 (width)
        """
        # List duplication: [16,24,24] * 2 = [16,24,24,16,24,24]
        # This is how the original HuggingFace Qwen2-VL code works.
        sections = self.mrope_section * 2
        cos = torch.cat(
            [m[i % 3] for i, m in enumerate(cos.split(sections, dim=-1))],
            dim=-1,
        )
        sin = torch.cat(
            [m[i % 3] for i, m in enumerate(sin.split(sections, dim=-1))],
            dim=-1,
        )
        return cos, sin

    def _assemble_interleaved(
        self,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Qwen3-VL / Qwen3-Omni interleaved M-RoPE assembly.

        Starts from temporal frequencies and overwrites height/width at stride-3
        positions within each section.

        Args:
            freqs: (3, batch, seq, head_dim // 2)

        Returns:
            Assembled frequencies (batch, seq, head_dim // 2)
        """
        freqs_t = freqs[0].clone()
        for dim_idx, offset in enumerate((1, 2), start=1):  # H, W
            length = self.mrope_section[dim_idx] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim_idx, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute M-RoPE cos/sin from 3D position IDs.

        Args:
            x: Reference tensor for dtype (e.g. inputs_embeds). Only dtype is used.
            position_ids: (3, batch_size, seq_len) position indices for the three
                         dimensions (temporal, height, width).

        Returns:
            Tuple of (cos, sin), each of shape (batch_size, seq_len, head_dim)
            in the dtype of x. For "chunked" mode the leading dim-3 is already
            folded; for "interleaved" mode likewise.
        """
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        inv_freq = self._get_inv_freq()
        # (3, batch, head_dim//2, 1)
        inv_freq_expanded = inv_freq[None, None, :, None].expand(
            3, position_ids.shape[1], -1, 1
        )
        # (3, batch, 1, seq_len) -> matmul -> (3, batch, head_dim//2, seq_len)
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        # freqs: (3, batch, seq_len, head_dim // 2)

        if self.mode == "interleaved":
            # Assemble before doubling
            freqs = self._assemble_interleaved(freqs)
            # freqs: (batch, seq_len, head_dim // 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        else:
            # "chunked": double first, compute cos/sin, then assemble
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            cos, sin = self._assemble_chunked(cos, sin)

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ============================================================================
# Benchmark Configuration
# ============================================================================
