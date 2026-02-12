"""
Qwen2 Text Encoder (Qwen2_5_VLTextModel)

A standalone text encoder extracted from the Qwen2-VL language model backbone.
Used by HunyuanVideo 1.5 as the primary text encoder.

Architecture:
- Embedding -> Qwen2DecoderLayer x N -> RMSNorm
- Each decoder layer: RMSNorm -> GQA Attention (with M-RoPE) -> residual
                      -> RMSNorm -> SiLU-gated MLP -> residual
- Returns last_hidden_state (no lm_head)

Matches HuggingFace transformers Qwen2_5_VLTextModel state-dict layout:
    embed_tokens.weight
    layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}
    layers.{i}.self_attn.o_proj.weight
    layers.{i}.input_layernorm.weight
    layers.{i}.post_attention_layernorm.weight
    layers.{i}.mlp.{gate,up,down}_proj.weight
    norm.weight

Level1 operators used:
- Linear from level1/matmul/_10_Linear
- RMSNorm from level1/normalization/_4_RMSNorm
- Embedding from level1/embeddings/_2_Embedding
- Swish (SiLU) from level1/activations/_7_Swish
- RotaryEmbedding from level1/embeddings/_1_RotaryEmbedding
- ScaledDotProductAttention from level1/attention/_2_Attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from KernelBench.level1.matmul._10_Linear import Model as Linear
from KernelBench.level1.normalization._4_RMSNorm import Model as RMSNorm
from KernelBench.level1.embeddings._2_Embedding import Model as Embedding
from KernelBench.level1.activations._7_Swish import Model as Swish
from KernelBench.level1.embeddings._1_RotaryEmbedding import Model as RotaryEmbedding
from KernelBench.level1.attention._2_Attention import ScaledDotProductAttention


# ============================================================================
# Helper: rotate_half (for M-RoPE application)
# ============================================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Multimodal Rotary Position Embedding to query and key tensors.

    Matches HuggingFace apply_multimodal_rotary_pos_emb exactly.

    Args:
        q: (batch, heads, seq_len, head_dim)
        k: (batch, kv_heads, seq_len, head_dim)
        cos: (3, batch, seq_len, head_dim)
        sin: (3, batch, seq_len, head_dim)
        mrope_section: list of ints for temporal/height/width channel splits
    """
    mrope_section = mrope_section * 2
    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))],
        dim=-1,
    ).unsqueeze(1)  # (batch, 1, seq_len, head_dim)
    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))],
        dim=-1,
    ).unsqueeze(1)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ============================================================================
# Qwen2 Rotary Embedding (matches Qwen2_5_VLRotaryEmbedding)
# ============================================================================

class Qwen2RotaryEmbedding(nn.Module):
    """Rotary position embedding for Qwen2 with M-RoPE support.

    Matches HuggingFace Qwen2_5_VLRotaryEmbedding for the default rope_type.
    Computes 3D cos/sin embeddings from 3D position_ids (temporal, height, width).
    """

    def __init__(self, head_dim: int, max_position_embeddings: int = 32768,
                 base: float = 1000000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary embeddings.

        Args:
            x: hidden states, used only for dtype/device
            position_ids: (3, batch, seq_len) M-RoPE position indices

        Returns:
            cos, sin: each (3, batch, seq_len, head_dim)
        """
        # inv_freq_expanded: (3, batch, head_dim/2, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
            3, position_ids.shape[1], -1, 1
        )
        # position_ids_expanded: (3, batch, 1, seq_len)
        position_ids_expanded = position_ids[:, :, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ============================================================================
# Qwen2 MLP (SiLU-gated)
# ============================================================================

class Qwen2MLP(nn.Module):
    """SiLU-gated MLP matching HuggingFace Qwen2MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.swish(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# Qwen2 Attention (GQA + M-RoPE)
# ============================================================================

class Qwen2Attention(nn.Module):
    """Multi-headed attention with GQA and M-RoPE.

    Uses level1 operators:
    - Linear for Q/K/V/O projections
    - ScaledDotProductAttention for attention computation
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        mrope_section: Optional[List[int]] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.mrope_section = mrope_section if mrope_section is not None else [16, 24, 24]

        self.q_proj = Linear(hidden_size, num_heads * self.head_dim, bias=True)
        self.k_proj = Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.v_proj = Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.o_proj = Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # Level1 ScaledDotProductAttention
        self.sdpa = ScaledDotProductAttention(mode="eager")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # Apply M-RoPE
        cos, sin = position_embeddings
        query_states, key_states = _apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.mrope_section
        )

        # GQA: repeat KV heads
        if self.num_kv_groups > 1:
            key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        # Attention via level1 SDPA
        attn_output = self.sdpa(
            query_states, key_states, value_states,
            attn_mask=attention_mask, scale=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


# ============================================================================
# Qwen2 Decoder Layer
# ============================================================================

class Qwen2DecoderLayer(nn.Module):
    """Decoder layer with RMSNorm, GQA attention, SiLU-gated MLP."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
        mrope_section: Optional[List[int]] = None,
    ):
        super().__init__()
        self.self_attn = Qwen2Attention(
            hidden_size, num_heads, num_kv_heads,
            mrope_section=mrope_section,
        )
        self.mlp = Qwen2MLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(
            hidden_size, eps=rms_norm_eps, learnable_weight=True, dim=-1
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size, eps=rms_norm_eps, learnable_weight=True, dim=-1
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ============================================================================
# Qwen2 Text Encoder (main class)
# ============================================================================

class Qwen2TextEncoder(nn.Module):
    """Qwen2 text encoder matching HuggingFace Qwen2_5_VLTextModel.

    This is the language model backbone without lm_head or vision encoder.
    Returns last_hidden_state for use as text conditioning in diffusion models.

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        num_hidden_layers: Number of decoder layers
        num_attention_heads: Number of query heads
        num_key_value_heads: Number of KV heads (for GQA)
        intermediate_size: MLP intermediate dimension
        rms_norm_eps: RMSNorm epsilon
        rope_theta: RoPE base frequency
        max_position_embeddings: Maximum sequence length
        mrope_section: M-RoPE section sizes [temporal, height, width]
    """

    def __init__(
        self,
        vocab_size: int = 152064,
        hidden_size: int = 8192,
        num_hidden_layers: int = 80,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        intermediate_size: int = 29568,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 32768,
        mrope_section: Optional[List[int]] = None,
    ):
        super().__init__()
        if mrope_section is None:
            mrope_section = [16, 24, 24]

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.mrope_section = mrope_section

        self.embed_tokens = Embedding(vocab_size, hidden_size)

        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
                mrope_section=mrope_section,
            )
            for _ in range(num_hidden_layers)
        ])

        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps, learnable_weight=True, dim=-1)

        head_dim = hidden_size // num_attention_heads
        self.rotary_emb = Qwen2RotaryEmbedding(
            head_dim, max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.embedding.weight.dtype

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Forward pass returning last hidden state.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) padding mask (1 = attend, 0 = ignore)
            position_ids: (3, batch, seq_len) M-RoPE position indices.
                          If None, defaults to simple sequential positions.

        Returns:
            hidden_states: (batch, seq_len, hidden_size)
        """
        inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_len = input_ids.shape

        # Prepare position_ids
        if position_ids is None:
            if attention_mask is not None:
                pos = attention_mask.long().cumsum(-1) - 1
                pos.masked_fill_(attention_mask == 0, 1)
                position_ids = pos.unsqueeze(0).expand(3, -1, -1)
            else:
                position_ids = (
                    torch.arange(seq_len, device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, batch_size, -1)
                )

        # Compute rotary embeddings
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # Build causal attention mask
        causal_mask = None
        if attention_mask is not None or seq_len > 1:
            causal_mask = torch.full(
                (batch_size, 1, seq_len, seq_len),
                float("-inf"),
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            causal_mask = causal_mask.triu(diagonal=1)
            if attention_mask is not None:
                # Mask out padding positions
                padding_mask = attention_mask[:, None, None, :].eq(0)
                causal_mask = causal_mask.masked_fill(padding_mask, float("-inf"))

        # Decoder layers
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states
