"""
T5 Encoder Model (encoder-only, for SD3.5 text encoding)

Implements the T5 encoder aligned with HuggingFace's
``transformers.T5EncoderModel`` as used in Stable Diffusion 3.5.

Architecture:
  - Shared token embedding
  - N x T5Block:
      - T5LayerSelfAttention: T5LayerNorm -> T5Attention -> dropout -> residual
      - T5LayerFF: T5LayerNorm -> (gated-gelu or dense-relu) MLP -> dropout -> residual
  - final_layer_norm (T5LayerNorm / RMSNorm)
  - dropout

T5 specifics:
  - T5LayerNorm = RMSNorm (no bias, no mean subtraction)
  - Relative position bias (only in first block, shared across layers)
  - Gated-GELU MLP: wi_0 (gate), wi_1 (value), wo (output)
  - No bias in Linear layers

HuggingFace weight structure (T5EncoderModel):
  shared.weight
  encoder.embed_tokens.weight  (tied to shared)
  encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight
  encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight  (i==0 only)
  encoder.block.{i}.layer.0.layer_norm.weight
  encoder.block.{i}.layer.1.DenseReluDense.{wi_0,wi_1,wo}.weight  (gated)
  encoder.block.{i}.layer.1.layer_norm.weight
  encoder.final_layer_norm.weight

Level1 operators used:
  - Linear                     from level1/matmul/_10_Linear
  - Dropout                    from level1/regularization/_1_Dropout
  - Embedding                  from level1/embeddings/_2_Embedding
  - ScaledDotProductAttention  from level1/attention/_2_Attention (eager mode)

Custom operators (not reusable as level1 due to T5-specific dtype casting):
  - T5LayerNorm: RMSNorm with learnable weight that casts output to
    weight dtype (not input dtype), required for fp16 numerical stability
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

# Level1 operator imports
from KernelBench.level1.matmul._10_Linear import Model as Linear
from KernelBench.level1.regularization._1_Dropout import Model as Dropout
from KernelBench.level1.embeddings._2_Embedding import Model as Embedding
from KernelBench.level1.attention._2_Attention import ScaledDotProductAttention


# ============================================================================
# T5LayerNorm (= RMSNorm with learnable weight, no bias)
# ============================================================================

class T5LayerNorm(nn.Module):
    """T5-style layer norm: RMSNorm with learnable scale, no bias.

    Matches HuggingFace T5LayerNorm exactly.

    NOTE: This is NOT a simple wrapper around the level1 RMSNorm because
    T5LayerNorm has a specific dtype-casting behaviour: it always casts
    the output to ``self.weight.dtype`` (typically float16), regardless of
    the input dtype.  This is critical when the FFN ``wo`` layer outputs
    float32 — the next layer_norm must cast back to float16 before the
    attention projections.  The level1 RMSNorm casts to ``input_dtype``
    instead, which would propagate float32 through the rest of the block.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


# ============================================================================
# Relative position bias
# ============================================================================

def _relative_position_bucket(relative_position, bidirectional=True,
                               num_buckets=32, max_distance=128):
    """Compute binned relative position bucket (matches T5)."""
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(
            relative_position, torch.zeros_like(relative_position))

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1))

    relative_buckets += torch.where(
        is_small, relative_position, relative_position_if_large)
    return relative_buckets


# ============================================================================
# T5 Attention
# ============================================================================

class T5Attention(nn.Module):
    """T5 self-attention with relative position bias.

    Uses the level1 ScaledDotProductAttention (eager mode) for the core
    attention computation, with ``scale=1.0`` (T5 does not scale by
    1/sqrt(d)) and ``attn_bias`` for relative position bias.
    """

    def __init__(
        self,
        d_model: int,
        d_kv: int,
        num_heads: int,
        has_relative_attention_bias: bool = False,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = num_heads
        self.key_value_proj_dim = d_kv
        self.inner_dim = num_heads * d_kv
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate

        self.q = Linear(d_model, self.inner_dim, bias=False)
        self.k = Linear(d_model, self.inner_dim, bias=False)
        self.v = Linear(d_model, self.inner_dim, bias=False)
        self.o = Linear(self.inner_dim, d_model, bias=False)

        # Level1 ScaledDotProductAttention in eager mode for exact alignment
        self.attn = ScaledDotProductAttention(mode="eager")

        if has_relative_attention_bias:
            self.relative_attention_bias = Embedding(
                relative_attention_num_buckets, num_heads)

    def compute_bias(self, query_length: int, key_length: int,
                     device: torch.device) -> torch.Tensor:
        """Compute relative position bias."""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position

        relative_position_bucket = _relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = hidden_states.shape[:2]

        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        q = q.view(batch_size, -1, self.n_heads,
                    self.key_value_proj_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads,
                    self.key_value_proj_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads,
                    self.key_value_proj_dim).transpose(1, 2)

        # Compute position bias if not provided
        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(
                    seq_length, seq_length, device=q.device)
            else:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, seq_length),
                    device=q.device, dtype=q.dtype)

            if mask is not None:
                position_bias = position_bias + mask

        # Use level1 ScaledDotProductAttention with scale=1.0 (T5 does not
        # scale by 1/sqrt(d)) and attn_bias for relative position bias.
        attn_output = self.attn(
            q, k, v,
            attn_bias=position_bias,
            scale=1.0,
            dropout_p=self.dropout_rate if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)

        return attn_output, position_bias


# ============================================================================
# T5 Feed-Forward (Gated GELU)
# ============================================================================

class _NewGELU(nn.Module):
    """NewGELUActivation matching HuggingFace's manual implementation exactly.

    ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``

    This avoids numerical differences with ``F.gelu(x, approximate='tanh')``
    in fp16.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5DenseGatedActDense(nn.Module):
    """T5 gated feed-forward: wi_0 (gate) * act, wi_1 (value), wo (output).

    The ``wo`` projection is kept in float32 to avoid fp16 overflow,
    matching HuggingFace's ``_keep_in_fp32_modules`` behaviour for T5.
    """

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.wi_0 = Linear(d_model, d_ff, bias=False)
        self.wi_1 = Linear(d_model, d_ff, bias=False)
        self.wo = Linear(d_ff, d_model, bias=False)
        self.dropout = Dropout(p=dropout_rate)
        # Use manual NewGELU to match HuggingFace exactly in fp16
        self.act = _NewGELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.act(self.wi_0(hidden_states))
        hidden_states = gate * self.wi_1(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Cast to wo weight dtype (float32) to avoid fp16 overflow,
        # matching HuggingFace T5's _keep_in_fp32_modules behaviour.
        if (hidden_states.dtype != self.wo.weight.dtype
                and self.wo.weight.dtype != torch.int8):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


# ============================================================================
# T5 Layer components
# ============================================================================

class T5LayerSelfAttention(nn.Module):
    """T5 self-attention sub-layer: norm -> attention -> dropout -> residual."""

    def __init__(self, d_model: int, d_kv: int, num_heads: int,
                 has_relative_attention_bias: bool = False,
                 relative_attention_num_buckets: int = 32,
                 relative_attention_max_distance: int = 128,
                 dropout_rate: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.SelfAttention = T5Attention(
            d_model=d_model, d_kv=d_kv, num_heads=num_heads,
            has_relative_attention_bias=has_relative_attention_bias,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
            dropout_rate=dropout_rate,
        )
        self.layer_norm = T5LayerNorm(d_model, eps=eps)
        self.dropout = Dropout(p=dropout_rate)

    def forward(self, hidden_states: torch.Tensor,
                position_bias: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.layer_norm(hidden_states)
        attn_output, position_bias = self.SelfAttention(
            normed, position_bias=position_bias, mask=mask)
        hidden_states = hidden_states + self.dropout(attn_output)
        return hidden_states, position_bias


class T5LayerFF(nn.Module):
    """T5 feed-forward sub-layer: norm -> FFN -> dropout -> residual."""

    def __init__(self, d_model: int, d_ff: int,
                 dropout_rate: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(
            d_model, d_ff, dropout_rate)
        self.layer_norm = T5LayerNorm(d_model, eps=eps)
        self.dropout = Dropout(p=dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.layer_norm(hidden_states)
        ff_output = self.DenseReluDense(normed)
        hidden_states = hidden_states + self.dropout(ff_output)
        return hidden_states


# ============================================================================
# T5 Block
# ============================================================================

class T5Block(nn.Module):
    """Single T5 encoder block: self-attention + feed-forward."""

    def __init__(self, d_model: int, d_kv: int, d_ff: int, num_heads: int,
                 has_relative_attention_bias: bool = False,
                 relative_attention_num_buckets: int = 32,
                 relative_attention_max_distance: int = 128,
                 dropout_rate: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.layer = nn.ModuleList([
            T5LayerSelfAttention(
                d_model=d_model, d_kv=d_kv, num_heads=num_heads,
                has_relative_attention_bias=has_relative_attention_bias,
                relative_attention_num_buckets=relative_attention_num_buckets,
                relative_attention_max_distance=relative_attention_max_distance,
                dropout_rate=dropout_rate, eps=eps,
            ),
            T5LayerFF(d_model=d_model, d_ff=d_ff,
                      dropout_rate=dropout_rate, eps=eps),
        ])

    def forward(self, hidden_states: torch.Tensor,
                position_bias: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, position_bias = self.layer[0](
            hidden_states, position_bias=position_bias, mask=mask)
        hidden_states = self.layer[1](hidden_states)
        return hidden_states, position_bias


# ============================================================================
# T5 Encoder
# ============================================================================

class T5Encoder(nn.Module):
    """T5 Encoder Model (encoder-only).

    Drop-in replacement for ``transformers.T5EncoderModel`` with the same
    forward interface (subset used by SD3.5 pipeline).

    The ``wo`` (output projection) layers in each feed-forward block are
    kept in float32 to prevent fp16 overflow, matching HuggingFace's
    ``_keep_in_fp32_modules`` behaviour.
    """

    # Modules that must stay in float32 (matching HF T5)
    _fp32_module_suffix = ".DenseReluDense.wo"

    def __init__(
        self,
        vocab_size: int = 32128,
        d_model: int = 4096,
        d_kv: int = 64,
        d_ff: int = 10240,
        num_heads: int = 64,
        num_layers: int = 24,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
    ):
        super().__init__()

        # Shared embedding (matches HF's shared + encoder.embed_tokens)
        self.shared = Embedding(vocab_size, d_model)

        # Encoder stack
        self.encoder = _T5Stack(
            embed_tokens=self.shared,
            d_model=d_model,
            d_kv=d_kv,
            d_ff=d_ff,
            num_heads=num_heads,
            num_layers=num_layers,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
            dropout_rate=dropout_rate,
            layer_norm_epsilon=layer_norm_epsilon,
        )

    @property
    def dtype(self):
        # Return the dtype of a non-fp32-pinned parameter
        return self.shared.embedding.weight.dtype

    def to(self, *args, **kwargs):
        """Override to keep ``wo`` layers in float32."""
        result = super().to(*args, **kwargs)
        # Re-cast wo layers back to float32
        for name, module in self.named_modules():
            if name.endswith(self._fp32_module_suffix):
                module.to(torch.float32)
        return result

    def half(self):
        """Override to keep ``wo`` layers in float32."""
        result = super().half()
        for name, module in self.named_modules():
            if name.endswith(self._fp32_module_suffix):
                module.to(torch.float32)
        return result

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Optional mask (batch, seq_len)

        Returns:
            Tuple of (last_hidden_state,)
        """
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return (encoder_output,)


class _T5Stack(nn.Module):
    """Internal T5 encoder stack."""

    def __init__(self, embed_tokens, d_model, d_kv, d_ff, num_heads,
                 num_layers, relative_attention_num_buckets,
                 relative_attention_max_distance, dropout_rate,
                 layer_norm_epsilon):
        super().__init__()
        self.embed_tokens = embed_tokens

        self.block = nn.ModuleList([
            T5Block(
                d_model=d_model, d_kv=d_kv, d_ff=d_ff, num_heads=num_heads,
                has_relative_attention_bias=(i == 0),
                relative_attention_num_buckets=relative_attention_num_buckets,
                relative_attention_max_distance=relative_attention_max_distance,
                dropout_rate=dropout_rate, eps=layer_norm_epsilon,
            )
            for i in range(num_layers)
        ])
        self.final_layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = Dropout(p=dropout_rate)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)

        # Prepare extended attention mask if needed
        mask = None
        if attention_mask is not None:
            # (batch, seq) -> (batch, 1, 1, seq) with 0/-inf
            mask = attention_mask[:, None, None, :].to(
                dtype=hidden_states.dtype)
            mask = (1.0 - mask) * torch.finfo(hidden_states.dtype).min

        position_bias = None
        for block in self.block:
            hidden_states, position_bias = block(
                hidden_states, position_bias=position_bias, mask=mask)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
