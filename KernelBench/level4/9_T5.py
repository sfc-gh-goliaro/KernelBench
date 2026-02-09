"""
T5 Encoder-Decoder Model

Implements T5 (Text-to-Text Transfer Transformer) architecture aligned with
HuggingFace T5ForConditionalGeneration:
- Encoder-decoder structure
- Relative position bias (bidirectional for encoder, unidirectional for decoder)
- Pre-norm RMSNorm (T5LayerNorm)
- Supports both dense-relu-dense and gated-gelu MLP variants

HuggingFace weight structure (T5ForConditionalGeneration):
  shared.weight
  encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight
  encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight  (i==0 only)
  encoder.block.{i}.layer.0.layer_norm.weight
  encoder.block.{i}.layer.1.DenseReluDense.{wi,wo}.weight                 (non-gated)
  encoder.block.{i}.layer.1.DenseReluDense.{wi_0,wi_1,wo}.weight          (gated)
  encoder.block.{i}.layer.1.layer_norm.weight
  encoder.final_layer_norm.weight
  decoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight
  decoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight  (i==0 only)
  decoder.block.{i}.layer.0.layer_norm.weight
  decoder.block.{i}.layer.1.EncDecAttention.{q,k,v,o}.weight
  decoder.block.{i}.layer.1.layer_norm.weight
  decoder.block.{i}.layer.2.DenseReluDense.{wi,wo}.weight                 (non-gated)
  decoder.block.{i}.layer.2.DenseReluDense.{wi_0,wi_1,wo}.weight          (gated)
  decoder.block.{i}.layer.2.layer_norm.weight
  decoder.final_layer_norm.weight
  lm_head.weight

Tested against: google/flan-t5-large

This model uses level1 operators from KernelBench:
- RMSNorm from level1/normalization/4_RMSNorm (T5LayerNorm)
- Linear from level1/matmul/10_Linear (all projections)
- Embedding from level1/embeddings/2_Embedding (shared vocab embedding)
- GELU from level1/activations/8_GELU (gated MLP)
- ReLU from level1/activations/1_ReLU (non-gated MLP)
- Softmax from level1/activations/5_Softmax (attention weights)
- MatMul from level1/matmul/1_MatMul (attention scores & context)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any, Tuple, List

# Import level1 operators
from ..level1.normalization._4_RMSNorm import Model as RMSNorm
from ..level1.matmul._10_Linear import Model as Linear
from ..level1.matmul._1_MatMul import Model as MatMul
from ..level1.embeddings._2_Embedding import Model as Embedding
from ..level1.activations._8_GELU import Model as GELUOp
from ..level1.activations._1_ReLU import Model as ReLUOp
from ..level1.activations._5_Softmax import Model as Softmax


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "Base": "google-t5/t5-base",
    "Large": "google-t5/t5-large",
    "3B": "google-t5/t5-3b",
    "FlanSmall": "google/flan-t5-small",
    "FlanBase": "google/flan-t5-base",
    "FlanLarge": "google/flan-t5-large",
}


# ============================================================================
# T5 Layer Norm via level1 RMSNorm
# ============================================================================

class T5LayerNorm(nn.Module):
    """T5-style RMSNorm: no bias, no mean subtraction.

    Matches HuggingFace T5LayerNorm exactly.
    Implemented using level1 RMSNorm operator.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        # T5LayerNorm is RMSNorm with learnable weight, dim=-1
        self.rmsnorm = RMSNorm(
            num_features=hidden_size, eps=eps,
            learnable_weight=True, dim=-1
        )

    @property
    def weight(self):
        """Expose weight for HF weight copying compatibility."""
        return self.rmsnorm.weight

    @weight.setter
    def weight(self, value):
        self.rmsnorm.weight = value

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(hidden_states)


# ============================================================================
# Component Modules (matching HF weight layout)
# ============================================================================

class T5Attention(nn.Module):
    """T5 self/cross-attention matching HF T5Attention.

    Weight names match HF:
      self.q, self.k, self.v, self.o
      self.relative_attention_bias  (only for first layer self-attention)

    Uses level1 operators: Linear, MatMul, Softmax, Embedding
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_kv: int,
        is_decoder: bool = False,
        has_relative_attention_bias: bool = False,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
    ):
        super().__init__()
        self.is_decoder = is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.d_model = d_model
        self.key_value_proj_dim = d_kv
        self.n_heads = num_heads
        self.inner_dim = num_heads * d_kv

        # Level1 Linear for Q, K, V, O projections
        self.q = Linear(d_model, self.inner_dim, bias=False)
        self.k = Linear(d_model, self.inner_dim, bias=False)
        self.v = Linear(d_model, self.inner_dim, bias=False)
        self.o = Linear(self.inner_dim, d_model, bias=False)

        if has_relative_attention_bias:
            # Level1 Embedding for relative position bias lookup
            self.relative_attention_bias = Embedding(
                relative_attention_num_buckets, num_heads
            )

        # Level1 operators
        self.matmul = MatMul()
        self.softmax = Softmax(dim=-1)

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """Compute relative position bucket (matches HF exactly)."""
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device) -> torch.Tensor:
        """Compute relative position bias (matches HF)."""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[
            :, None
        ]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
            None, :
        ]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = hidden_states.shape[:2]

        is_cross_attention = key_value_states is not None
        current_states = key_value_states if is_cross_attention else hidden_states
        kv_len = current_states.shape[1]

        query_states = self.q(hidden_states).view(
            batch_size, -1, self.n_heads, self.key_value_proj_dim
        ).transpose(1, 2)
        key_states = self.k(current_states).view(
            batch_size, -1, self.n_heads, self.key_value_proj_dim
        ).transpose(1, 2)
        value_states = self.v(current_states).view(
            batch_size, -1, self.n_heads, self.key_value_proj_dim
        ).transpose(1, 2)

        scores = self.matmul(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(
                    seq_length, kv_len, device=hidden_states.device
                )
            else:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, kv_len),
                    device=scores.device,
                    dtype=scores.dtype,
                )

            if mask is not None:
                position_bias = position_bias + mask

        scores += position_bias

        attn_weights = self.softmax(scores.float()).type_as(scores)
        attn_output = self.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.inner_dim
        )
        attn_output = self.o(attn_output)

        return attn_output, position_bias


class T5LayerSelfAttention(nn.Module):
    """T5 self-attention sub-layer matching HF T5LayerSelfAttention.

    HF weight names:
      self.SelfAttention.{q,k,v,o}.weight
      self.SelfAttention.relative_attention_bias.weight  (layer 0 only)
      self.layer_norm.weight
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_kv: int,
        is_decoder: bool = False,
        has_relative_attention_bias: bool = False,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
    ):
        super().__init__()
        self.SelfAttention = T5Attention(
            d_model=d_model,
            num_heads=num_heads,
            d_kv=d_kv,
            is_decoder=is_decoder,
            has_relative_attention_bias=has_relative_attention_bias,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
        )
        self.layer_norm = T5LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.layer_norm(hidden_states)
        attn_output, position_bias = self.SelfAttention(
            normed, position_bias=position_bias, mask=attention_mask
        )
        hidden_states = hidden_states + attn_output
        return hidden_states, position_bias


class T5LayerCrossAttention(nn.Module):
    """T5 cross-attention sub-layer matching HF T5LayerCrossAttention.

    HF weight names:
      self.EncDecAttention.{q,k,v,o}.weight
      self.layer_norm.weight
    """
    def __init__(self, d_model: int, num_heads: int, d_kv: int):
        super().__init__()
        self.EncDecAttention = T5Attention(
            d_model=d_model,
            num_heads=num_heads,
            d_kv=d_kv,
            is_decoder=True,
            has_relative_attention_bias=False,
        )
        self.layer_norm = T5LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        position_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.layer_norm(hidden_states)
        attn_output, position_bias = self.EncDecAttention(
            normed,
            key_value_states=encoder_hidden_states,
            position_bias=position_bias,
            mask=attention_mask,
        )
        hidden_states = hidden_states + attn_output
        return hidden_states, position_bias


class T5DenseActDense(nn.Module):
    """Non-gated T5 MLP (relu/gelu) matching HF T5DenseActDense.

    HF weight names: self.wi.weight, self.wo.weight
    Uses level1 Linear, ReLU/GELU operators.
    """
    def __init__(self, d_model: int, d_ff: int, act_fn: str = "relu"):
        super().__init__()
        self.wi = Linear(d_model, d_ff, bias=False)
        self.wo = Linear(d_ff, d_model, bias=False)
        if act_fn == "relu":
            self.act = ReLUOp()
        else:
            self.act = GELUOp()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.wo(hidden_states)


class T5DenseGatedActDense(nn.Module):
    """Gated T5 MLP (gated-gelu) matching HF T5DenseGatedActDense.

    HF weight names: self.wi_0.weight, self.wi_1.weight, self.wo.weight
    Uses level1 Linear and GELU operators.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.wi_0 = Linear(d_model, d_ff, bias=False)
        self.wi_1 = Linear(d_model, d_ff, bias=False)
        self.wo = Linear(d_ff, d_model, bias=False)
        self.act = GELUOp()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        return self.wo(hidden_states)


class T5LayerFF(nn.Module):
    """T5 feed-forward sub-layer matching HF T5LayerFF.

    HF weight names:
      self.DenseReluDense.{wi,wo}.weight     (non-gated)
      self.DenseReluDense.{wi_0,wi_1,wo}.weight  (gated)
      self.layer_norm.weight
    """
    def __init__(
        self, d_model: int, d_ff: int, is_gated_act: bool = False, act_fn: str = "relu"
    ):
        super().__init__()
        if is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(d_model, d_ff)
        else:
            self.DenseReluDense = T5DenseActDense(d_model, d_ff, act_fn=act_fn)
        self.layer_norm = T5LayerNorm(d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forwarded = self.layer_norm(hidden_states)
        forwarded = self.DenseReluDense(forwarded)
        return hidden_states + forwarded


class T5Block(nn.Module):
    """T5 transformer block matching HF T5Block.

    Encoder block: self.layer = [SelfAttention, FF]
    Decoder block: self.layer = [SelfAttention, CrossAttention, FF]

    HF uses nn.ModuleList named 'layer' with indices:
      Encoder: layer[0] = SelfAttention, layer[1] = FF
      Decoder: layer[0] = SelfAttention, layer[1] = CrossAttention, layer[2] = FF
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_kv: int,
        d_ff: int,
        is_decoder: bool = False,
        has_relative_attention_bias: bool = False,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        is_gated_act: bool = False,
        act_fn: str = "relu",
    ):
        super().__init__()
        self.is_decoder = is_decoder
        self.layer = nn.ModuleList()

        # layer[0]: self-attention
        self.layer.append(
            T5LayerSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                d_kv=d_kv,
                is_decoder=is_decoder,
                has_relative_attention_bias=has_relative_attention_bias,
                relative_attention_num_buckets=relative_attention_num_buckets,
                relative_attention_max_distance=relative_attention_max_distance,
            )
        )

        # layer[1]: cross-attention (decoder only)
        if is_decoder:
            self.layer.append(
                T5LayerCrossAttention(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_kv=d_kv,
                )
            )

        # layer[-1]: feed-forward
        self.layer.append(
            T5LayerFF(
                d_model=d_model,
                d_ff=d_ff,
                is_gated_act=is_gated_act,
                act_fn=act_fn,
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_decoder_position_bias: Optional[torch.Tensor] = None,
    ):
        # Self-attention
        hidden_states, position_bias = self.layer[0](
            hidden_states,
            position_bias=position_bias,
            attention_mask=attention_mask,
        )

        # Cross-attention (decoder only)
        if self.is_decoder and encoder_hidden_states is not None:
            hidden_states, encoder_decoder_position_bias = self.layer[1](
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                position_bias=encoder_decoder_position_bias,
                attention_mask=encoder_attention_mask,
            )

        # Feed-forward
        hidden_states = self.layer[-1](hidden_states)

        outputs = (hidden_states, position_bias)
        if self.is_decoder:
            outputs = outputs + (encoder_decoder_position_bias,)
        return outputs


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    T5 encoder-decoder transformer matching HuggingFace T5ForConditionalGeneration.

    The weight structure is designed so that state_dict keys match HF keys
    (after stripping the 'encoder.'/'decoder.' prefixes in the test harness).

    HF structure:
      shared -> self.shared (Embedding)
      encoder.block.{i} -> self.encoder_blocks[i] (T5Block)
      encoder.final_layer_norm -> self.encoder_final_layer_norm
      decoder.block.{i} -> self.decoder_blocks[i] (T5Block)
      decoder.final_layer_norm -> self.decoder_final_layer_norm
      lm_head -> self.lm_head (Linear)

    Uses level1 operators:
    - RMSNorm from level1/normalization/4_RMSNorm (T5LayerNorm)
    - Linear from level1/matmul/10_Linear (all projections)
    - Embedding from level1/embeddings/2_Embedding (shared vocab)
    - GELU from level1/activations/8_GELU
    - ReLU from level1/activations/1_ReLU
    - Softmax from level1/activations/5_Softmax
    - MatMul from level1/matmul/1_MatMul
    """

    def __init__(
        self,
        d_model: int = 1024,
        num_heads: int = 16,
        d_kv: int = 64,
        d_ff: int = 2816,
        vocab_size: int = 32128,
        num_encoder_layers: int = 24,
        num_decoder_layers: int = 24,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        is_gated_act: bool = True,
        dense_act_fn: str = "gelu_new",
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.tie_word_embeddings = tie_word_embeddings

        # Level1 Embedding (same as HF: shared)
        self.shared = Embedding(vocab_size, d_model)

        # Encoder blocks (same as HF: encoder.block.{i})
        self.encoder_blocks = nn.ModuleList([
            T5Block(
                d_model=d_model,
                num_heads=num_heads,
                d_kv=d_kv,
                d_ff=d_ff,
                is_decoder=False,
                has_relative_attention_bias=(i == 0),
                relative_attention_num_buckets=relative_attention_num_buckets,
                relative_attention_max_distance=relative_attention_max_distance,
                is_gated_act=is_gated_act,
                act_fn=dense_act_fn,
            )
            for i in range(num_encoder_layers)
        ])
        self.encoder_final_layer_norm = T5LayerNorm(d_model)

        # Decoder blocks (same as HF: decoder.block.{i})
        self.decoder_blocks = nn.ModuleList([
            T5Block(
                d_model=d_model,
                num_heads=num_heads,
                d_kv=d_kv,
                d_ff=d_ff,
                is_decoder=True,
                has_relative_attention_bias=(i == 0),
                relative_attention_num_buckets=relative_attention_num_buckets,
                relative_attention_max_distance=relative_attention_max_distance,
                is_gated_act=is_gated_act,
                act_fn=dense_act_fn,
            )
            for i in range(num_decoder_layers)
        ])
        self.decoder_final_layer_norm = T5LayerNorm(d_model)

        # Level1 Linear for LM head
        self.lm_head = Linear(d_model, vocab_size, bias=False)

        # Optionally tie weights
        if tie_word_embeddings:
            self.lm_head.weight = self.shared.embedding.weight

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the encoder stack."""
        hidden_states = self.shared(input_ids)
        position_bias = None

        for block in self.encoder_blocks:
            outputs = block(hidden_states, position_bias=position_bias)
            hidden_states = outputs[0]
            position_bias = outputs[1]

        hidden_states = self.encoder_final_layer_norm(hidden_states)
        return hidden_states

    def _build_causal_mask(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build causal mask for decoder self-attention."""
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=dtype) * torch.finfo(dtype).min,
            diagonal=1,
        )
        return causal_mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass matching HF T5ForConditionalGeneration (no-cache mode).

        Args:
            input_ids: (batch_size, encoder_seq_len) encoder input IDs
            decoder_input_ids: (batch_size, decoder_seq_len) decoder input IDs

        Returns:
            logits: (batch_size, decoder_seq_len, vocab_size)
        """
        # Encode
        encoder_hidden_states = self._encode(input_ids)

        # Decode
        decoder_hidden = self.shared(decoder_input_ids)
        seq_len = decoder_input_ids.shape[1]
        causal_mask = self._build_causal_mask(
            seq_len, decoder_input_ids.device, decoder_hidden.dtype
        )

        position_bias = None
        encoder_decoder_position_bias = None

        for block in self.decoder_blocks:
            outputs = block(
                decoder_hidden,
                attention_mask=causal_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
            )
            decoder_hidden = outputs[0]
            position_bias = outputs[1]
            encoder_decoder_position_bias = outputs[2]

        decoder_hidden = self.decoder_final_layer_norm(decoder_hidden)
        return self.lm_head(decoder_hidden)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        pad_token_id: int = 0,
        return_logits: bool = False,
        **kwargs,
    ):
        """
        Greedy autoregressive generation for seq2seq.

        Args:
            input_ids: (batch_size, encoder_seq_len) encoder input IDs
            decoder_input_ids: (batch_size, 1) starting decoder tokens.
                If None, uses pad_token_id.
            max_new_tokens: Number of tokens to generate.
                If 0, runs a full forward pass with a single decoder token
                and returns the logits.
            pad_token_id: Token to start generation with.
            return_logits: If True, also return logits.

        Returns:
            If return_logits=False: generated_ids (batch_size, prompt+generated)
            If return_logits=True: (generated_ids, [logits_list])
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Encode once
        encoder_hidden_states = self._encode(input_ids)

        if decoder_input_ids is None:
            decoder_input_ids = torch.full(
                (batch_size, 1), pad_token_id, dtype=torch.long, device=device
            )

        # Prefill-only mode
        if max_new_tokens == 0:
            logits = self.forward(input_ids, decoder_input_ids)
            if return_logits:
                return decoder_input_ids, [logits]
            return decoder_input_ids

        # Autoregressive generation
        generated = [decoder_input_ids]
        all_logits = [] if return_logits else None
        cur_decoder_ids = decoder_input_ids

        for step in range(max_new_tokens):
            # Full decoder forward (no KV cache - simple but correct)
            decoder_hidden = self.shared(cur_decoder_ids)
            seq_len = cur_decoder_ids.shape[1]
            causal_mask = self._build_causal_mask(
                seq_len, device, decoder_hidden.dtype
            )

            position_bias = None
            enc_dec_position_bias = None
            for block in self.decoder_blocks:
                outputs = block(
                    decoder_hidden,
                    attention_mask=causal_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_decoder_position_bias=enc_dec_position_bias,
                )
                decoder_hidden = outputs[0]
                position_bias = outputs[1]
                enc_dec_position_bias = outputs[2]

            decoder_hidden = self.decoder_final_layer_norm(decoder_hidden)
            logits = self.lm_head(decoder_hidden)

            if return_logits:
                all_logits.append(logits)

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur_decoder_ids = torch.cat([cur_decoder_ids, next_token], dim=1)
            generated.append(next_token)

        generated_ids = torch.cat(generated, dim=1)
        if return_logits:
            return generated_ids, all_logits
        return generated_ids


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 4
encoder_seq_len = 256
decoder_seq_len = 128
vocab_size = 32128
d_model = 1024
num_heads = 16
d_kv = 64
d_ff = 2816
num_layers = 6


def get_inputs():
    input_ids = torch.randint(0, vocab_size, (batch_size, encoder_seq_len))
    decoder_input_ids = torch.randint(0, vocab_size, (batch_size, decoder_seq_len))
    return [input_ids, decoder_input_ids]


def get_init_inputs():
    return [{
        'd_model': d_model,
        'num_heads': num_heads,
        'd_kv': d_kv,
        'd_ff': d_ff,
        'num_encoder_layers': num_layers,
        'num_decoder_layers': num_layers,
        'vocab_size': vocab_size,
        'is_gated_act': True,
        'dense_act_fn': 'gelu_new',
    }]
