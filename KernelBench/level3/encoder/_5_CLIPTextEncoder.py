"""
CLIP Text Encoder

Implements CLIPTextModel and CLIPTextModelWithProjection aligned with
HuggingFace's ``transformers`` library.

Architecture:
  - Token embedding + positional embedding
  - N x CLIPEncoderLayer:
      - LayerNorm -> SelfAttention (causal mask) -> residual
      - LayerNorm -> MLP (fc1 -> act -> fc2) -> residual
  - final_layer_norm
  - (optional) text_projection linear layer

Activation functions:
  - CLIPTextModel (text_encoder):   QuickGELU  x * sigmoid(1.702 * x)
  - CLIPTextModelWithProjection (text_encoder_2): GELU (exact)

HuggingFace weight structure:
  text_model.embeddings.token_embedding.weight
  text_model.embeddings.position_embedding.weight
  text_model.encoder.layers.{i}.self_attn.{q_proj,k_proj,v_proj,out_proj}.{weight,bias}
  text_model.encoder.layers.{i}.layer_norm1.{weight,bias}
  text_model.encoder.layers.{i}.mlp.fc1.{weight,bias}
  text_model.encoder.layers.{i}.mlp.fc2.{weight,bias}
  text_model.encoder.layers.{i}.layer_norm2.{weight,bias}
  text_model.final_layer_norm.{weight,bias}
  text_projection.weight  (CLIPTextModelWithProjection only, no bias)

Level1 operators used:
  - LayerNorm                  from level1/normalization/_6_LayerNorm
  - Linear                     from level1/matmul/_10_Linear
  - Embedding                  from level1/embeddings/_2_Embedding
  - GELU                       from level1/activations/_8_GELU
  - ScaledDotProductAttention  from level1/attention/_2_Attention
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

# Level1 operator imports
from KernelBench.level1.normalization._6_LayerNorm import Model as LayerNorm
from KernelBench.level1.matmul._10_Linear import Model as Linear
from KernelBench.level1.embeddings._2_Embedding import Model as Embedding
from KernelBench.level1.activations._8_GELU import Model as GELUAct
from KernelBench.level1.attention._2_Attention import ScaledDotProductAttention


# ============================================================================
# Activation helpers
# ============================================================================

class QuickGELU(nn.Module):
    """QuickGELU: x * sigmoid(1.702 * x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


# ============================================================================
# CLIP MLP
# ============================================================================

class CLIPMLP(nn.Module):
    """CLIP feed-forward network: fc1 -> activation -> fc2."""

    def __init__(self, hidden_size: int, intermediate_size: int,
                 hidden_act: str = "quick_gelu"):
        super().__init__()
        self.fc1 = Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = Linear(intermediate_size, hidden_size, bias=True)
        if hidden_act == "quick_gelu":
            self.activation_fn = QuickGELU()
        elif hidden_act == "gelu":
            self.activation_fn = GELUAct(approximate='none')
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# ============================================================================
# CLIP Attention
# ============================================================================

class CLIPAttention(nn.Module):
    """Multi-head self-attention with causal masking for CLIP text model."""

    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = Linear(hidden_size, hidden_size, bias=True)
        self.attn = ScaledDotProductAttention()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length, embed_dim = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Combine masks
        combined_mask = None
        if attention_mask is not None and causal_attention_mask is not None:
            combined_mask = attention_mask + causal_attention_mask
        elif causal_attention_mask is not None:
            combined_mask = causal_attention_mask
        elif attention_mask is not None:
            combined_mask = attention_mask

        attn_output = self.attn(
            q, k, v,
            attn_mask=combined_mask,
            scale=self.scale,
        )

        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, None


# ============================================================================
# CLIP Encoder Layer
# ============================================================================

class CLIPEncoderLayer(nn.Module):
    """Single CLIP transformer encoder layer."""

    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_attention_heads: int, hidden_act: str = "quick_gelu",
                 layer_norm_eps: float = 1e-5):
        super().__init__()
        self.self_attn = CLIPAttention(hidden_size, num_attention_heads)
        self.layer_norm1 = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = CLIPMLP(hidden_size, intermediate_size, hidden_act)
        self.layer_norm2 = LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states, attention_mask, causal_attention_mask,
            output_attentions)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


# ============================================================================
# CLIP Text Embeddings
# ============================================================================

class CLIPTextEmbeddings(nn.Module):
    """Token + positional embeddings for CLIP text model."""

    def __init__(self, vocab_size: int, hidden_size: int,
                 max_position_embeddings: int = 77):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, hidden_size)
        self.position_embedding = Embedding(max_position_embeddings,
                                             hidden_size)

    def forward(self, input_ids: torch.Tensor,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_length = input_ids.shape[-1]
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, device=input_ids.device).unsqueeze(0)
        inputs_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        return inputs_embeds + position_embeds


# ============================================================================
# Helper: create 4D causal attention mask
# ============================================================================

def _create_4d_causal_attention_mask(
    input_shape: Tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create a 4D causal mask (batch, 1, seq, seq) with -inf above diagonal."""
    batch_size, seq_length = input_shape
    mask = torch.full((seq_length, seq_length), float("-inf"),
                      dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)


# ============================================================================
# Output containers
# ============================================================================

class _BaseModelOutput:
    """Minimal output container with hidden_states."""
    __slots__ = ("last_hidden_state", "hidden_states", "attentions")

    def __init__(self, last_hidden_state, hidden_states=None, attentions=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states
        self.attentions = attentions


class _BaseModelOutputWithPooling:
    """Output container with pooler_output."""
    __slots__ = ("last_hidden_state", "pooler_output", "hidden_states",
                 "attentions")

    def __init__(self, last_hidden_state, pooler_output=None,
                 hidden_states=None, attentions=None):
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output
        self.hidden_states = hidden_states
        self.attentions = attentions

    def __getitem__(self, idx):
        """Support indexing: [0] = last_hidden_state, [1] = pooler_output."""
        if idx == 0:
            return self.last_hidden_state
        elif idx == 1:
            return self.pooler_output
        raise IndexError(idx)


class _CLIPTextModelOutput:
    """Output container for CLIPTextModelWithProjection."""
    __slots__ = ("text_embeds", "last_hidden_state", "hidden_states",
                 "attentions")

    def __init__(self, text_embeds, last_hidden_state=None,
                 hidden_states=None, attentions=None):
        self.text_embeds = text_embeds
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states
        self.attentions = attentions

    def __getitem__(self, idx):
        """Support indexing: [0] = text_embeds."""
        if idx == 0:
            return self.text_embeds
        raise IndexError(idx)


# ============================================================================
# CLIPTextModel (text_encoder — no projection)
# ============================================================================

class CLIPTextModel(nn.Module):
    """CLIP Text Model (no projection layer).

    Used as ``text_encoder`` in SDXL and SD3.5 pipelines.
    Returns BaseModelOutputWithPooling with hidden_states support.
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        max_position_embeddings: int = 77,
        hidden_act: str = "quick_gelu",
        layer_norm_eps: float = 1e-5,
        projection_dim: int = 768,
    ):
        super().__init__()
        self._hidden_size = hidden_size

        # text_model sub-module (matches HF structure)
        self.text_model = _CLIPTextTransformer(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> _BaseModelOutputWithPooling:
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


# ============================================================================
# CLIPTextModelWithProjection (text_encoder_2 — with projection)
# ============================================================================

class CLIPTextModelWithProjection(nn.Module):
    """CLIP Text Model with projection layer.

    Used as ``text_encoder_2`` in SDXL and SD3.5 pipelines.
    Returns CLIPTextModelOutput with text_embeds (projected pooled output).
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        hidden_size: int = 1280,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 20,
        max_position_embeddings: int = 77,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        projection_dim: int = 1280,
    ):
        super().__init__()
        self._projection_dim = projection_dim

        # Store config for pipeline compatibility
        self._config_obj = type("Config", (), {
            "projection_dim": projection_dim,
        })()

        self.text_model = _CLIPTextTransformer(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )
        self.text_projection = Linear(hidden_size, projection_dim, bias=False)

    @property
    def config(self):
        return self._config_obj

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> _CLIPTextModelOutput:
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = text_outputs.pooler_output
        text_embeds = self.text_projection(pooled_output)

        return _CLIPTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=text_outputs.last_hidden_state,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )


# ============================================================================
# Internal: CLIPTextTransformer
# ============================================================================

class _CLIPTextTransformer(nn.Module):
    """Internal CLIP text transformer (shared by both model variants)."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        max_position_embeddings: int,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.embeddings = CLIPTextEmbeddings(
            vocab_size, hidden_size, max_position_embeddings)
        self.encoder = _CLIPEncoder(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )
        self.final_layer_norm = LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> _BaseModelOutputWithPooling:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids, position_ids)

        # Causal attention mask
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, hidden_states.device)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # Pool: take features from the EOS token (highest token id in sequence)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0],
                         device=last_hidden_state.device),
            input_ids.to(dtype=torch.int,
                         device=last_hidden_state.device).argmax(dim=-1),
        ]

        return _BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# ============================================================================
# Internal: CLIPEncoder
# ============================================================================

class _CLIPEncoder(nn.Module):
    """Stack of CLIP encoder layers."""

    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_hidden_layers: int, num_attention_heads: int,
                 hidden_act: str, layer_norm_eps: float):
        super().__init__()
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                hidden_act=hidden_act,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> _BaseModelOutput:
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = layer(
                hidden_states, attention_mask, causal_attention_mask,
                output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return _BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )
