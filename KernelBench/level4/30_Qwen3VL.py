"""
Qwen3-VL Vision-Language Model

Implements the Qwen3-VL architecture aligned with HuggingFace
Qwen3VLForConditionalGeneration:
- Vision encoder: 3D patch embedding (Conv3d) + interpolated position embeddings
  + vision rotary pos emb + Transformer blocks with per-image attention
  + PatchMerger + DeepStack (multi-level visual features)
- Language model (Qwen3 decoder): embed_tokens + interleaved M-RoPE + GQA
  with QK-norm + SiLU-gated MLP + RMSNorm
- lm_head

Key differences from Qwen2-VL (13_Qwen2VL.py):
- Vision encoder uses gelu_pytorch_tanh activation (not quick_gelu)
- Vision encoder has nn.Embedding-based position embeddings with bilinear
  interpolation (fast_pos_embed_interpolate)
- DeepStack: intermediate vision features injected into early decoder layers
- PatchMerger has different structure (linear_fc1/linear_fc2, postshuffle_norm)
- Conv3d patch embedding has bias=True
- Text attention has q_norm/k_norm (RMSNorm on head_dim)
- Text attention has no bias on q/k/v/o projections
- Interleaved M-RoPE instead of chunked M-RoPE
- Different get_rope_index (timestamps for videos, llm_grid_t always 1)
- 4D position_ids: [text_pos; temporal; height; width]

HuggingFace weight structure (Qwen3VLForConditionalGeneration):
  model.visual.patch_embed.proj.{weight,bias}             # Conv3d
  model.visual.pos_embed.weight                            # nn.Embedding
  model.visual.blocks.{i}.norm1.{weight,bias}              # LayerNorm
  model.visual.blocks.{i}.attn.qkv.{weight,bias}          # Linear(dim, dim*3)
  model.visual.blocks.{i}.attn.proj.{weight,bias}          # Linear(dim, dim)
  model.visual.blocks.{i}.norm2.{weight,bias}              # LayerNorm
  model.visual.blocks.{i}.mlp.linear_fc1.{weight,bias}    # Linear
  model.visual.blocks.{i}.mlp.linear_fc2.{weight,bias}    # Linear
  model.visual.merger.norm.{weight,bias}                   # LayerNorm
  model.visual.merger.linear_fc1.{weight,bias}             # Linear
  model.visual.merger.linear_fc2.{weight,bias}             # Linear
  model.visual.deepstack_merger_list.{j}.norm.{weight,bias}
  model.visual.deepstack_merger_list.{j}.linear_fc1.{weight,bias}
  model.visual.deepstack_merger_list.{j}.linear_fc2.{weight,bias}
  model.visual.rotary_pos_emb.inv_freq                     # buffer
  model.language_model.embed_tokens.weight
  model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
  model.language_model.layers.{i}.self_attn.q_norm.weight
  model.language_model.layers.{i}.self_attn.k_norm.weight
  model.language_model.layers.{i}.input_layernorm.weight
  model.language_model.layers.{i}.post_attention_layernorm.weight
  model.language_model.layers.{i}.mlp.{gate,up,down}_proj.weight
  model.language_model.norm.weight
  model.language_model.rotary_emb.inv_freq                 # buffer
  lm_head.weight

Tested against: Qwen/Qwen3-VL-8B-Instruct

This model uses level1 operators from KernelBench:
- Linear from level1/matmul/_10_Linear
- LayerNorm from level1/normalization/_6_LayerNorm
- RMSNorm from level1/normalization/_4_RMSNorm
- Embedding from level1/embeddings/_2_Embedding
- PatchEmbed3D from level1/vision/_2_PatchEmbed3D
- GELU from level1/activations/_8_GELU
- Swish (SiLU) from level1/activations/_7_Swish
- RotaryEmbedding from level1/embeddings/_1_RotaryEmbedding
- MultimodalRotaryEmbedding from level1/embeddings/_5_MultimodalRotaryEmbedding
- VisionRotaryEmbedding from level1/embeddings/_6_VisionRotaryEmbedding
- ScaledDotProductAttention(mode="eager") from level1/attention/_2_Attention

Note: Using level1 wrappers changes the state-dict key names (e.g.
LayerNorm adds ".ln.", Embedding adds ".embedding."). The weight-copying
logic in test_hf_alignment.py unwraps these prefixes when mapping KB keys
to HF keys.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

# Import level1 operators
from ..level1.matmul._10_Linear import Model as Linear
from ..level1.normalization._6_LayerNorm import Model as LayerNorm
from ..level1.normalization._4_RMSNorm import Model as RMSNorm
from ..level1.embeddings._2_Embedding import Model as Embedding
from ..level1.vision._2_PatchEmbed3D import Model as PatchEmbed3D
from ..level1.activations._8_GELU import Model as GELU
from ..level1.activations._7_Swish import Model as Swish
from ..level1.embeddings._1_RotaryEmbedding import Model as RotaryEmbedding
from ..level1.embeddings._5_MultimodalRotaryEmbedding import Model as MultimodalRotaryEmbedding
from ..level1.embeddings._6_VisionRotaryEmbedding import Model as VisionRotaryEmbedding
from ..level1.attention._2_Attention import ScaledDotProductAttention


# ============================================================================
# Model Variants
# ============================================================================

VARIANTS: Dict[str, str] = {
    "8B": "Qwen/Qwen3-VL-8B-Instruct",
}


# ============================================================================
# Vision Encoder Components
# ============================================================================


class VisionMLP(nn.Module):
    """Vision encoder MLP with configurable activation.
    Uses level1 Linear and GELU operators."""
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "gelu_pytorch_tanh"):
        super().__init__()
        self.linear_fc1 = Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = Linear(intermediate_size, hidden_size, bias=True)
        if hidden_act == "gelu_pytorch_tanh":
            self.act_fn = GELU(approximate="tanh")
        elif hidden_act == "gelu":
            self.act_fn = GELU()
        else:
            self.act_fn = GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionAttention(nn.Module):
    """Vision encoder attention with rotary position embedding.

    Processes variable-length sequences defined by cu_seqlens (cumulative
    sequence lengths), applying per-image attention (not across images).

    Uses level1 operators:
    - Linear for QKV and output projections
    - RotaryEmbedding for the core rotate_half operation
    - ScaledDotProductAttention(mode="eager") for attention math
      (matches HuggingFace eager attention numerically)
    """
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = Linear(dim, dim * 3, bias=True)
        self.proj = Linear(dim, dim, bias=True)
        self.scaling = self.head_dim ** -0.5
        # Level1 RotaryEmbedding for the core rotate_half operation
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=1, base=10000.0)
        # Level1 ScaledDotProductAttention with eager mode for HF-aligned attention
        self.sdpa = ScaledDotProductAttention(mode="eager")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        # Apply vision rotary embedding using level1 RotaryEmbedding
        cos, sin = position_embeddings
        orig_dtype = query_states.dtype
        q_float, k_float = query_states.float(), key_states.float()
        cos_f = cos.unsqueeze(-2).float()
        sin_f = sin.unsqueeze(-2).float()
        query_states, key_states = self.rotary.apply_rotary(
            q_float, k_float, cos_f, sin_f
        )
        query_states = query_states.to(orig_dtype)
        key_states = key_states.to(orig_dtype)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=2)
            for tensor in (query_states, key_states, value_states)
        ]

        attn_outputs = []
        for q, k, v in zip(*splits):
            # Level1 ScaledDotProductAttention (eager mode) for HF-aligned attention
            attn_out = self.sdpa(q, k, v, scale=self.scaling)
            attn_out = attn_out.transpose(1, 2).contiguous()
            attn_outputs.append(attn_out)

        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3VLVisionBlock(nn.Module):
    """Vision transformer block. Uses level1 LayerNorm operator."""
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int,
                 hidden_act: str = "gelu_pytorch_tanh"):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = LayerNorm(hidden_size, eps=1e-6)
        self.attn = VisionAttention(hidden_size, num_heads)
        self.mlp = VisionMLP(hidden_size, intermediate_size, hidden_act)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class PatchMerger(nn.Module):
    """Merge spatial patches with optional post-shuffle normalization.
    Uses level1 LayerNorm, Linear, and GELU operators."""
    def __init__(self, hidden_size: int, out_hidden_size: int,
                 spatial_merge_size: int = 2, use_postshuffle_norm: bool = False):
        super().__init__()
        self.hidden_size_merged = hidden_size * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = LayerNorm(
            self.hidden_size_merged if use_postshuffle_norm else hidden_size, eps=1e-6
        )
        self.linear_fc1 = Linear(self.hidden_size_merged, self.hidden_size_merged, bias=True)
        self.act_fn = GELU()
        self.linear_fc2 = Linear(self.hidden_size_merged, out_hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(
            x.view(-1, self.hidden_size_merged) if self.use_postshuffle_norm else x
        ).view(-1, self.hidden_size_merged)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class VisionEncoder(nn.Module):
    """Qwen3-VL Vision Transformer encoder with DeepStack."""
    def __init__(
        self,
        depth: int = 27,
        hidden_size: int = 1152,
        out_hidden_size: int = 3584,
        num_heads: int = 16,
        intermediate_size: int = 4304,
        in_channels: int = 3,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        hidden_act: str = "gelu_pytorch_tanh",
        num_position_embeddings: int = 2304,
        deepstack_visual_indexes: Optional[List[int]] = None,
    ):
        super().__init__()
        if deepstack_visual_indexes is None:
            deepstack_visual_indexes = [8, 16, 24]

        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.spatial_merge_unit = spatial_merge_size * spatial_merge_size
        self.deepstack_visual_indexes = deepstack_visual_indexes

        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
            bias=True,
        )

        # Learnable position embeddings with bilinear interpolation
        self.pos_embed = nn.Embedding(num_position_embeddings, hidden_size)
        self.num_grid_per_side = int(num_position_embeddings ** 0.5)

        head_dim = hidden_size // num_heads
        # Level1 VisionRotaryEmbedding for 2D spatial position embeddings
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([
            Qwen3VLVisionBlock(hidden_size, num_heads, intermediate_size, hidden_act)
            for _ in range(depth)
        ])

        self.merger = PatchMerger(
            hidden_size=hidden_size,
            out_hidden_size=out_hidden_size,
            spatial_merge_size=spatial_merge_size,
            use_postshuffle_norm=False,
        )

        # DeepStack mergers for intermediate visual features
        self.deepstack_merger_list = nn.ModuleList([
            PatchMerger(
                hidden_size=hidden_size,
                out_hidden_size=out_hidden_size,
                spatial_merge_size=spatial_merge_size,
                use_postshuffle_norm=True,
            )
            for _ in range(len(deepstack_visual_indexes))
        ])

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Bilinear interpolation of position embeddings."""
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            merged_hidden_states: (total_merged_patches, out_hidden_size)
            deepstack_features: list of (total_merged_patches, out_hidden_size)
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        # Compute vision rotary position embeddings using level1 operator
        position_embeddings = self.rotary_pos_emb(grid_thw, self.spatial_merge_size)

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        merged_hidden_states = self.merger(hidden_states)
        return merged_hidden_states, deepstack_feature_lists


# ============================================================================
# Language Model (Decoder) Components
# ============================================================================

class Qwen3VLAttention(nn.Module):
    """Multi-headed attention with GQA, QK-norm, and interleaved M-RoPE, with optional KV cache.

    Uses level1 operators:
    - Linear for Q/K/V/O projections
    - RMSNorm for QK normalization
    - RotaryEmbedding for the core rotate_half operation
    - ScaledDotProductAttention(mode="eager") for attention math
      (handles GQA and matches HuggingFace eager attention numerically)
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.scaling = head_dim ** -0.5

        # No bias on projections (attention_bias=False in Qwen3)
        self.q_proj = Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = Linear(num_heads * head_dim, hidden_size, bias=False)

        # QK normalization (Qwen3 specific) - level1 RMSNorm
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps, learnable_weight=True, dim=-1)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps, learnable_weight=True, dim=-1)
        # Level1 RotaryEmbedding for the core rotate_half operation
        self.rotary = RotaryEmbedding(head_dim, max_seq_len=1, base=10000.0)
        # Level1 ScaledDotProductAttention with eager mode for HF-aligned attention
        self.sdpa = ScaledDotProductAttention(mode="eager")

        # KV cache (populated during generation)
        self._cached_k: Optional[torch.Tensor] = None
        self._cached_v: Optional[torch.Tensor] = None

    def reset_cache(self):
        self._cached_k = None
        self._cached_v = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # Apply pre-assembled M-RoPE cos/sin using level1 RotaryEmbedding rotate_half
        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)  # (batch, 1, seq, head_dim)
        sin = sin.unsqueeze(1)
        query_states, key_states = self.rotary.apply_rotary(
            query_states, key_states, cos, sin
        )

        # KV cache: append and use full history
        if use_cache:
            if self._cached_k is not None:
                key_states = torch.cat([self._cached_k, key_states], dim=2)
                value_states = torch.cat([self._cached_v, value_states], dim=2)
            self._cached_k = key_states
            self._cached_v = value_states

        # Level1 ScaledDotProductAttention (eager mode) handles GQA + attention math
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_output = self.sdpa(
            query_states, key_states, value_states,
            attn_mask=attn_mask, scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class Qwen3MLP(nn.Module):
    """SiLU-gated MLP (Qwen3 style)."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.swish(self.gate_proj(x)) * self.up_proj(x))


class Qwen3VLDecoderLayer(nn.Module):
    """Decoder layer with RMSNorm, QK-norm attention, MLP."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.self_attn = Qwen3VLAttention(
            hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps,
        )
        self.mlp = Qwen3MLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, learnable_weight=True, dim=-1)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, learnable_weight=True, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Qwen3-VL vision-language model aligned with HuggingFace
    Qwen3VLForConditionalGeneration.

    Uses level1 operators from KernelBench:
    - Linear from level1/matmul/_10_Linear
    - LayerNorm from level1/normalization/_6_LayerNorm
    - RMSNorm from level1/normalization/_4_RMSNorm
    - Embedding from level1/embeddings/_2_Embedding
    - PatchEmbed3D from level1/vision/_2_PatchEmbed3D
    - GELU from level1/activations/_8_GELU
    - Swish/SiLU from level1/activations/_7_Swish
    - RotaryEmbedding from level1/embeddings/_1_RotaryEmbedding
    - MultimodalRotaryEmbedding from level1/embeddings/_5_MultimodalRotaryEmbedding
    - VisionRotaryEmbedding from level1/embeddings/_6_VisionRotaryEmbedding
    - ScaledDotProductAttention(mode="eager") from level1/attention/_2_Attention

    Supports variants: 8B
    """

    VARIANTS = VARIANTS

    def __init__(
        self,
        # Vision config
        vision_depth: int = 27,
        vision_hidden_size: int = 1152,
        vision_out_hidden_size: int = 3584,
        vision_num_heads: int = 16,
        vision_intermediate_size: int = 4304,
        vision_hidden_act: str = "gelu_pytorch_tanh",
        in_channels: int = 3,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        num_position_embeddings: int = 2304,
        deepstack_visual_indexes: Optional[List[int]] = None,
        # LLM config
        hidden_size: int = 3584,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        intermediate_size: int = 18944,
        vocab_size: int = 151936,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 5000000.0,
        mrope_section: Optional[List[int]] = None,
        # Special token ids
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        **kwargs,
    ):
        super().__init__()

        if mrope_section is None:
            mrope_section = [24, 20, 20]
        if deepstack_visual_indexes is None:
            deepstack_visual_indexes = [8, 16, 24]

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.spatial_merge_size = spatial_merge_size
        self.mrope_section = mrope_section
        self.deepstack_visual_indexes = deepstack_visual_indexes

        # Vision encoder
        self.visual = VisionEncoder(
            depth=vision_depth,
            hidden_size=vision_hidden_size,
            out_hidden_size=vision_out_hidden_size,
            num_heads=vision_num_heads,
            intermediate_size=vision_intermediate_size,
            in_channels=in_channels,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            spatial_merge_size=spatial_merge_size,
            hidden_act=vision_hidden_act,
            num_position_embeddings=num_position_embeddings,
            deepstack_visual_indexes=deepstack_visual_indexes,
        )

        # Language model
        self.embed_tokens = Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            Qwen3VLDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps, learnable_weight=True, dim=-1)

        # Level1 MultimodalRotaryEmbedding for LLM M-RoPE (interleaved mode for Qwen3-VL)
        self.rotary_emb = MultimodalRotaryEmbedding(
            head_dim, base=rope_theta, mrope_section=mrope_section, mode="interleaved",
        )

        # LM head
        self.lm_head = Linear(hidden_size, vocab_size, bias=False)

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Add DeepStack visual features to decoder hidden states."""
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate 3D rope index for Qwen3-VL.
        Different from Qwen2-VL: uses timestamps for videos (llm_grid_t always 1).
        """
        # Split video_grid_thw by temporal dimension (timestamps)
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.image_token_id
        video_token_id = self.video_token_id
        vision_start_token_id = self.vision_start_token_id
        mrope_position_deltas = []

        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1],
                dtype=input_ids.dtype, device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids_i in enumerate(total_input_ids):
                input_ids_i = input_ids_i[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids_i == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids_i[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids_i.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas

    def reset_cache(self):
        """Reset KV caches in all attention layers."""
        for layer in self.layers:
            layer.self_attn.reset_cache()

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass matching HuggingFace Qwen3VLForConditionalGeneration.

        Args:
            use_cache: if True, use and update KV caches for generation

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Text embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # Track visual masks and deepstack features
        image_mask = None
        deepstack_image_embeds = None
        video_mask = None
        deepstack_video_embeds = None

        # Process images through vision encoder
        if pixel_values is not None and image_grid_thw is not None:
            pixel_values = pixel_values.to(dtype=self.visual.patch_embed.proj.weight.dtype)
            image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = (input_ids == self.image_token_id)
            special_image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask.to(inputs_embeds.device), image_embeds
            )

        # Process videos similarly
        if pixel_values_videos is not None and video_grid_thw is not None:
            pixel_values_videos = pixel_values_videos.to(
                dtype=self.visual.patch_embed.proj.weight.dtype
            )
            video_embeds, deepstack_video_embeds = self.visual(
                pixel_values_videos, grid_thw=video_grid_thw
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            video_mask = (input_ids == self.video_token_id)
            special_video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                special_video_mask.to(inputs_embeds.device), video_embeds
            )

        # Build visual_pos_masks and deepstack_visual_embeds
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1])
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        # Compute M-RoPE position ids
        if position_ids is None:
            position_ids, _ = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask,
            )

        # Build causal attention mask
        batch_size, seq_len = inputs_embeds.shape[:2]
        min_dtype = torch.finfo(inputs_embeds.dtype).min

        if use_cache:
            past_len = self.layers[0].self_attn._cached_k.shape[2] if self.layers[0].self_attn._cached_k is not None else 0
            total_len = past_len + seq_len
            if seq_len == 1:
                causal_mask = torch.zeros(
                    (batch_size, 1, 1, total_len),
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )
            else:
                causal_mask = torch.zeros(
                    (batch_size, 1, seq_len, total_len),
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )
                for qi in range(seq_len):
                    causal_mask[:, :, qi, past_len + qi + 1:] = min_dtype
        elif attention_mask is not None:
            causal_mask = torch.zeros(
                (batch_size, 1, seq_len, seq_len),
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            causal_triu = torch.triu(
                torch.ones(seq_len, seq_len, device=inputs_embeds.device, dtype=torch.bool),
                diagonal=1,
            )
            causal_mask.masked_fill_(causal_triu.unsqueeze(0).unsqueeze(0), min_dtype)
            padding_positions = (attention_mask == 0)
            causal_mask.masked_fill_(
                padding_positions[:, None, None, :], min_dtype
            )
        else:
            causal_mask = torch.zeros(
                (1, 1, seq_len, seq_len),
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            causal_triu = torch.triu(
                torch.ones(seq_len, seq_len, device=inputs_embeds.device, dtype=torch.bool),
                diagonal=1,
            )
            causal_mask.masked_fill_(causal_triu.unsqueeze(0).unsqueeze(0), min_dtype)

        # Compute rotary embeddings
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # Decoder layers with DeepStack injection
        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
            )
            # DeepStack: inject visual features into early decoder layers
            # Only during prefill (not during decode steps)
            if not use_cache or self.layers[0].self_attn._cached_k is None or self.layers[0].self_attn._cached_k.shape[2] == seq_len:
                if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                    hidden_states = self._deepstack_process(
                        hidden_states,
                        visual_pos_masks,
                        deepstack_visual_embeds[layer_idx],
                    )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 20,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Greedy autoregressive generation with KV caching.

        Returns:
            generated_ids: (batch, prompt_len + max_new_tokens)
        """
        self.reset_cache()
        batch_size, prompt_len = input_ids.shape

        position_ids, mrope_position_deltas = self.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, attention_mask,
        )

        # Prefill
        logits = self.forward(
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            pixel_values_videos=pixel_values_videos,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [input_ids, next_token]

        # Decode loop
        for step in range(max_new_tokens - 1):
            seq_pos = prompt_len + step + mrope_position_deltas.squeeze(1)
            new_position_ids = seq_pos.unsqueeze(0).unsqueeze(-1).expand(3, batch_size, 1).long()

            logits = self.forward(
                next_token,
                position_ids=new_position_ids,
                use_cache=True,
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)

        return torch.cat(generated, dim=1)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 1
sequence_length = 256
vocab_size = 151936


def get_inputs():
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return [input_ids]


def get_init_inputs():
    return [{
        'vision_depth': 4,
        'vision_hidden_size': 1152,
        'vision_out_hidden_size': 3584,
        'vision_num_heads': 16,
        'vision_intermediate_size': 4304,
        'hidden_size': 3584,
        'num_hidden_layers': 4,
        'num_attention_heads': 28,
        'num_key_value_heads': 4,
        'intermediate_size': 18944,
        'vocab_size': vocab_size,
        'deepstack_visual_indexes': [1, 2, 3],
    }]
