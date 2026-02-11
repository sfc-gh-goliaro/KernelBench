"""
Qwen3-Omni-MoE Thinker Model (Multimodal: Vision + Audio + Text)

Implements the Thinker component of Qwen3-Omni-MoE aligned with HuggingFace
Qwen3OmniMoeThinkerForConditionalGeneration:
- Audio encoder: Conv2d downsampling + sinusoidal position embeddings
  + Transformer layers (Whisper-like) + projection MLP
- Vision encoder: 3D patch embedding (Conv3d) + interpolated position embeddings
  + vision rotary pos emb + Transformer blocks with per-image attention
  + PatchMerger + DeepStack (multi-level visual features)
- Language model (MoE decoder): embed_tokens + interleaved M-RoPE + GQA
  with QK-norm + Mixture-of-Experts (SparseMoeBlock) + RMSNorm
- lm_head

Key features:
- Mixture of Experts (MoE): 128 experts, top-8 routing per token
- Multimodal RoPE (M-RoPE): 3D position encoding for temporal/height/width
- DeepStack: intermediate vision features injected into early decoder layers
- Audio support: Whisper-style audio encoder with chunked convolution
- Interleaved M-RoPE for text decoder

HuggingFace weight structure (Qwen3OmniMoeThinkerForConditionalGeneration):
  thinker.audio_tower.conv2d{1,2,3}.{weight,bias}
  thinker.audio_tower.conv_out.weight
  thinker.audio_tower.positional_embedding.positional_embedding  # buffer
  thinker.audio_tower.layers.{i}.self_attn_layer_norm.{weight,bias}
  thinker.audio_tower.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}
  thinker.audio_tower.layers.{i}.self_attn.out_proj.{weight,bias}
  thinker.audio_tower.layers.{i}.final_layer_norm.{weight,bias}
  thinker.audio_tower.layers.{i}.fc1.{weight,bias}
  thinker.audio_tower.layers.{i}.fc2.{weight,bias}
  thinker.audio_tower.ln_post.{weight,bias}
  thinker.audio_tower.proj1.{weight,bias}
  thinker.audio_tower.proj2.{weight,bias}
  thinker.visual.patch_embed.proj.{weight,bias}
  thinker.visual.pos_embed.pos_embed.weight          # nn.Embedding (via level1 InterpolatedPositionEmbedding)
  thinker.visual.blocks.{i}.norm1.{weight,bias}
  thinker.visual.blocks.{i}.attn.qkv.{weight,bias}
  thinker.visual.blocks.{i}.attn.proj.{weight,bias}
  thinker.visual.blocks.{i}.norm2.{weight,bias}
  thinker.visual.blocks.{i}.mlp.linear_fc1.{weight,bias}
  thinker.visual.blocks.{i}.mlp.linear_fc2.{weight,bias}
  thinker.visual.merger.ln_q.{weight,bias}
  thinker.visual.merger.mlp.{0,2}.{weight,bias}
  thinker.visual.merger_list.{j}.ln_q.{weight,bias}
  thinker.visual.merger_list.{j}.mlp.{0,2}.{weight,bias}
  thinker.model.embed_tokens.weight
  thinker.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
  thinker.model.layers.{i}.self_attn.q_norm.weight
  thinker.model.layers.{i}.self_attn.k_norm.weight
  thinker.model.layers.{i}.input_layernorm.weight
  thinker.model.layers.{i}.post_attention_layernorm.weight
  thinker.model.layers.{i}.mlp.experts.gate_up_proj  (MoE layers)
  thinker.model.layers.{i}.mlp.experts.down_proj      (MoE layers)
  thinker.model.layers.{i}.mlp.gate.weight             (MoE layers)
  thinker.model.layers.{i}.mlp.gate_proj.weight        (dense layers)
  thinker.model.layers.{i}.mlp.up_proj.weight          (dense layers)
  thinker.model.layers.{i}.mlp.down_proj.weight        (dense layers)
  thinker.model.norm.weight
  thinker.model.rotary_emb.inv_freq
  thinker.lm_head.weight

Tested against: Qwen/Qwen3-Omni-30B-A3B-Instruct

This model uses level1 operators from KernelBench:
- Linear from level1/matmul/_10_Linear
- Conv2d from level1/convolutions/_8_Conv2d_Square
- LayerNorm from level1/normalization/_6_LayerNorm
- RMSNorm from level1/normalization/_4_RMSNorm
- Embedding from level1/embeddings/_2_Embedding
- SinusoidalPosEmbed from level1/embeddings/_3_SinusoidalPosEmbed
- PatchEmbed3D from level1/vision/_2_PatchEmbed3D
- GELU from level1/activations/_8_GELU
- Swish (SiLU) from level1/activations/_7_Swish
- Softmax from level1/activations/_5_Softmax
- RotaryEmbedding from level1/embeddings/_1_RotaryEmbedding
- MultimodalRotaryEmbedding from level1/embeddings/_5_MultimodalRotaryEmbedding
- VisionRotaryEmbedding from level1/embeddings/_6_VisionRotaryEmbedding
- InterpolatedPositionEmbedding from level1/embeddings/_7_InterpolatedPositionEmbedding
- ScaledDotProductAttention(mode="eager") from level1/attention/_2_Attention

Note: Using level1 wrappers changes the state-dict key names (e.g.
LayerNorm adds ".ln.", Embedding adds ".embedding."). The weight-copying
logic in test_hf_alignment.py unwraps these prefixes when mapping KB keys
to HF keys.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple

# Import level1 operators
from ..level1.matmul._10_Linear import Model as Linear
from ..level1.normalization._6_LayerNorm import Model as LayerNorm
from ..level1.normalization._4_RMSNorm import Model as RMSNorm
from ..level1.embeddings._2_Embedding import Model as Embedding
from ..level1.vision._2_PatchEmbed3D import Model as PatchEmbed3D
from ..level1.activations._8_GELU import Model as GELU
from ..level1.activations._7_Swish import Model as Swish
from ..level1.activations._5_Softmax import Model as Softmax
from ..level1.convolutions._8_Conv2d_Square import Model as Conv2d
from ..level1.embeddings._1_RotaryEmbedding import Model as RotaryEmbedding
from ..level1.embeddings._5_MultimodalRotaryEmbedding import Model as MultimodalRotaryEmbedding
from ..level1.embeddings._6_VisionRotaryEmbedding import Model as VisionRotaryEmbedding
from ..level1.embeddings._3_SinusoidalPosEmbed import Model as SinusoidalPosEmbed
from ..level1.embeddings._7_InterpolatedPositionEmbedding import Model as InterpolatedPositionEmbedding
from ..level1.attention._2_Attention import ScaledDotProductAttention


# ============================================================================
# Model Variants
# ============================================================================

VARIANTS: Dict[str, str] = {
    "30B-A3B": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
}


# ============================================================================
# Audio Encoder Components
# ============================================================================

class AudioAttention(nn.Module):
    """Multi-headed attention for audio encoder.

    Uses level1 operators:
    - Linear for Q/K/V/O projections
    - ScaledDotProductAttention(mode="eager") for attention math
      (matches HuggingFace eager attention numerically)
    """
    def __init__(self, embed_dim: int, num_heads: int, attention_dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = Linear(embed_dim, embed_dim, bias=True)
        # Level1 ScaledDotProductAttention with eager mode for HF-aligned attention
        self.sdpa = ScaledDotProductAttention(mode="eager")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        # (seq, heads, head_dim) -> (1, heads, seq, head_dim)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Level1 ScaledDotProductAttention (eager mode) for HF-aligned attention
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_output = self.sdpa(query_states, key_states, value_states,
                                attn_mask=attn_mask, scale=self.scaling)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output


class AudioEncoderLayer(nn.Module):
    """Single audio encoder transformer layer.
    Uses level1 LayerNorm, Linear, and GELU operators."""
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int,
                 activation_function: str = "gelu", attention_dropout: float = 0.0):
        super().__init__()
        self.embed_dim = d_model
        self.self_attn = AudioAttention(d_model, num_heads, attention_dropout)
        self.self_attn_layer_norm = LayerNorm(d_model)
        self.activation_fn = GELU()
        self.fc1 = Linear(d_model, ffn_dim, bias=True)
        self.fc2 = Linear(ffn_dim, d_model, bias=True)
        self.final_layer_norm = LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states


def _get_feat_extract_output_lengths(input_lengths):
    """Compute output length of audio convolutional layers."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


class AudioEncoder(nn.Module):
    """Qwen3-Omni audio encoder (Whisper-like with chunked conv)."""
    def __init__(
        self,
        num_mel_bins: int = 128,
        d_model: int = 1280,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        activation_function: str = "gelu",
        attention_dropout: float = 0.0,
        max_source_positions: int = 1500,
        output_dim: int = 2048,
        n_window: int = 50,
        n_window_infer: int = 800,
        conv_chunksize: int = 500,
        downsample_hidden_size: int = 480,
        scale_embedding: bool = False,
    ):
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.max_source_positions = max_source_positions
        self.embed_scale = math.sqrt(d_model) if scale_embedding else 1.0
        self.n_window = n_window
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize

        self.positional_embedding = SinusoidalPosEmbed(
            hidden_size=d_model, max_seq_length=max_source_positions,
            mode="concatenated",
        )
        self.layers = nn.ModuleList([
            AudioEncoderLayer(d_model, encoder_attention_heads, encoder_ffn_dim,
                              activation_function, attention_dropout)
            for _ in range(encoder_layers)
        ])
        self.ln_post = LayerNorm(d_model)

        # Conv2d downsampling
        self.conv2d1 = Conv2d(1, downsample_hidden_size, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2d2 = Conv2d(downsample_hidden_size, downsample_hidden_size, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2d3 = Conv2d(downsample_hidden_size, downsample_hidden_size, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv_out = Linear(
            downsample_hidden_size * ((((num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            d_model,
            bias=False,
        )

        # GELU activation for conv layers
        self.conv_act = GELU()

        # Projection MLP
        self.proj1 = Linear(d_model, d_model, bias=True)
        self.act = GELU()
        self.proj2 = Linear(d_model, output_dim, bias=True)

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_features: (num_mel_bins, total_mel_frames) - concatenated mel features
            feature_lens: (batch_size,) - length of each audio in mel frames
        Returns:
            hidden_states: (total_output_tokens, output_dim)
        """
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.full((chunk_num.sum(),), self.n_window * 2, dtype=torch.long, device=feature_lens.device)
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)

        # Split to chunk to avoid OOM during convolution
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = self.conv_act(self.conv2d1(chunk))
            padded_embed = self.conv_act(self.conv2d2(padded_embed))
            padded_embed = self.conv_act(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding(padded_embed.shape[1])
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]

        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

        # Note: HF's eager audio attention does NOT apply an attention mask.
        # The cu_seqlens are only used for Flash Attention's varlen path.
        # With eager attention, the audio encoder uses full bidirectional attention.
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens,
                attention_mask=None,
            )

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states


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
        else:
            self.act_fn = GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionAttention(nn.Module):
    """Vision encoder attention with rotary position embedding.

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


class VisionBlock(nn.Module):
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


class VisionPatchMerger(nn.Module):
    """Merge spatial patches. Uses mlp as ModuleList [Linear, GELU, Linear].
    Uses level1 LayerNorm, Linear, and GELU operators."""
    def __init__(self, hidden_size: int, out_hidden_size: int,
                 spatial_merge_size: int = 2, use_postshuffle_norm: bool = False):
        super().__init__()
        self.hidden_size_merged = hidden_size * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.ln_q = LayerNorm(
            self.hidden_size_merged if use_postshuffle_norm else hidden_size, eps=1e-6
        )
        self.mlp = nn.ModuleList([
            Linear(self.hidden_size_merged, self.hidden_size_merged, bias=True),
            GELU(),
            Linear(self.hidden_size_merged, out_hidden_size, bias=True),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(
            x.view(-1, self.hidden_size_merged) if self.use_postshuffle_norm else x
        ).view(-1, self.hidden_size_merged)
        for layer in self.mlp:
            x = layer(x)
        return x


class VisionEncoder(nn.Module):
    """Qwen3-Omni Vision Transformer encoder with DeepStack."""
    def __init__(
        self,
        depth: int = 27,
        hidden_size: int = 1152,
        out_hidden_size: int = 2048,
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

        # Level1 InterpolatedPositionEmbedding for bilinear-interpolated learned pos embeds
        self.pos_embed = InterpolatedPositionEmbedding(
            hidden_size=hidden_size,
            num_position_embeddings=num_position_embeddings,
            spatial_merge_size=spatial_merge_size,
        )

        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([
            VisionBlock(hidden_size, num_heads, intermediate_size, hidden_act)
            for _ in range(depth)
        ])

        self.merger = VisionPatchMerger(
            hidden_size=hidden_size,
            out_hidden_size=out_hidden_size,
            spatial_merge_size=spatial_merge_size,
            use_postshuffle_norm=False,
        )

        self.merger_list = nn.ModuleList([
            VisionPatchMerger(
                hidden_size=hidden_size,
                out_hidden_size=out_hidden_size,
                spatial_merge_size=spatial_merge_size,
                use_postshuffle_norm=True,
            )
            for _ in range(len(deepstack_visual_indexes))
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hidden_states = self.patch_embed(hidden_states)
        # Compute interpolated position embeddings using level1 operator
        pos_embeds = self.pos_embed(grid_thw)
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
                deepstack_feature = self.merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        merged_hidden_states = self.merger(hidden_states)
        return merged_hidden_states, deepstack_feature_lists


# ============================================================================
# Language Model (MoE Decoder) Components
# ============================================================================

class ThinkerTextAttention(nn.Module):
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
        head_dim: int = 64,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.scaling = head_dim ** -0.5

        self.q_proj = Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = Linear(num_heads * head_dim, hidden_size, bias=False)

        # QK normalization - level1 RMSNorm
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


class ThinkerTextMLP(nn.Module):
    """Dense SiLU-gated MLP (for mlp_only_layers or when decoder_sparse_step doesn't match).
    Uses level1 Linear and Swish operators."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.swish(self.gate_proj(x)) * self.up_proj(x))


class ThinkerTextTopKRouter(nn.Module):
    """Top-K router for MoE layers.
    Uses level1 Linear and Softmax operators."""
    def __init__(self, hidden_size: int, num_experts: int, num_experts_per_tok: int,
                 norm_topk_prob: bool = True):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.hidden_dim = hidden_size
        # Named 'weight' so state_dict key stays .weight.weight matching HF .weight
        self.weight = Linear(hidden_size, num_experts, bias=False)
        self.softmax = Softmax(dim=-1)

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.weight(hidden_states)
        router_logits = self.softmax(router_logits.float())
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        # Cast routing weights back to the input dtype (e.g. bfloat16) to match HF
        router_top_value = router_top_value.to(input_dtype)
        return router_logits, router_top_value, router_indices


class ThinkerTextExperts(nn.Module):
    """Expert module with fused gate_up_proj and down_proj parameters.
    Uses level1 Swish operator for SiLU activation."""
    def __init__(self, hidden_size: int, moe_intermediate_size: int,
                 num_experts: int, hidden_act: str = "silu"):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_size
        self.intermediate_dim = moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * moe_intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, moe_intermediate_size))
        self.swish = Swish()

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = (current_state @ self.gate_up_proj[expert_idx].T).chunk(2, dim=-1)
            current_hidden_states = self.swish(gate) * up
            current_hidden_states = current_hidden_states @ self.down_proj[expert_idx].T
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class ThinkerTextSparseMoeBlock(nn.Module):
    """Sparse Mixture of Experts block."""
    def __init__(self, hidden_size: int, moe_intermediate_size: int,
                 num_experts: int, num_experts_per_tok: int,
                 norm_topk_prob: bool = True, hidden_act: str = "silu"):
        super().__init__()
        self.experts = ThinkerTextExperts(hidden_size, moe_intermediate_size, num_experts, hidden_act)
        self.gate = ThinkerTextTopKRouter(hidden_size, num_experts, num_experts_per_tok, norm_topk_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class ThinkerTextDecoderLayer(nn.Module):
    """Decoder layer with RMSNorm, QK-norm attention, and MoE or dense MLP."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        head_dim: int = 64,
        rms_norm_eps: float = 1e-6,
        # MoE params
        use_moe: bool = True,
        moe_intermediate_size: int = 768,
        num_experts: int = 128,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.self_attn = ThinkerTextAttention(
            hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps,
        )
        if use_moe:
            self.mlp = ThinkerTextSparseMoeBlock(
                hidden_size, moe_intermediate_size, num_experts,
                num_experts_per_tok, norm_topk_prob,
            )
        else:
            self.mlp = ThinkerTextMLP(hidden_size, intermediate_size)
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
    Qwen3-Omni-MoE Thinker model aligned with HuggingFace
    Qwen3OmniMoeThinkerForConditionalGeneration.

    Supports multimodal inputs: text, images, audio.
    Uses Mixture of Experts (MoE) in the text decoder.
    """

    VARIANTS = VARIANTS

    def __init__(
        self,
        # Audio config
        audio_num_mel_bins: int = 128,
        audio_d_model: int = 1280,
        audio_encoder_layers: int = 32,
        audio_encoder_attention_heads: int = 20,
        audio_encoder_ffn_dim: int = 5120,
        audio_activation_function: str = "gelu",
        audio_max_source_positions: int = 1500,
        audio_output_dim: int = 2048,
        audio_n_window: int = 50,
        audio_n_window_infer: int = 800,
        audio_conv_chunksize: int = 500,
        audio_downsample_hidden_size: int = 480,
        audio_scale_embedding: bool = False,
        # Vision config
        vision_depth: int = 27,
        vision_hidden_size: int = 1152,
        vision_out_hidden_size: int = 2048,
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
        hidden_size: int = 2048,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        head_dim: int = 64,
        intermediate_size: int = 768,
        vocab_size: int = 152064,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        mrope_section: Optional[List[int]] = None,
        # MoE config
        num_experts: int = 128,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 768,
        decoder_sparse_step: int = 1,
        mlp_only_layers: Optional[List[int]] = None,
        norm_topk_prob: bool = True,
        # Special token ids
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        audio_token_id: int = 151675,
        vision_start_token_id: int = 151652,
        audio_start_token_id: int = 151647,
        position_id_per_seconds: int = 13,
        **kwargs,
    ):
        super().__init__()

        if mrope_section is None:
            mrope_section = [24, 20, 20]
        if deepstack_visual_indexes is None:
            deepstack_visual_indexes = [8, 16, 24]
        if mlp_only_layers is None:
            mlp_only_layers = []

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.vision_start_token_id = vision_start_token_id
        self.audio_start_token_id = audio_start_token_id
        self.position_id_per_seconds = position_id_per_seconds
        self.spatial_merge_size = spatial_merge_size
        self.mrope_section = mrope_section
        self.deepstack_visual_indexes = deepstack_visual_indexes

        # Audio encoder
        self.audio_tower = AudioEncoder(
            num_mel_bins=audio_num_mel_bins,
            d_model=audio_d_model,
            encoder_layers=audio_encoder_layers,
            encoder_attention_heads=audio_encoder_attention_heads,
            encoder_ffn_dim=audio_encoder_ffn_dim,
            activation_function=audio_activation_function,
            max_source_positions=audio_max_source_positions,
            output_dim=audio_output_dim,
            n_window=audio_n_window,
            n_window_infer=audio_n_window_infer,
            conv_chunksize=audio_conv_chunksize,
            downsample_hidden_size=audio_downsample_hidden_size,
            scale_embedding=audio_scale_embedding,
        )

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
        self.layers = nn.ModuleList()
        for layer_idx in range(num_hidden_layers):
            # Determine if this layer uses MoE or dense MLP
            use_moe = (
                (layer_idx not in mlp_only_layers) and
                (num_experts > 0 and (layer_idx + 1) % decoder_sparse_step == 0)
            )
            self.layers.append(ThinkerTextDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                use_moe=use_moe,
                moe_intermediate_size=moe_intermediate_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                norm_topk_prob=norm_topk_prob,
            ))
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps, learnable_weight=True, dim=-1)

        # Level1 MultimodalRotaryEmbedding for LLM M-RoPE (interleaved mode for Qwen3-Omni)
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
        audio_seqlens: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate 3D rope index for multimodal inputs (vision + audio + text)."""
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.image_token_id
        video_token_id = self.video_token_id
        audio_token_id = self.audio_token_id
        vision_start_token_id = self.vision_start_token_id
        audio_start_token_id = self.audio_start_token_id
        position_id_per_seconds = self.position_id_per_seconds

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is not None:
                attention_mask_bool = attention_mask == 1
            else:
                attention_mask_bool = torch.ones_like(total_input_ids, dtype=torch.bool)
            position_ids = torch.zeros(
                3, input_ids.shape[0], input_ids.shape[1],
                dtype=torch.float, device=input_ids.device,
            )
            image_idx, video_idx, audio_idx = 0, 0, 0
            for i, input_ids_i in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids_i = input_ids_i[attention_mask_bool[i]]
                image_nums, video_nums, audio_nums = 0, 0, 0
                vision_start_indices = torch.argwhere(input_ids_i == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids_i[vision_start_indices + 1]
                audio_nums = torch.sum(input_ids_i == audio_start_token_id)
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids_i.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
                multimodal_nums = image_nums + video_nums + audio_nums

                for _ in range(multimodal_nums):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    if (image_token_id in input_tokens or video_token_id in input_tokens) and (
                        remain_videos > 0 or remain_images > 0
                    ):
                        ed_vision_start = input_tokens.index(vision_start_token_id, st)
                    else:
                        ed_vision_start = len(input_tokens) + 1
                    if audio_token_id in input_tokens and remain_audios > 0:
                        ed_audio_start = input_tokens.index(audio_start_token_id, st)
                    else:
                        ed_audio_start = len(input_tokens) + 1
                    min_ed = min(ed_vision_start, ed_audio_start)

                    text_len = min_ed - st
                    if text_len != 0:
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                        st_idx += text_len

                    bos_len, eos_len = 1, 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                    st_idx += bos_len

                    if min_ed == ed_audio_start:
                        # Audio
                        audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_idx])
                        llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        llm_pos_ids_list.append(llm_pos_ids)
                        st += int(text_len + bos_len + audio_len + eos_len)
                        audio_idx += 1
                        remain_audios -= 1
                    elif min_ed == ed_vision_start and input_ids_i[ed_vision_start + 1] == image_token_id:
                        # Image
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        llm_grid_h = grid_hs[image_idx] // spatial_merge_size
                        llm_grid_w = grid_ws[image_idx] // spatial_merge_size
                        t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).float()
                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten().float()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten().float()
                        t_idx = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
                        llm_pos_ids = torch.stack([t_idx, h_index, w_index]) + st_idx
                        llm_pos_ids_list.append(llm_pos_ids)
                        image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size ** 2)
                        st += int(text_len + bos_len + image_len + eos_len)
                        image_idx += 1
                        remain_images -= 1
                    elif min_ed == ed_vision_start and input_ids_i[ed_vision_start + 1] == video_token_id:
                        # Video
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        llm_grid_h = grid_hs[video_idx] // spatial_merge_size
                        llm_grid_w = grid_ws[video_idx] // spatial_merge_size
                        t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).float()
                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten().float()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten().float()
                        t_idx = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
                        llm_pos_ids = torch.stack([t_idx, h_index, w_index]) + st_idx
                        llm_pos_ids_list.append(llm_pos_ids)
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size ** 2)
                        st += int(text_len + bos_len + video_len + eos_len)
                        video_idx += 1
                        remain_videos -= 1

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat([item.float() for item in llm_pos_ids_list], dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask_bool[i]] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids_i))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.float().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
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
        # Audio inputs
        input_features: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        # Vision inputs
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        # Common
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass matching HuggingFace Qwen3OmniMoeThinkerForConditionalGeneration.

        Args:
            use_cache: if True, use and update KV caches for generation

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Text embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # Process audio
        if input_features is not None:
            if feature_attention_mask is not None:
                audio_feature_lengths_local = torch.sum(feature_attention_mask, dim=1)
                input_features_flat = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
            else:
                input_features_flat = input_features
                audio_feature_lengths_local = audio_feature_lengths

            audio_features = self.audio_tower(input_features_flat, feature_lens=audio_feature_lengths_local)
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_audio_mask = (input_ids == self.audio_token_id)
            special_audio_mask_expanded = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask_expanded, audio_features)

        # Process images
        image_mask = None
        deepstack_image_embeds = None
        if pixel_values is not None and image_grid_thw is not None:
            pixel_values = pixel_values.to(dtype=self.visual.patch_embed.proj.weight.dtype)
            image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = (input_ids == self.image_token_id)
            special_image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask.to(inputs_embeds.device), image_embeds
            )

        # Process videos
        video_mask = None
        deepstack_video_embeds = None
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

        # Compute audio_seqlens for rope_index
        if feature_attention_mask is not None:
            audio_seqlens_for_rope = torch.sum(feature_attention_mask, dim=1)
        elif audio_feature_lengths is not None:
            audio_seqlens_for_rope = audio_feature_lengths
        else:
            audio_seqlens_for_rope = None

        # Compute M-RoPE position ids
        if position_ids is None:
            position_ids, _ = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask,
                audio_seqlens=audio_seqlens_for_rope,
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
        # Audio inputs
        input_features: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        # Vision inputs
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        # Common
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Greedy autoregressive generation with KV caching.

        Returns:
            generated_ids: (batch, prompt_len + max_new_tokens)
        """
        self.reset_cache()
        batch_size, prompt_len = input_ids.shape

        # Compute audio_seqlens for rope_index
        if feature_attention_mask is not None:
            audio_seqlens_for_rope = torch.sum(feature_attention_mask, dim=1)
        elif audio_feature_lengths is not None:
            audio_seqlens_for_rope = audio_feature_lengths
        else:
            audio_seqlens_for_rope = None

        position_ids, mrope_position_deltas = self.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, attention_mask,
            audio_seqlens=audio_seqlens_for_rope,
        )

        # Prefill
        logits = self.forward(
            input_ids,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
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
