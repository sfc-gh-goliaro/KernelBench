"""
Qwen2-VL Vision-Language Model

Implements the Qwen2-VL architecture aligned with HuggingFace
Qwen2VLForConditionalGeneration:
- Vision encoder: 3D patch embedding (Conv3d) + vision rotary pos emb
  + Transformer blocks with per-image attention + PatchMerger
- Language model (Qwen2 decoder): embed_tokens + M-RoPE + GQA + SiLU-gated MLP
  + RMSNorm
- lm_head

HuggingFace weight structure (Qwen2VLForConditionalGeneration):
  model.visual.patch_embed.proj.weight               # Conv3d (embed_dim, 3, temporal_ps, ps, ps)
  model.visual.blocks.{i}.norm1.{weight,bias}         # LayerNorm
  model.visual.blocks.{i}.attn.qkv.{weight,bias}      # Linear(dim, dim*3)
  model.visual.blocks.{i}.attn.proj.{weight,bias}      # Linear(dim, dim)
  model.visual.blocks.{i}.norm2.{weight,bias}         # LayerNorm
  model.visual.blocks.{i}.mlp.fc1.{weight,bias}        # Linear
  model.visual.blocks.{i}.mlp.fc2.{weight,bias}        # Linear
  model.visual.merger.ln_q.{weight,bias}               # LayerNorm
  model.visual.merger.mlp.0.{weight,bias}             # Linear
  model.visual.merger.mlp.2.{weight,bias}             # Linear
  model.visual.rotary_pos_emb.inv_freq                # buffer
  model.language_model.embed_tokens.weight
  model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,bias}
  model.language_model.layers.{i}.input_layernorm.weight
  model.language_model.layers.{i}.post_attention_layernorm.weight
  model.language_model.layers.{i}.mlp.{gate,up,down}_proj.weight
  model.language_model.norm.weight
  model.language_model.rotary_emb.inv_freq            # buffer
  lm_head.weight

Tested against: Qwen/Qwen2-VL-7B-Instruct

This model uses level1 operators from KernelBench:
- Linear from level1/matmul/_10_Linear
- LayerNorm from level1/normalization/_6_LayerNorm
- RMSNorm from level1/normalization/_4_RMSNorm
- Embedding from level1/embeddings/_2_Embedding
- PatchEmbed3D from level1/vision/_2_PatchEmbed3D
- GELU from level1/activations/_8_GELU
- Swish (SiLU) from level1/activations/_7_Swish
- Sigmoid from level1/activations/_3_Sigmoid
- RotaryEmbedding from level1/embeddings/_1_RotaryEmbedding
- MultimodalRotaryEmbedding from level1/embeddings/_5_MultimodalRotaryEmbedding
- VisionRotaryEmbedding from level1/embeddings/_6_VisionRotaryEmbedding
- PagedKVCache + AttentionMetadata from level1/attention/_1_PagedKVCache
- ScaledDotProductAttention(mode="eager") from level1/attention/_2_Attention
- MultiHeadAttention(mode="eager") from level1/attention/_2_Attention
- MatMul from level1/matmul/_1_MatMul

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
from ..level1.activations._3_Sigmoid import Model as Sigmoid
from ..level1.embeddings._1_RotaryEmbedding import Model as RotaryEmbedding
from ..level1.embeddings._5_MultimodalRotaryEmbedding import Model as MultimodalRotaryEmbedding
from ..level1.embeddings._6_VisionRotaryEmbedding import Model as VisionRotaryEmbedding
from ..level1.attention._1_PagedKVCache import AttentionMetadata, create_attention_metadata
from ..level1.attention._2_Attention import ScaledDotProductAttention
from ..level1.attention._2_Attention import MultiHeadAttention


# ============================================================================
# Model Variants
# ============================================================================

VARIANTS: Dict[str, str] = {
    "2B": "Qwen/Qwen2-VL-2B-Instruct",
    "7B": "Qwen/Qwen2-VL-7B-Instruct",
}



# ============================================================================
# Vision Encoder Components
# ============================================================================

class VisionMLP(nn.Module):
    """Vision encoder MLP with configurable activation.
    Uses level1 Linear, Sigmoid, and GELU operators."""
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str = "quick_gelu"):
        super().__init__()
        self.fc1 = Linear(dim, hidden_dim, bias=True)
        self.fc2 = Linear(hidden_dim, dim, bias=True)
        self.hidden_act = hidden_act
        if hidden_act == "quick_gelu":
            self.sigmoid = Sigmoid()
        else:
            self.gelu = GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.hidden_act == "quick_gelu":
            x = x * self.sigmoid(1.702 * x)
        else:
            x = self.gelu(x)
        return self.fc2(x)


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
        # QKV projection: (seq_len, dim) -> (seq_len, 3, num_heads, head_dim)
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        # Apply vision rotary embedding using level1 RotaryEmbedding
        # q/k: (seq_len, num_heads, head_dim), cos/sin: (seq_len, head_dim)
        cos, sin = position_embeddings
        orig_dtype = query_states.dtype
        q_float, k_float = query_states.float(), key_states.float()
        cos_f = cos.unsqueeze(-2).float()  # (seq_len, 1, head_dim)
        sin_f = sin.unsqueeze(-2).float()
        query_states, key_states = self.rotary.apply_rotary(
            q_float, k_float, cos_f, sin_f
        )
        query_states = query_states.to(orig_dtype)
        key_states = key_states.to(orig_dtype)

        # Reshape to (1, num_heads, seq_len, head_dim) for attention
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Process each image/chunk separately (per-image attention)
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


class Qwen2VLVisionBlock(nn.Module):
    """Vision transformer block. Uses level1 LayerNorm operator."""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 hidden_act: str = "quick_gelu"):
        super().__init__()
        self.norm1 = LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = LayerNorm(embed_dim, eps=1e-6)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.attn = VisionAttention(embed_dim, num_heads)
        self.mlp = VisionMLP(embed_dim, mlp_hidden_dim, hidden_act)

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
    """Merge spatial patches: groups spatial_merge_size^2 patches into one.
    Uses level1 LayerNorm, Linear, and GELU operators."""
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = LayerNorm(context_dim, eps=1e-6)
        self.mlp_fc1 = Linear(self.hidden_size, self.hidden_size, bias=True)
        self.mlp_act = GELU()
        self.mlp_fc2 = Linear(self.hidden_size, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x).view(-1, self.hidden_size)
        x = self.mlp_fc1(x)
        x = self.mlp_act(x)
        x = self.mlp_fc2(x)
        return x


class VisionEncoder(nn.Module):
    """Qwen2-VL Vision Transformer encoder."""
    def __init__(
        self,
        depth: int = 32,
        embed_dim: int = 1280,
        hidden_size: int = 3584,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        mlp_ratio: float = 4.0,
        hidden_act: str = "quick_gelu",
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            bias=False,
        )
        head_dim = embed_dim // num_heads
        # Level1 VisionRotaryEmbedding for 2D spatial position embeddings
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([
            Qwen2VLVisionBlock(embed_dim, num_heads, mlp_ratio, hidden_act)
            for _ in range(depth)
        ])
        self.merger = PatchMerger(
            dim=hidden_size, context_dim=embed_dim,
            spatial_merge_size=spatial_merge_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Flattened pixel values of shape (total_patches, C*temp*H*W)
            grid_thw: (num_images, 3) -- temporal, height, width grid for each image
        Returns:
            Merged vision embeddings of shape (total_merged_patches, hidden_size)
        """
        hidden_states = self.patch_embed(hidden_states)
        # Compute vision rotary position embeddings using level1 operator
        position_embeddings = self.rotary_pos_emb(grid_thw, self.spatial_merge_size)

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        return self.merger(hidden_states)


# ============================================================================
# Language Model (Decoder) Components
# ============================================================================


class Qwen2VLAttention(nn.Module):
    """Multi-headed attention with GQA, M-RoPE, and paged KV cache.
    
    Uses level1 operators:
    - Linear for Q/K/V/O projections
    - RotaryEmbedding for the core rotate_half operation
    - MultiHeadAttention(mode="eager") for paged KV cache + GQA +
      attention math (matches HuggingFace eager attention numerically)

    Position embeddings (cos, sin) arrive pre-assembled from the level1
    MultimodalRotaryEmbedding operator at the model level."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        block_size: int = 16,
        num_blocks: int = 1024,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_heads = num_kv_heads

        self.q_proj = Linear(hidden_size, num_heads * self.head_dim, bias=True)
        self.k_proj = Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.v_proj = Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.o_proj = Linear(num_heads * self.head_dim, hidden_size, bias=False)
        # Level1 RotaryEmbedding for the core rotate_half operation
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=1, base=10000.0)
        # Level1 MultiHeadAttention with eager mode: paged KV cache + GQA + attention
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=self.head_dim,
            block_size=block_size,
            num_blocks=num_blocks,
            mode="eager",
        )

    def reset_cache(self):
        self.mha.reset_cache()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply pre-assembled M-RoPE cos/sin using level1 RotaryEmbedding rotate_half
        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)  # (batch, 1, seq, head_dim)
        sin = sin.unsqueeze(1)
        query_states, key_states = self.rotary.apply_rotary(
            query_states, key_states, cos, sin
        )

        # Level1 MultiHeadAttention handles: KV cache write/gather + GQA repeat + attention math
        attn_output = self.mha(query_states, key_states, value_states, attn_metadata)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class Qwen2MLP(nn.Module):
    """SiLU-gated MLP (Qwen2 style)."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.swish(self.gate_proj(x)) * self.up_proj(x))


class Qwen2VLDecoderLayer(nn.Module):
    """Decoder layer with RMSNorm, attention (paged KV cache), MLP."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
        block_size: int = 16,
        num_blocks: int = 1024,
    ):
        super().__init__()
        self.self_attn = Qwen2VLAttention(
            hidden_size, num_heads, num_kv_heads,
            block_size=block_size, num_blocks=num_blocks,
        )
        self.mlp = Qwen2MLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, learnable_weight=True, dim=-1)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, learnable_weight=True, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attn_metadata=attn_metadata,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def reset_cache(self):
        """Reset this layer's KV cache."""
        self.self_attn.reset_cache()


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Qwen2-VL vision-language model aligned with HuggingFace
    Qwen2VLForConditionalGeneration.

    Uses level1 operators from KernelBench:
    - Linear from level1/matmul/_10_Linear
    - LayerNorm from level1/normalization/_6_LayerNorm
    - RMSNorm from level1/normalization/_4_RMSNorm
    - Embedding from level1/embeddings/_2_Embedding
    - PatchEmbed3D from level1/vision/_2_PatchEmbed3D
    - GELU from level1/activations/_8_GELU
    - Swish/SiLU from level1/activations/_7_Swish
    - Sigmoid from level1/activations/_3_Sigmoid
    - RotaryEmbedding from level1/embeddings/_1_RotaryEmbedding
    - MultimodalRotaryEmbedding from level1/embeddings/_5_MultimodalRotaryEmbedding
    - VisionRotaryEmbedding from level1/embeddings/_6_VisionRotaryEmbedding
    - PagedKVCache from level1/attention/_1_PagedKVCache
    - ScaledDotProductAttention(mode="eager") from level1/attention/_2_Attention
    - MultiHeadAttention(mode="eager") from level1/attention/_2_Attention
    - MatMul from level1/matmul/_1_MatMul

    Supports variants: 2B, 7B
    """

    VARIANTS = VARIANTS

    def __init__(
        self,
        # Vision config
        vision_depth: int = 32,
        vision_embed_dim: int = 1280,
        vision_num_heads: int = 16,
        vision_hidden_size: int = 3584,
        vision_hidden_act: str = "quick_gelu",
        vision_mlp_ratio: float = 4.0,
        in_channels: int = 3,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        # LLM config
        hidden_size: int = 3584,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,
        intermediate_size: int = 18944,
        vocab_size: int = 152064,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        mrope_section: Optional[List[int]] = None,
        # Paged KV cache config
        block_size: int = 16,
        num_blocks: int = 1024,
        # Special token ids
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        **kwargs,
    ):
        super().__init__()

        if mrope_section is None:
            mrope_section = [16, 24, 24]

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.spatial_merge_size = spatial_merge_size
        self.mrope_section = mrope_section
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Vision encoder
        self.visual = VisionEncoder(
            depth=vision_depth,
            embed_dim=vision_embed_dim,
            hidden_size=vision_hidden_size,
            num_heads=vision_num_heads,
            in_channels=in_channels,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            spatial_merge_size=spatial_merge_size,
            mlp_ratio=vision_mlp_ratio,
            hidden_act=vision_hidden_act,
        )

        # Language model
        self.embed_tokens = Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            Qwen2VLDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
                block_size=block_size,
                num_blocks=num_blocks,
            )
            for _ in range(num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps, learnable_weight=True, dim=-1)

        head_dim = hidden_size // num_attention_heads
        # Level1 MultimodalRotaryEmbedding for LLM M-RoPE (chunked mode for Qwen2-VL)
        self.rotary_emb = MultimodalRotaryEmbedding(
            head_dim, base=rope_theta, mrope_section=mrope_section, mode="chunked",
        )

        # LM head
        self.lm_head = Linear(hidden_size, vocab_size, bias=False)

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate 3D rope index based on image/video temporal, height, width in LLM.
        Matches HuggingFace Qwen2VLModel.get_rope_index exactly.
        """
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
            layer.reset_cache()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attn_metadata: AttentionMetadata,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass matching HuggingFace Qwen2VLForConditionalGeneration.

        Args:
            input_ids: (batch, seq_len) token ids
            attn_metadata: Attention metadata for paged KV cache
            pixel_values: flattened image pixel values for vision encoder
            image_grid_thw: (num_images, 3) -- temporal, height, width
            video_grid_thw: (num_videos, 3)
            pixel_values_videos: flattened video pixel values
            attention_mask: (batch, seq_len) padding mask
            position_ids: (3, batch, seq_len) pre-computed M-RoPE positions

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Text embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # Process images through vision encoder and inject into text embeddings
        if pixel_values is not None and image_grid_thw is not None:
            pixel_values = pixel_values.to(dtype=self.visual.patch_embed.proj.weight.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            # Create mask and scatter vision embeddings
            special_image_mask = (input_ids == self.image_token_id)
            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask.to(inputs_embeds.device), image_embeds
            )

        # Process videos similarly
        if pixel_values_videos is not None and video_grid_thw is not None:
            pixel_values_videos = pixel_values_videos.to(
                dtype=self.visual.patch_embed.proj.weight.dtype
            )
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            special_video_mask = (input_ids == self.video_token_id)
            special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                special_video_mask.to(inputs_embeds.device), video_embeds
            )

        # Compute M-RoPE position ids
        if position_ids is None:
            position_ids, _ = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask,
            )

        # Compute M-RoPE cos/sin using level1 MultimodalRotaryEmbedding
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # Decoder layers
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attn_metadata=attn_metadata,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def _prefill(
        self,
        input_ids: torch.LongTensor,
        block_table: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Prefill (prompt processing) - process all prompt tokens at once.

        Args:
            input_ids: (batch_size, seq_len) prompt token IDs
            block_table: (batch_size, max_blocks_per_seq) block assignments
            pixel_values, image_grid_thw, etc.: multimodal inputs
            attention_mask: (batch, seq_len) padding mask
            position_ids: (3, batch, seq_len) pre-computed M-RoPE positions

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        context_lens = torch.zeros(batch_size, dtype=torch.long, device=device)
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)

        attn_metadata = create_attention_metadata(
            batch_size=batch_size,
            seq_lens=seq_lens,
            context_lens=context_lens,
            block_table=block_table,
            block_size=self.block_size,
            device=device,
        )

        return self.forward(
            input_ids,
            attn_metadata=attn_metadata,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            pixel_values_videos=pixel_values_videos,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def _decode(
        self,
        input_ids: torch.LongTensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Decode (token generation) - process one token at a time.

        Args:
            input_ids: (batch_size, 1) new token IDs
            block_table: (batch_size, max_blocks_per_seq) block assignments
            context_lens: (batch_size,) tokens already processed
            position_ids: (3, batch, 1) M-RoPE positions for the new token

        Returns:
            logits: (batch_size, 1, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        assert seq_len == 1, "Decode should process one token at a time"

        seq_lens = context_lens + 1

        attn_metadata = create_attention_metadata(
            batch_size=batch_size,
            seq_lens=seq_lens,
            context_lens=context_lens,
            block_table=block_table,
            block_size=self.block_size,
            device=device,
        )

        return self.forward(
            input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
        )

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
        block_table: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Greedy autoregressive generation with paged KV caching.

        Args:
            input_ids: (batch, prompt_len) prompt token IDs
            max_new_tokens: number of tokens to generate
            pixel_values, image_grid_thw, etc.: multimodal inputs (used in prefill only)
            attention_mask: (batch, prompt_len) padding mask
            block_table: optional pre-allocated block table

        Returns:
            generated_ids: (batch, prompt_len + max_new_tokens)
        """
        self.reset_cache()
        batch_size, prompt_len = input_ids.shape
        device = input_ids.device

        # Allocate block table if not provided
        if block_table is None:
            max_seq_len = prompt_len + max(max_new_tokens, 1)
            max_blocks = (max_seq_len + self.block_size - 1) // self.block_size
            block_table = torch.arange(
                max_blocks, device=device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1).contiguous()
            for i in range(batch_size):
                block_table[i] = torch.arange(
                    i * max_blocks,
                    (i + 1) * max_blocks,
                    device=device,
                ) % self.num_blocks

        # Compute position_ids and mrope_position_deltas for the full prompt
        position_ids, mrope_position_deltas = self.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, attention_mask,
        )

        # Prefill: process all prompt tokens, cache K/V
        logits = self._prefill(
            input_ids,
            block_table=block_table,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            pixel_values_videos=pixel_values_videos,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [input_ids, next_token]
        context_lens = torch.full((batch_size,), prompt_len, dtype=torch.long, device=device)

        # Decode loop
        for step in range(max_new_tokens - 1):
            # Position for the new token: last position + 1
            # For M-RoPE, all 3 dimensions increment by 1 for text tokens
            seq_pos = prompt_len + step + mrope_position_deltas.squeeze(1)
            new_position_ids = seq_pos.unsqueeze(0).unsqueeze(-1).expand(3, batch_size, 1).long()

            logits = self._decode(
                next_token,
                block_table=block_table,
                context_lens=context_lens,
                position_ids=new_position_ids,
            )
            context_lens += 1
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)

        return torch.cat(generated, dim=1)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 1
sequence_length = 256
image_size = 448
vocab_size = 152064


def get_inputs():
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return [input_ids]


def get_init_inputs():
    return [{
        'vision_depth': 4,
        'vision_embed_dim': 1280,
        'vision_num_heads': 16,
        'vision_hidden_size': 3584,
        'hidden_size': 3584,
        'num_hidden_layers': 4,
        'num_attention_heads': 28,
        'num_key_value_heads': 4,
        'intermediate_size': 18944,
        'vocab_size': vocab_size,
    }]
