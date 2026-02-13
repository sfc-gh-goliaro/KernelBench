"""
RT-DETR v2 Object Detection Model

Implements RT-DETR v2 (Real-Time Detection Transformer v2) architecture:
- ResNet backbone with frozen batch norm
- Hybrid encoder (transformer encoder + FPN + PAN)
- Deformable attention decoder with iterative box refinement
- Multi-scale deformable attention (v2 variant)

Variant: PekingU/rtdetr_v2_r18vd (ResNet-18 backbone)

This model uses level1 operators from KernelBench:
- LayerNorm from level1/normalization/6_LayerNorm
- ReLU from level1/activations/1_ReLU  
- GELU from level1/activations/8_GELU
- SiLU (Swish) from level1/activations/7_Swish
- MatMul from level1/matmul/1_MatMul

Architecture mirrors HuggingFace transformers RTDetrV2ForObjectDetection
for exact weight transfer and alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from typing import Optional, Dict, Any, Tuple, List

# Import level1 operators
from ..level1.normalization._6_LayerNorm import Model as LayerNorm
from ..level1.activations._1_ReLU import Model as ReLU
from ..level1.activations._8_GELU import Model as GELU
from ..level1.activations._7_Swish import Model as SiLU
from ..level1.matmul._1_MatMul import Model as MatMul

from .base import ModelConfig, OperatorLevel

# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "r18vd": "PekingU/rtdetr_v2_r18vd",
}


# ============================================================================
# Activation function mapping (using level1 operators)
# ============================================================================

_LEVEL1_ACT = {
    "relu": ReLU,
    "gelu": GELU,
    "silu": SiLU,
}


def _get_activation(name: str) -> nn.Module:
    """Get activation module by name, using level1 operators when available."""
    if name in _LEVEL1_ACT:
        return _LEVEL1_ACT[name]()
    raise ValueError(f"Unknown activation: {name}")


# ============================================================================
# Utility functions (matching HF exactly)
# ============================================================================

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def multi_scale_deformable_attention_v2(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: List[int],
    method="default",
) -> torch.Tensor:
    """V2 multi-scale deformable attention - matches HF implementation exactly."""
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points = sampling_locations.shape
    value_list = (
        value.permute(0, 2, 3, 1)
        .flatten(0, 1)
        .split([height * width for height, width in value_spatial_shapes], dim=-1)
    )
    if method == "default":
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_grids = sampling_grids.split(num_points_list, dim=-2)
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level_id].reshape(batch_size * num_heads, hidden_dim, height, width)
        sampling_grid_l_ = sampling_grids[level_id]
        if method == "default":
            sampling_value_l_ = nn.functional.grid_sample(
                value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        elif method == "discrete":
            sampling_coord = (sampling_grid_l_ * torch.tensor([[width, height]], device=value.device) + 0.5).to(
                torch.int64
            )
            sampling_coord_x = sampling_coord[..., 0].clamp(0, width - 1)
            sampling_coord_y = sampling_coord[..., 1].clamp(0, height - 1)
            sampling_coord = torch.stack([sampling_coord_x, sampling_coord_y], dim=-1)
            sampling_coord = sampling_coord.reshape(batch_size * num_heads, num_queries * num_points_list[level_id], 2)
            sampling_idx = (
                torch.arange(sampling_coord.shape[0], device=value.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l_ = value_l_[sampling_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]]
            sampling_value_l_ = sampling_value_l_.permute(0, 2, 1).reshape(
                batch_size * num_heads, hidden_dim, num_queries, num_points_list[level_id]
            )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.permute(0, 2, 1, 3).reshape(
        batch_size * num_heads, 1, num_queries, sum(num_points_list)
    )
    output = (
        (torch.concat(sampling_value_list, dim=-1) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


# ============================================================================
# Frozen Batch Norm (matches HF exactly)
# ============================================================================

class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d where the batch statistics and the affine parameters are fixed."""
    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        epsilon = 1e-5
        scale = weight * (running_var + epsilon).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


def replace_batch_norm(model):
    """Recursively replace all nn.BatchNorm2d with FrozenBatchNorm2d."""
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_module = FrozenBatchNorm2d(module.num_features)
            if module.weight.device != torch.device("meta"):
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)
            model._modules[name] = new_module
        if len(list(module.children())) > 0:
            replace_batch_norm(module)


# ============================================================================
# Backbone: Native ResNet (replaces HF RTDetrResNetBackbone)
# ============================================================================

class ResNetConvLayer(nn.Module):
    """Conv2d + FrozenBatchNorm2d + optional activation, matching HF RTDetrResNetConvLayer."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, activation: str = "relu"):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False
        )
        self.normalization = FrozenBatchNorm2d(out_channels)
        self.activation = _get_activation(activation) if activation is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class ResNetShortCut(nn.Module):
    """1x1 conv + FrozenBatchNorm2d shortcut, matching HF RTDetrResNetShortCut."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.normalization = FrozenBatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        x = self.normalization(x)
        return x


class ResNetBasicLayer(nn.Module):
    """Basic residual block matching HF RTDetrResNetBasicLayer.
    
    Uses level1 ReLU activation.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 should_apply_shortcut: bool = False, hidden_act: str = "relu"):
        super().__init__()
        if in_channels != out_channels:
            self.shortcut = (
                nn.Sequential(
                    nn.AvgPool2d(2, 2, 0, ceil_mode=True),
                    ResNetShortCut(in_channels, out_channels, stride=1)
                )
                if should_apply_shortcut
                else nn.Identity()
            )
        else:
            self.shortcut = (
                ResNetShortCut(in_channels, out_channels, stride=stride)
                if should_apply_shortcut
                else nn.Identity()
            )
        self.layer = nn.Sequential(
            ResNetConvLayer(in_channels, out_channels, stride=stride),
            ResNetConvLayer(out_channels, out_channels, activation=None),
        )
        self.activation = _get_activation(hidden_act)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class ResNetStage(nn.Module):
    """A ResNet stage composed of stacked basic layers, matching HF RTDetrResNetStage."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2,
                 depth: int = 2, hidden_act: str = "relu"):
        super().__init__()
        first_layer = ResNetBasicLayer(
            in_channels, out_channels, stride=stride,
            should_apply_shortcut=True, hidden_act=hidden_act
        )
        self.layers = nn.Sequential(
            first_layer,
            *[ResNetBasicLayer(out_channels, out_channels, hidden_act=hidden_act)
              for _ in range(depth - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ResNetEmbeddings(nn.Module):
    """ResNet stem (3 conv layers + maxpool), matching HF RTDetrResNetEmbeddings."""
    def __init__(self, num_channels: int = 3, embedding_size: int = 64,
                 hidden_act: str = "relu"):
        super().__init__()
        self.embedder = nn.Sequential(
            ResNetConvLayer(num_channels, embedding_size // 2, kernel_size=3,
                          stride=2, activation=hidden_act),
            ResNetConvLayer(embedding_size // 2, embedding_size // 2, kernel_size=3,
                          stride=1, activation=hidden_act),
            ResNetConvLayer(embedding_size // 2, embedding_size, kernel_size=3,
                          stride=1, activation=hidden_act),
        )
        self.pooler = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embedding = self.embedder(pixel_values)
        embedding = self.pooler(embedding)
        return embedding


class ResNetEncoder(nn.Module):
    """ResNet encoder (4 stages), matching HF RTDetrResNetEncoder."""
    def __init__(self, embedding_size: int = 64, hidden_sizes: list = None,
                 depths: list = None, downsample_in_first_stage: bool = False,
                 hidden_act: str = "relu"):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 128, 256, 512]
        if depths is None:
            depths = [2, 2, 2, 2]

        self.stages = nn.ModuleList()
        # First stage: no downsampling (stride=1)
        self.stages.append(
            ResNetStage(
                embedding_size, hidden_sizes[0],
                stride=2 if downsample_in_first_stage else 1,
                depth=depths[0], hidden_act=hidden_act
            )
        )
        # Subsequent stages: stride=2 downsampling
        for i in range(1, len(hidden_sizes)):
            self.stages.append(
                ResNetStage(
                    hidden_sizes[i - 1], hidden_sizes[i],
                    depth=depths[i], hidden_act=hidden_act
                )
            )

    def forward(self, hidden_state: torch.Tensor) -> List[torch.Tensor]:
        """Returns list of hidden states from each stage (including input)."""
        hidden_states = [hidden_state]  # stem output = stage index 0 (before stage 0)
        for stage in self.stages:
            hidden_state = stage(hidden_state)
            hidden_states.append(hidden_state)
        return hidden_states


class ResNetBackbone(nn.Module):
    """Native ResNet backbone matching HF RTDetrResNetBackbone.
    
    Outputs feature maps for specified out_indices.
    Uses level1 ReLU activation and FrozenBatchNorm2d.
    """
    def __init__(self, backbone_config):
        super().__init__()
        # Extract backbone config parameters
        if hasattr(backbone_config, 'to_dict'):
            bc = backbone_config
        else:
            # backbone_config is a dict-like object
            bc = type('Config', (), backbone_config)()

        num_channels = getattr(bc, 'num_channels', 3)
        embedding_size = getattr(bc, 'embedding_size', 64)
        hidden_sizes = getattr(bc, 'hidden_sizes', [64, 128, 256, 512])
        depths = getattr(bc, 'depths', [2, 2, 2, 2])
        hidden_act = getattr(bc, 'hidden_act', 'relu')
        downsample_in_first_stage = getattr(bc, 'downsample_in_first_stage', False)
        out_indices = getattr(bc, 'out_indices', [2, 3, 4])

        self.embedder = ResNetEmbeddings(num_channels, embedding_size, hidden_act)
        self.encoder = ResNetEncoder(
            embedding_size, hidden_sizes, depths,
            downsample_in_first_stage, hidden_act
        )
        self.out_indices = out_indices
        # stage_names: stem, stage1, stage2, stage3, stage4
        self.stage_names = ['stem'] + [f'stage{i+1}' for i in range(len(hidden_sizes))]
        # Channels for output feature maps
        num_features = [embedding_size] + hidden_sizes
        self.channels = [num_features[i] for i in out_indices]

    def forward(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        """Returns list of feature maps for the configured out_indices."""
        embedding = self.embedder(pixel_values)
        # hidden_states: [stem_output, stage1_out, stage2_out, stage3_out, stage4_out]
        hidden_states = self.encoder(embedding)
        # hidden_states has len = num_stages + 1 (stem output + each stage output)
        # Index 0 = stem output (before any stage)
        # Index 1 = stage1 output, etc.
        # out_indices=[2,3,4] means stage2, stage3, stage4
        feature_maps = [hidden_states[i] for i in self.out_indices]
        return feature_maps


class ConvEncoder(nn.Module):
    """Convolutional backbone wrapper, now using native ResNet implementation."""
    def __init__(self, config):
        super().__init__()
        self.model = ResNetBackbone(config.backbone_config)
        self.intermediate_channel_sizes = self.model.channels

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        feature_maps = self.model(pixel_values)
        out = []
        for feature_map in feature_maps:
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out


# ============================================================================
# Conv + Norm Layer
# ============================================================================

class ConvNormLayer(nn.Module):
    """Conv2d + BatchNorm + optional activation."""
    def __init__(self, config, in_channels, out_channels, kernel_size, stride, padding=None, activation=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels, config.batch_norm_eps)
        self.activation = nn.Identity() if activation is None else _get_activation(activation)

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = self.norm(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


# ============================================================================
# Multihead Attention (using level1 MatMul)
# ============================================================================

class MultiheadAttention(nn.Module):
    """Multi-headed attention with position embeddings, using level1 MatMul."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads")
        self.scaling = self.head_dim ** -0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.matmul = MatMul()

    def _reshape(self, tensor, seq_len, batch_size):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor, position_embeddings):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, output_attentions=False):
        batch_size, target_len, embed_dim = hidden_states.size()
        if position_embeddings is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._reshape(self.k_proj(hidden_states), -1, batch_size)
        value_states = self._reshape(self.v_proj(hidden_states_original), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._reshape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.size(1)
        # Use level1 MatMul for attention score computation
        attn_weights = self.matmul(query_states, key_states.transpose(1, 2))

        if attention_mask is not None:
            attention_mask = attention_mask.expand(batch_size, 1, *attention_mask.size())
            if attention_mask.dtype == torch.bool:
                attention_mask = torch.zeros_like(attention_mask, dtype=attn_weights.dtype).masked_fill_(
                    attention_mask, -torch.inf
                )
            attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, source_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, target_len, source_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # Use level1 MatMul for value aggregation
        attn_output = self.matmul(attn_probs, value_states)

        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


# ============================================================================
# V2 Multiscale Deformable Attention
# ============================================================================

class MultiscaleDeformableAttention(nn.Module):
    """RT-DETR V2 multiscale deformable attention."""
    def __init__(self, config):
        super().__init__()
        num_heads = config.decoder_attention_heads
        n_points = config.decoder_n_points
        if config.d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads")
        
        self.d_model = config.d_model
        self.n_levels = config.decoder_n_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(config.d_model, num_heads * self.n_levels * n_points * 2)
        self.attention_weights = nn.Linear(config.d_model, num_heads * self.n_levels * n_points)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        self.offset_scale = config.decoder_offset_scale
        self.method = config.decoder_method

        n_points_list = [self.n_points for _ in range(self.n_levels)]
        self.n_points_list = n_points_list
        n_points_scale = [1 / n for n in n_points_list for _ in range(n)]
        self.register_buffer("n_points_scale", torch.tensor(n_points_scale, dtype=torch.float32))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        output_attentions=False,
    ):
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value.masked_fill(~attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            n_points_scale = self.n_points_scale.to(dtype=hidden_states.dtype).unsqueeze(-1)
            offset = sampling_offsets * n_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        output = multi_scale_deformable_attention_v2(
            value, spatial_shapes_list, sampling_locations, attention_weights, self.n_points_list, self.method
        )
        output = self.output_proj(output)
        return output, attention_weights


# ============================================================================
# Decoder Layer (using level1 LayerNorm and ReLU)
# ============================================================================

class DecoderLayer(nn.Module):
    """RT-DETR V2 decoder layer using level1 operators."""
    def __init__(self, config):
        super().__init__()
        # Self-attention
        self.self_attn = MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = _get_activation(config.decoder_activation_function)
        self.activation_dropout = config.activation_dropout

        # Level1 LayerNorm
        self.self_attn_layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        # Cross attention (v2 deformable)
        self.encoder_attn = MultiscaleDeformableAttention(config)
        self.encoder_attn_layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        # FFN
        self.fc1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.d_model)
        self.final_layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        second_residual = hidden_states
        cross_attn_weights = None
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = second_residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # FFN
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs


# ============================================================================
# MLP Prediction Head
# ============================================================================

class MLPPredictionHead(nn.Module):
    """MLP prediction head for bounding boxes, using level1 ReLU."""
    def __init__(self, input_dim, d_model, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [d_model] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.relu = ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ============================================================================
# Encoder Layer (using level1 LayerNorm and GELU)
# ============================================================================

class EncoderLayer(nn.Module):
    """RT-DETR V2 encoder layer using level1 operators."""
    def __init__(self, config):
        super().__init__()
        self.normalize_before = config.normalize_before
        self.self_attn = MultiheadAttention(
            embed_dim=config.encoder_hidden_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.dropout,
        )
        self.self_attn_layer_norm = LayerNorm(config.encoder_hidden_dim, eps=config.layer_norm_eps)
        self.dropout = config.dropout
        self.activation_fn = _get_activation(config.encoder_activation_function)
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(config.encoder_hidden_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.encoder_hidden_dim)
        self.final_layer_norm = LayerNorm(config.encoder_hidden_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask, position_embeddings=None, output_attentions=False, **kwargs):
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
        residual = hidden_states

        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


# ============================================================================
# RepVGG Block (using level1 SiLU)
# ============================================================================

class RepVggBlock(nn.Module):
    """RepVGG block with level1 SiLU activation."""
    def __init__(self, config):
        super().__init__()
        activation = config.activation_function
        hidden_channels = int(config.encoder_hidden_dim * config.hidden_expansion)
        self.conv1 = ConvNormLayer(config, hidden_channels, hidden_channels, 3, 1, padding=1)
        self.conv2 = ConvNormLayer(config, hidden_channels, hidden_channels, 1, 1, padding=0)
        self.activation = nn.Identity() if activation is None else _get_activation(activation)

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        return self.activation(y)


# ============================================================================
# CSP Rep Layer
# ============================================================================

class CSPRepLayer(nn.Module):
    """Cross Stage Partial network layer with RepVGG blocks."""
    def __init__(self, config):
        super().__init__()
        in_channels = config.encoder_hidden_dim * 2
        out_channels = config.encoder_hidden_dim
        num_blocks = 3
        activation = config.activation_function
        hidden_channels = int(out_channels * config.hidden_expansion)
        self.conv1 = ConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.conv2 = ConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.bottlenecks = nn.Sequential(*[RepVggBlock(config) for _ in range(num_blocks)])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(config, hidden_channels, out_channels, 1, 1, activation=activation)
        else:
            self.conv3 = nn.Identity()

    def forward(self, hidden_state):
        hidden_state_1 = self.conv1(hidden_state)
        hidden_state_1 = self.bottlenecks(hidden_state_1)
        hidden_state_2 = self.conv2(hidden_state)
        return self.conv3(hidden_state_1 + hidden_state_2)


# ============================================================================
# Encoder (stack of encoder layers)
# ============================================================================

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])

    def forward(self, src, src_mask=None, pos_embed=None, output_attentions=False):
        hidden_states = src
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=src_mask,
                position_embeddings=pos_embed,
                output_attentions=output_attentions,
            )
        return hidden_states


# ============================================================================
# Hybrid Encoder (Transformer + FPN + PAN)
# ============================================================================

class HybridEncoder(nn.Module):
    """Hybrid encoder: transformer encoder + FPN + PAN."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_channels = config.encoder_in_channels
        self.feat_strides = config.feat_strides
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.encode_proj_layers = config.encode_proj_layers
        self.positional_encoding_temperature = config.positional_encoding_temperature
        self.eval_size = config.eval_size
        self.out_channels = [self.encoder_hidden_dim for _ in self.in_channels]
        self.out_strides = self.feat_strides
        self.num_fpn_stages = len(self.in_channels) - 1
        self.num_pan_stages = len(self.in_channels) - 1
        activation = config.activation_function

        # Encoder transformer
        self.encoder = nn.ModuleList([Encoder(config) for _ in range(len(self.encode_proj_layers))])

        # Top-down FPN
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(self.num_fpn_stages):
            lateral_conv = ConvNormLayer(
                config, self.encoder_hidden_dim, self.encoder_hidden_dim, 1, 1, activation=activation
            )
            fpn_block = CSPRepLayer(config)
            self.lateral_convs.append(lateral_conv)
            self.fpn_blocks.append(fpn_block)

        # Bottom-up PAN
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(self.num_pan_stages):
            downsample_conv = ConvNormLayer(
                config, self.encoder_hidden_dim, self.encoder_hidden_dim, 3, 2, activation=activation
            )
            pan_block = CSPRepLayer(config)
            self.downsample_convs.append(downsample_conv)
            self.pan_blocks.append(pan_block)

    @staticmethod
    def build_2d_sincos_position_embedding(width, height, embed_dim=256, temperature=10000.0, device="cpu", dtype=torch.float32):
        grid_w = torch.arange(int(width), device=device).to(dtype)
        grid_h = torch.arange(int(height), device=device).to(dtype)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        if embed_dim % 4 != 0:
            raise ValueError("Embed dimension must be divisible by 4")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, device=device).to(dtype) / pos_dim
        omega = 1.0 / (temperature ** omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, inputs_embeds=None, **kwargs):
        output_attentions = kwargs.get('output_attentions', False)
        output_hidden_states = kwargs.get('output_hidden_states', False)
        return_dict = kwargs.get('return_dict', True)

        hidden_states = inputs_embeds
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if self.config.encoder_layers > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states[enc_ind],)
                height, width = hidden_states[enc_ind].shape[2:]
                src_flatten = hidden_states[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        width, height, self.encoder_hidden_dim,
                        self.positional_encoding_temperature,
                        device=src_flatten.device, dtype=src_flatten.dtype,
                    )
                else:
                    pos_embed = None
                layer_outputs = self.encoder[i](
                    src_flatten, pos_embed=pos_embed, output_attentions=output_attentions,
                )
                hidden_states[enc_ind] = (
                    layer_outputs[0].permute(0, 2, 1).reshape(-1, self.encoder_hidden_dim, height, width).contiguous()
                )
                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states[enc_ind],)

        # Top-down FPN
        fpn_feature_maps = [hidden_states[-1]]
        for idx, (lateral_conv, fpn_block) in enumerate(zip(self.lateral_convs, self.fpn_blocks)):
            backbone_feature_map = hidden_states[self.num_fpn_stages - idx - 1]
            top_fpn_feature_map = fpn_feature_maps[-1]
            top_fpn_feature_map = lateral_conv(top_fpn_feature_map)
            fpn_feature_maps[-1] = top_fpn_feature_map
            top_fpn_feature_map = F.interpolate(top_fpn_feature_map, scale_factor=2.0, mode="nearest")
            fused_feature_map = torch.concat([top_fpn_feature_map, backbone_feature_map], dim=1)
            new_fpn_feature_map = fpn_block(fused_feature_map)
            fpn_feature_maps.append(new_fpn_feature_map)
        fpn_feature_maps = fpn_feature_maps[::-1]

        # Bottom-up PAN
        pan_feature_maps = [fpn_feature_maps[0]]
        for idx, (downsample_conv, pan_block) in enumerate(zip(self.downsample_convs, self.pan_blocks)):
            top_pan_feature_map = pan_feature_maps[-1]
            fpn_feature_map = fpn_feature_maps[idx + 1]
            downsampled_feature_map = downsample_conv(top_pan_feature_map)
            fused_feature_map = torch.concat([downsampled_feature_map, fpn_feature_map], dim=1)
            new_pan_feature_map = pan_block(fused_feature_map)
            pan_feature_maps.append(new_pan_feature_map)

        return pan_feature_maps, encoder_states, all_attentions


# ============================================================================
# Decoder
# ============================================================================

class Decoder(nn.Module):
    """RT-DETR V2 Decoder with iterative box refinement."""
    def __init__(self, config):
        super().__init__()
        self.dropout = config.dropout
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.query_pos_head = MLPPredictionHead(4, 2 * config.d_model, config.d_model, num_layers=2)
        self.bbox_embed = None
        self.class_embed = None

    def forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        intermediate = ()
        intermediate_reference_points = ()
        intermediate_logits = ()

        reference_points = F.sigmoid(reference_points)

        for idx, decoder_layer in enumerate(self.layers):
            reference_points_input = reference_points.unsqueeze(2)
            position_embeddings = self.query_pos_head(reference_points)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if self.bbox_embed is not None:
                predicted_corners = self.bbox_embed[idx](hidden_states)
                new_reference_points = F.sigmoid(predicted_corners + inverse_sigmoid(reference_points))
                reference_points = new_reference_points.detach()

            intermediate += (hidden_states,)
            intermediate_reference_points += (
                (new_reference_points,) if self.bbox_embed is not None else (reference_points,)
            )

            if self.class_embed is not None:
                logits = self.class_embed[idx](hidden_states)
                intermediate_logits += (logits,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)
        if self.class_embed is not None:
            intermediate_logits = torch.stack(intermediate_logits, dim=1)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return (
            hidden_states,
            intermediate,
            intermediate_logits,
            intermediate_reference_points,
            all_hidden_states,
            all_self_attns,
            all_cross_attentions,
        )


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    RT-DETR v2 object detection model (RTDetrV2ForObjectDetection equivalent).
    
    Uses level1 operators from KernelBench:
    - LayerNorm from level1/normalization/6_LayerNorm
    - ReLU from level1/activations/1_ReLU
    - GELU from level1/activations/8_GELU
    - SiLU from level1/activations/7_Swish
    - MatMul from level1/matmul/1_MatMul
    
    Supports variants: r18vd (configs loaded from HuggingFace)
    
    Architecture mirrors HuggingFace RTDetrV2ForObjectDetection for weight alignment.
    The forward pass takes pixel_values and returns (logits, pred_boxes).
    """
    
    VARIANTS = VARIANTS
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # Build a config-like namespace from kwargs
        config = _build_config(kwargs)
        self.config = config
        
        # ---- RTDetrV2Model (self.model.*) ----
        # Backbone
        self.backbone = ConvEncoder(config)
        intermediate_channel_sizes = self.backbone.intermediate_channel_sizes

        # Encoder input projection
        encoder_input_proj_list = []
        for i in range(len(intermediate_channel_sizes)):
            in_channels = intermediate_channel_sizes[i]
            encoder_input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, config.encoder_hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(config.encoder_hidden_dim),
                )
            )
        self.encoder_input_proj = nn.ModuleList(encoder_input_proj_list)

        # Hybrid encoder
        self.encoder = HybridEncoder(config)

        # Denoising
        if config.num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                config.num_labels + 1, config.d_model, padding_idx=config.num_labels
            )

        # Decoder embedding
        if config.learn_initial_query:
            self.weight_embedding = nn.Embedding(config.num_queries, config.d_model)

        # Encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
        )
        self.enc_score_head = nn.Linear(config.d_model, config.num_labels)
        self.enc_bbox_head = MLPPredictionHead(config.d_model, config.d_model, 4, num_layers=3)

        # Decoder input projection
        num_backbone_outs = len(config.decoder_in_channels)
        decoder_input_proj_list = []
        for i in range(num_backbone_outs):
            in_channels = config.decoder_in_channels[i]
            decoder_input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, config.d_model, kernel_size=1, bias=False),
                    nn.BatchNorm2d(config.d_model, config.batch_norm_eps),
                )
            )
        for _ in range(config.num_feature_levels - num_backbone_outs):
            decoder_input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(config.d_model, config.batch_norm_eps),
                )
            )
            in_channels = config.d_model
        self.decoder_input_proj = nn.ModuleList(decoder_input_proj_list)

        # Decoder
        self.decoder = Decoder(config)

        # ---- RTDetrV2ForObjectDetection detection heads ----
        self.class_embed = nn.ModuleList([nn.Linear(config.d_model, config.num_labels) for _ in range(config.decoder_layers)])
        self.bbox_embed = nn.ModuleList([MLPPredictionHead(config.d_model, config.d_model, 4, num_layers=3) for _ in range(config.decoder_layers)])

        # Wire up decoder references for iterative refinement
        self.decoder.class_embed = self.class_embed
        self.decoder.bbox_embed = self.bbox_embed

    def generate_anchors(self, spatial_shapes=None, grid_size=0.05, device="cpu", dtype=torch.float32):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.config.anchor_image_size[0] / s), int(self.config.anchor_image_size[1] / s)]
                for s in self.config.feat_strides
            ]
        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=height, device=device).to(dtype),
                torch.arange(end=width, device=device).to(dtype),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)
            grid_xy = grid_xy.unsqueeze(0) + 0.5
            grid_xy[..., 0] /= width
            grid_xy[..., 1] /= height
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** level)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, height * width, 4))
        eps = 1e-2
        anchors = torch.concat(anchors, 1)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.tensor(torch.finfo(dtype).max, dtype=dtype, device=device))
        return anchors, valid_mask

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Forward pass for RT-DETR v2.
        
        Args:
            pixel_values: Input images (batch, 3, H, W)
            pixel_mask: Mask for valid pixels (batch, H, W)
            
        Returns:
            Tuple of (logits, pred_boxes):
                logits: Classification logits (batch, num_queries, num_labels)
                pred_boxes: Predicted bounding boxes (batch, num_queries, 4)
        """
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device
        dtype = pixel_values.dtype

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), device=device)

        # Backbone
        features = self.backbone(pixel_values, pixel_mask)
        proj_feats = [self.encoder_input_proj[level](source) for level, (source, mask) in enumerate(features)]

        # Encoder
        encoder_outputs = self.encoder(proj_feats)
        pan_feature_maps = encoder_outputs[0]

        # Decoder input projection
        sources = []
        for level, source in enumerate(pan_feature_maps):
            sources.append(self.decoder_input_proj[level](source))

        if self.config.num_feature_levels > len(sources):
            _len_sources = len(sources)
            sources.append(self.decoder_input_proj[_len_sources](pan_feature_maps[-1]))
            for i in range(_len_sources + 1, self.config.num_feature_levels):
                sources.append(self.decoder_input_proj[i](pan_feature_maps[-1]))

        # Flatten
        source_flatten = []
        spatial_shapes_list = []
        spatial_shapes = torch.empty((len(sources), 2), device=device, dtype=torch.long)
        for level, source in enumerate(sources):
            h, w = source.shape[-2:]
            spatial_shapes[level, 0] = h
            spatial_shapes[level, 1] = w
            spatial_shapes_list.append((h, w))
            source = source.flatten(2).transpose(1, 2)
            source_flatten.append(source)
        source_flatten = torch.cat(source_flatten, 1)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # No denoising at inference
        denoising_class, denoising_bbox_unact, attention_mask = None, None, None

        # Generate anchors
        spatial_shapes_tuple = tuple(spatial_shapes_list)
        anchors, valid_mask = self.generate_anchors(spatial_shapes_tuple, device=device, dtype=dtype)

        memory = valid_mask.to(source_flatten.dtype) * source_flatten
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.config.num_queries, dim=1)

        reference_points_unact = enc_outputs_coord_logits.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_logits.shape[-1])
        )

        if self.config.learn_initial_query:
            target = self.weight_embedding.weight.unsqueeze(0).tile([batch_size, 1, 1])
        else:
            target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        init_reference_points = reference_points_unact.detach()

        # Decoder
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=attention_mask,
            reference_points=init_reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )

        # decoder_outputs: (hidden_states, intermediate, intermediate_logits, intermediate_reference_points, ...)
        outputs_class = decoder_outputs[2]  # intermediate_logits
        outputs_coord = decoder_outputs[3]  # intermediate_reference_points

        logits = outputs_class[:, -1]
        pred_boxes = outputs_coord[:, -1]

        return logits, pred_boxes


# ============================================================================
# Backbone Config (replaces HF RTDetrResNetConfig)
# ============================================================================

class _BackboneConfig:
    """Simple config object for the native ResNet backbone."""
    def __init__(self, **kwargs):
        defaults = {
            'num_channels': 3,
            'embedding_size': 64,
            'hidden_sizes': [64, 128, 256, 512],
            'depths': [2, 2, 2, 2],
            'layer_type': 'basic',
            'hidden_act': 'relu',
            'downsample_in_first_stage': False,
            'downsample_in_bottleneck': False,
            'out_features': None,
            'out_indices': [2, 3, 4],
        }
        for k, v in defaults.items():
            setattr(self, k, v)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return self.__dict__.copy()


# ============================================================================
# Config builder
# ============================================================================

def _build_config(kwargs):
    """Build a config namespace object from kwargs dict."""
    
    class Config:
        pass
    
    config = Config()
    
    # Set all kwargs as attributes
    for k, v in kwargs.items():
        setattr(config, k, v)
    
    # Defaults (matching PekingU/rtdetr_v2_r18vd)
    defaults = {
        'd_model': 256,
        'encoder_hidden_dim': 256,
        'encoder_in_channels': [128, 256, 512],
        'encoder_ffn_dim': 1024,
        'encoder_layers': 1,
        'encoder_attention_heads': 8,
        'encoder_activation_function': 'gelu',
        'feat_strides': [8, 16, 32],
        'encode_proj_layers': [2],
        'positional_encoding_temperature': 10000,
        'eval_size': None,
        'normalize_before': False,
        'hidden_expansion': 0.5,
        'activation_function': 'silu',
        'batch_norm_eps': 1e-5,
        'layer_norm_eps': 1e-5,
        'dropout': 0.0,
        'activation_dropout': 0.0,
        'attention_dropout': 0.0,
        'num_queries': 300,
        'num_labels': 80,
        'num_feature_levels': 3,
        'decoder_layers': 3,
        'decoder_attention_heads': 8,
        'decoder_ffn_dim': 1024,
        'decoder_in_channels': [256, 256, 256],
        'decoder_n_points': 4,
        'decoder_n_levels': 3,
        'decoder_offset_scale': 0.5,
        'decoder_method': 'default',
        'decoder_activation_function': 'relu',
        'num_denoising': 100,
        'learn_initial_query': False,
        'anchor_image_size': None,
        'with_box_refine': True,
        'freeze_backbone_batch_norms': True,
        'use_pretrained_backbone': False,
        'use_timm_backbone': False,
        'backbone': None,
        'backbone_kwargs': None,
        'backbone_config': None,
    }
    
    for k, v in defaults.items():
        if not hasattr(config, k):
            setattr(config, k, v)
    
    # Build backbone config if not provided
    if config.backbone_config is None and config.backbone is None:
        config.backbone_config = _BackboneConfig(
            num_channels=3,
            embedding_size=64,
            hidden_sizes=[64, 128, 256, 512],
            depths=[2, 2, 2, 2],
            layer_type="basic",
            hidden_act="relu",
            downsample_in_first_stage=False,
            downsample_in_bottleneck=False,
            out_features=None,
            out_indices=[2, 3, 4],
        )
    elif isinstance(config.backbone_config, dict):
        # Remove model_type if present (not needed for native backbone)
        config.backbone_config.pop("model_type", None)
        config.backbone_config = _BackboneConfig(**config.backbone_config)
    
    return config
