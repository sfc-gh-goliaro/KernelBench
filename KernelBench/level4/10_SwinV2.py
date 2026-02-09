"""
Swin Transformer V2 Vision Model

Implements Swin Transformer V2 architecture aligned with HuggingFace
Swinv2ForImageClassification:
- Shifted window attention with cosine similarity
- Hierarchical feature maps with patch merging
- Continuous log-spaced relative position bias via MLP
- Post-layer-norm (layernorm_before = after attention, layernorm_after = after MLP)

HuggingFace weight structure (Swinv2ForImageClassification):
  swinv2.embeddings.patch_embeddings.projection.{weight,bias}
  swinv2.embeddings.norm.{weight,bias}
  swinv2.encoder.layers.{i}.blocks.{j}.attention.self.logit_scale
  swinv2.encoder.layers.{i}.blocks.{j}.attention.self.continuous_position_bias_mlp.0.{weight,bias}
  swinv2.encoder.layers.{i}.blocks.{j}.attention.self.continuous_position_bias_mlp.2.weight
  swinv2.encoder.layers.{i}.blocks.{j}.attention.self.query.{weight,bias}   <-- HF path
  swinv2.encoder.layers.{i}.blocks.{j}.attention.self.key.weight  (no bias!)  <-- HF path
  swinv2.encoder.layers.{i}.blocks.{j}.attention.self.value.{weight,bias}    <-- HF path
  NOTE: In KB, Q/K/V are at attention.query/key/value (not attention.self.query/key/value).
        The test harness key mapping handles this difference.
  swinv2.encoder.layers.{i}.blocks.{j}.attention.output.dense.{weight,bias}
  swinv2.encoder.layers.{i}.blocks.{j}.layernorm_before.{weight,bias}
  swinv2.encoder.layers.{i}.blocks.{j}.intermediate.dense.{weight,bias}
  swinv2.encoder.layers.{i}.blocks.{j}.output.dense.{weight,bias}
  swinv2.encoder.layers.{i}.blocks.{j}.layernorm_after.{weight,bias}
  swinv2.encoder.layers.{i}.downsample.reduction.weight
  swinv2.encoder.layers.{i}.downsample.norm.{weight,bias}
  swinv2.layernorm.{weight,bias}
  classifier.{weight,bias}

Tested against: microsoft/swinv2-large-patch4-window12-192-22k

This model uses level1 operators from KernelBench:
- WindowPartition2D from level1/vision/9_WindowPartition2D (pad/shift/partition + reverse)
- ShiftedWindowAttention from level1/attention/8_ShiftedWindowAttention (SwinV2 window attention)
- PatchEmbed2D from level1/vision/1_PatchEmbed2D (patch embedding via Conv2d)
- PatchMerging from level1/vision/3_PatchMerging (spatial downsampling between stages)
- LayerNorm from level1/normalization/6_LayerNorm
- Linear from level1/matmul/10_Linear (all projections)
- GELU from level1/activations/8_GELU (MLP intermediate)
- ReLU from level1/activations/1_ReLU (cpb_mlp, also used inside ShiftedWindowAttention)
- Softmax from level1/activations/5_Softmax (attention weights)
- Sigmoid from level1/activations/3_Sigmoid (position bias scaling)
- MatMul from level1/matmul/1_MatMul (attention scores & context)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List

# Import level1 operators
from ..level1.normalization._6_LayerNorm import Model as LayerNormOp
from ..level1.matmul._10_Linear import Model as Linear
from ..level1.activations._8_GELU import Model as GELU
from ..level1.vision._1_PatchEmbed2D import Model as PatchEmbed2D
from ..level1.vision._3_PatchMerging import Model as PatchMerging
from ..level1.vision._9_WindowPartition2D import Model as WindowPartition2D
from ..level1.attention._8_ShiftedWindowAttention import Model as ShiftedWindowAttention


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "T": "microsoft/swinv2-tiny-patch4-window8-256",
    "S": "microsoft/swinv2-small-patch4-window8-256",
    "B": "microsoft/swinv2-base-patch4-window12-192-22k",
    "L": "microsoft/swinv2-large-patch4-window12-192-22k",
}


# ============================================================================
# Component Modules (matching HF weight layout)
# ============================================================================

class Swinv2SelfOutput(nn.Module):
    """Output projection for attention (matches HF Swinv2SelfOutput).
    Uses level1 Linear.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dense = Linear(dim, dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dense(hidden_states)


class Swinv2Attention(nn.Module):
    """Full attention module matching HF Swinv2Attention.

    HF structure: self.self (ShiftedWindowAttention) + self.output (Swinv2SelfOutput)

    Q/K/V projections are owned here (like Llama's LlamaAttention owns
    q_proj/k_proj/v_proj while GroupedQueryAttention is projection-free).
    Pre-projected multi-head tensors are passed to ShiftedWindowAttention.

    Uses level1 Linear for Q/K/V projections and ShiftedWindowAttention
    for the core cosine attention + relative position bias computation.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        qkv_bias: bool = True,
        pretrained_window_size: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Q/K/V projections (key has NO bias, matching HuggingFace)
        self.query = Linear(dim, dim, bias=qkv_bias)
        self.key = Linear(dim, dim, bias=False)
        self.value = Linear(dim, dim, bias=qkv_bias)

        # Level1 ShiftedWindowAttention: cosine attention + position bias
        self.self = ShiftedWindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            pretrained_window_size=pretrained_window_size,
        )
        self.output = Swinv2SelfOutput(dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V and reshape to multi-head format
        # (batch, seq, dim) -> (batch, num_heads, seq, head_dim)
        q = (
            self.query(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        self_outputs = self.self(q, k, v, attention_mask)
        attention_output = self.output(self_outputs)
        return attention_output


class Swinv2Intermediate(nn.Module):
    """MLP intermediate layer matching HF Swinv2Intermediate.
    Uses level1 Linear and GELU.
    """
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dense = Linear(dim, int(mlp_ratio * dim), bias=True)
        self.act = GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.act(self.dense(hidden_states))


class Swinv2Output(nn.Module):
    """MLP output layer matching HF Swinv2Output.
    Uses level1 Linear.
    """
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dense = Linear(int(mlp_ratio * dim), dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dense(hidden_states)


class Swinv2Layer(nn.Module):
    """Swin Transformer V2 block matching HF Swinv2Layer.

    Key: HF uses POST-norm (layernorm_before applies AFTER attention residual,
    layernorm_after applies AFTER MLP residual). This differs from standard
    pre-norm architectures.

    HF forward:
      shortcut = x
      x = window_partition(x)   # pad + shift + partition
      x = attention(x, mask)
      x = window_reverse(x)     # merge + unshift + unpad
      hidden_states = layernorm_before(x)   # post-attention norm
      hidden_states = shortcut + drop_path(hidden_states)
      layer_output = intermediate(hidden_states)
      layer_output = output(layer_output)
      layer_output = hidden_states + drop_path(layernorm_after(layer_output))

    Uses level1 operators: WindowPartition2D, ShiftedWindowAttention, LayerNorm.
    """
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        pretrained_window_size: int = 0,
    ):
        super().__init__()
        self.input_resolution = input_resolution

        # Compute effective window/shift size
        effective_window_size = min(window_size, min(input_resolution))
        effective_shift_size = (
            0 if min(input_resolution) <= window_size else shift_size
        )

        # Level1 WindowPartition2D: pad + shift + partition / merge + unshift + unpad
        self.window_partition = WindowPartition2D(
            window_size=effective_window_size,
            shift_size=effective_shift_size,
        )

        self.attention = Swinv2Attention(
            dim=dim,
            num_heads=num_heads,
            window_size=effective_window_size,
            qkv_bias=qkv_bias,
            pretrained_window_size=pretrained_window_size,
        )
        # Level1 LayerNorm
        self.layernorm_before = LayerNormOp(dim)
        self.intermediate = Swinv2Intermediate(dim, mlp_ratio)
        self.output = Swinv2Output(dim, mlp_ratio)
        self.layernorm_after = LayerNormOp(dim)

    def forward(
        self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int]
    ) -> torch.Tensor:
        shortcut = hidden_states

        # Window partition: (B, H*W, C) -> (num_windows*B, window_size^2, C) + mask
        windows, attn_mask, ctx = self.window_partition(
            hidden_states, input_dimensions
        )

        # Window attention
        attention_output = self.attention(windows, attn_mask)

        # Window reverse: (num_windows*B, window_size^2, C) -> (B, H*W, C)
        hidden_states = self.window_partition.reverse(attention_output, ctx)

        # POST-NORM: layernorm_before applies AFTER attention (matches HF)
        hidden_states = shortcut + self.layernorm_before(hidden_states)

        # MLP with post-norm
        layer_output = self.intermediate(hidden_states)
        layer_output = self.output(layer_output)
        layer_output = hidden_states + self.layernorm_after(layer_output)

        return layer_output


class Swinv2Stage(nn.Module):
    """Stage containing multiple Swin blocks + optional downsampling.

    Matches HF Swinv2Stage structure:
      self.blocks = nn.ModuleList([Swinv2Layer, ...])
      self.downsample = PatchMerging (norm_before_reduction=False for SwinV2) or None
    """
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        pretrained_window_size: int = 0,
        downsample: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Swinv2Layer(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (j % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                pretrained_window_size=pretrained_window_size,
            )
            for j in range(depth)
        ])

        if downsample:
            # SwinV2: norm AFTER reduction (norm_before_reduction=False)
            self.downsample = PatchMerging(dim=dim, norm_before_reduction=False)
        else:
            self.downsample = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        height, width = input_dimensions
        for block in self.blocks:
            hidden_states = block(hidden_states, input_dimensions)

        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            height_ds, width_ds = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height_ds, width_ds)
            hidden_states = self.downsample(
                hidden_states_before_downsampling, input_dimensions
            )
        else:
            output_dimensions = (height, width)

        return hidden_states, output_dimensions


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Swin Transformer V2 for image classification matching HuggingFace
    Swinv2ForImageClassification.

    The weight structure is designed so state_dict keys match HF keys
    (after stripping the 'swinv2.' prefix in the test harness).

    HF structure:
      swinv2.embeddings.patch_embeddings.projection -> self.embeddings.patch_embeddings.projection
      swinv2.embeddings.norm -> self.embeddings.norm
      swinv2.encoder.layers.{i} -> self.encoder.layers.{i} (Swinv2Stage)
      swinv2.layernorm -> self.layernorm
      classifier -> self.classifier

    Uses level1 operators:
    - WindowPartition2D from level1/vision/9_WindowPartition2D (window partition/reverse)
    - ShiftedWindowAttention from level1/attention/8_ShiftedWindowAttention
    - PatchEmbed2D from level1/vision/1_PatchEmbed2D (patch embedding)
    - PatchMerging from level1/vision/3_PatchMerging (spatial downsampling)
    - LayerNorm from level1/normalization/6_LayerNorm
    - Linear from level1/matmul/10_Linear
    - GELU from level1/activations/8_GELU
    - ReLU from level1/activations/1_ReLU
    - Softmax from level1/activations/5_Softmax
    - Sigmoid from level1/activations/3_Sigmoid
    - MatMul from level1/matmul/1_MatMul
    """

    def __init__(
        self,
        image_size: int = 192,
        patch_size: int = 4,
        num_channels: int = 3,
        embed_dim: int = 192,
        depths: List[int] = [2, 2, 18, 2],
        num_heads: List[int] = [6, 12, 24, 48],
        window_size: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        num_labels: int = 21841,
        pretrained_window_sizes: List[int] = [0, 0, 0, 0],
        **kwargs,
    ):
        super().__init__()
        self.num_labels = num_labels
        num_stages = len(depths)
        grid_size = image_size // patch_size
        self.num_features = int(embed_dim * 2 ** (num_stages - 1))

        # --- Embeddings (matches HF Swinv2Embeddings) ---
        # Level1 PatchEmbed2D for patch embedding (Conv2d + flatten + transpose)
        self.embeddings = nn.Module()
        self.embeddings.patch_embeddings = PatchEmbed2D(
            img_size=image_size,
            patch_size=patch_size,
            in_channels=num_channels,
            embed_dim=embed_dim,
            flatten=True,
        )
        # Level1 LayerNorm for embedding norm
        self.embeddings.norm = LayerNormOp(embed_dim)

        # --- Encoder (matches HF Swinv2Encoder) ---
        self.encoder = nn.Module()
        layers = nn.ModuleList()
        for i in range(num_stages):
            dim = int(embed_dim * 2 ** i)
            resolution = grid_size // (2 ** i)
            stage = Swinv2Stage(
                dim=dim,
                input_resolution=(resolution, resolution),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                pretrained_window_size=pretrained_window_sizes[i]
                if i < len(pretrained_window_sizes)
                else 0,
                downsample=(i < num_stages - 1),
            )
            layers.append(stage)
        self.encoder.layers = layers

        # --- Final layer norm (level1 LayerNorm) ---
        self.layernorm = LayerNormOp(self.num_features)

        # --- Classification head (level1 Linear) ---
        self.classifier = Linear(self.num_features, num_labels, bias=True)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass matching HF Swinv2ForImageClassification.

        Args:
            pixel_values: (batch_size, num_channels, height, width)

        Returns:
            logits: (batch_size, num_labels)
        """
        # Patch embedding (level1 PatchEmbed2D: Conv2d + flatten + transpose)
        embeddings, input_dimensions = self.embeddings.patch_embeddings(pixel_values)
        embeddings = self.embeddings.norm(embeddings)
        hidden_states = embeddings

        # Encoder stages
        for stage in self.encoder.layers:
            hidden_states, input_dimensions = stage(
                hidden_states, input_dimensions
            )

        # Final norm
        hidden_states = self.layernorm(hidden_states)

        # Adaptive average pooling (like HF's pooler)
        pooled_output = hidden_states.mean(dim=1)

        # Classifier
        logits = self.classifier(pooled_output)
        return logits


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
image_size = 192
num_labels = 21841


def get_inputs():
    return [torch.randn(batch_size, 3, image_size, image_size)]


def get_init_inputs():
    return [{
        'image_size': image_size,
        'embed_dim': 192,
        'depths': [2, 2, 18, 2],
        'num_heads': [6, 12, 24, 48],
        'window_size': 12,
        'num_labels': num_labels,
        'pretrained_window_sizes': [0, 0, 0, 0],
    }]
