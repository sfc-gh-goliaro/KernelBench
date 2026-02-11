"""
Deformable DETR Object Detection Model

Implements Deformable DETR architecture:
- Multi-scale deformable attention
- Efficient transformer encoder-decoder
- Detection head with object queries

Variants from Table 5:
- Deformable-DETR-R50: ResNet-50 backbone

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple, List

# Import level1 operators (used directly - no wrapping needed)
from ..level1.normalization._6_LayerNorm import Model as LayerNorm
from ..level1.activations._1_ReLU import Model as ReLU
from ..level1.activations._8_GELU import Model as GELU
from ..level1.matmul._1_MatMul import Model as MatMul


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "R50": "SenseTime/deformable-detr",
}


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class PositionEmbeddingLearned(nn.Module):
    """Learnable 2D positional embedding."""
    def __init__(self, num_pos_feats: int = 128):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class MultiScaleDeformableAttention(nn.Module):
    """Multi-scale deformable attention using level1 operators."""
    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        self.matmul = MatMul()
        
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        nn.init.constant_(self.sampling_offsets.bias.data, 0.)
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        B, Len_q, _ = query.shape
        B, Len_v, _ = value.shape
        
        value = self.value_proj(value)
        value = value.view(B, Len_v, self.n_heads, self.d_model // self.n_heads)
        
        sampling_offsets = self.sampling_offsets(query).view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            B, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points
        )
        
        # Simplified sampling (using reference points + offsets)
        offset_normalizer = spatial_shapes.flip(-1).view(1, 1, 1, self.n_levels, 1, 2).float()
        sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer
        
        # Aggregate values (simplified bilinear sampling)
        output = torch.zeros(B, Len_q, self.n_heads, self.d_model // self.n_heads, device=query.device)
        
        for lvl in range(self.n_levels):
            start = level_start_index[lvl]
            end = level_start_index[lvl + 1] if lvl < self.n_levels - 1 else Len_v
            value_lvl = value[:, start:end]  # (B, H*W, heads, dim)
            
            # Simple average pooling as approximation
            output += attention_weights[:, :, :, lvl].sum(-1, keepdim=True) * value_lvl.mean(1, keepdim=True)
        
        output = output.view(B, Len_q, self.d_model)
        return self.output_proj(output)


class FFN(nn.Module):
    """Feed-forward network using level1 operators."""
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.relu = ReLU()
        self.linear2 = nn.Linear(d_ffn, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x)))


class DeformableTransformerEncoderLayer(nn.Module):
    """Deformable transformer encoder layer using level1 operators."""
    def __init__(self, d_model: int = 256, d_ffn: int = 1024, n_heads: int = 8, n_levels: int = 4):
        super().__init__()
        self.self_attn = MultiScaleDeformableAttention(d_model, n_levels, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ffn)
        self.norm2 = LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        src2 = self.self_attn(src, reference_points, src, spatial_shapes, level_start_index)
        src = src + src2
        src = self.norm1(src)
        src = src + self.ffn(src)
        src = self.norm2(src)
        return src


class DeformableTransformerDecoderLayer(nn.Module):
    """Deformable transformer decoder layer using level1 operators."""
    def __init__(self, d_model: int = 256, d_ffn: int = 1024, n_heads: int = 8, n_levels: int = 4):
        super().__init__()
        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = LayerNorm(d_model)
        
        # Cross attention (deformable)
        self.cross_attn = MultiScaleDeformableAttention(d_model, n_levels, n_heads)
        self.norm2 = LayerNorm(d_model)
        
        # FFN
        self.ffn = FFN(d_model, d_ffn)
        self.norm3 = LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        # Self attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)
        
        # Cross attention
        tgt2 = self.cross_attn(tgt, reference_points, memory, spatial_shapes, level_start_index)
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        
        # FFN
        tgt = tgt + self.ffn(tgt)
        tgt = self.norm3(tgt)
        
        return tgt


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Deformable DETR object detection model.
    
    Uses level1 operators from KernelBench:
    - LayerNorm from level1/normalization/6_LayerNorm
    - ReLU from level1/activations/1_ReLU
    - GELU from level1/activations/8_GELU
    - MatMul from level1/matmul/1_MatMul
    
    Supports variants: R50 (configs loaded from HuggingFace)
    """
    
    VARIANTS = VARIANTS
    
    @classmethod
    def from_pretrained(cls, variant: str = "R50", operator_level: Optional[OperatorLevel] = None, **kwargs):
        """Create model with config loaded from HuggingFace."""
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Available: {list(VARIANTS.keys())}")
        hf_config = load_hf_config(VARIANTS[variant])
        hf_config.update(kwargs)
        return cls(operator_level=operator_level, **hf_config)
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        operator_level: Optional[OperatorLevel] = None,
        **kwargs
    ):
        hidden_dim = kwargs.get('hidden_dim', kwargs.get('hidden_size', 256))
        nheads = kwargs.get('nheads', kwargs.get('num_heads', 8))
        num_encoder_layers = kwargs.get('num_encoder_layers', 6)
        num_decoder_layers = kwargs.get('num_decoder_layers', 6)
        num_feature_levels = kwargs.get('num_feature_levels', 4)
        num_queries = kwargs.get('num_queries', 300)
        num_classes = kwargs.get('num_classes', 91)
        
        if config is None:
            config = ModelConfig(
                hidden_size=hidden_dim,
                num_heads=nheads,
                vocab_size=num_classes,
            )
        
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        
        # Input projection for each level
        self.input_proj = nn.ModuleList([
            nn.Conv2d(256 * (2 ** i) if i < 3 else 256 * 4, hidden_dim, 1)
            for i in range(num_feature_levels)
        ])
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            DeformableTransformerEncoderLayer(hidden_dim, hidden_dim * 4, nheads, num_feature_levels)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DeformableTransformerDecoderLayer(hidden_dim, hidden_dim * 4, nheads, num_feature_levels)
            for _ in range(num_decoder_layers)
        ])
        
        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        
        # Detection heads
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self,
        features: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = features[0].shape[0]
        device = features[0].device
        
        # Project features
        srcs = []
        spatial_shapes = []
        for i, feat in enumerate(features[:self.num_feature_levels]):
            src = self.input_proj[i](feat)
            srcs.append(src)
            spatial_shapes.append(torch.tensor([feat.shape[2], feat.shape[3]], device=device))
        
        spatial_shapes = torch.stack(spatial_shapes)
        level_start_index = torch.cat([
            torch.tensor([0], device=device),
            torch.cumsum(spatial_shapes.prod(-1), 0)[:-1]
        ])
        
        # Flatten features
        memory = torch.cat([s.flatten(2).transpose(1, 2) for s in srcs], dim=1)
        
        # Create reference points
        reference_points = torch.zeros(batch_size, memory.shape[1], 2, device=device)
        
        # Encoder
        for layer in self.encoder_layers:
            memory = layer(memory, reference_points, spatial_shapes, level_start_index)
        
        # Decoder
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        tgt, query_pos = query_embed.split(self.hidden_dim, dim=-1)
        
        reference_points = torch.zeros(batch_size, self.num_queries, 2, device=device)
        
        for layer in self.decoder_layers:
            tgt = layer(tgt + query_pos, memory, reference_points, spatial_shapes, level_start_index)
        
        # Detection heads
        class_logits = self.class_head(tgt)
        bbox_pred = self.bbox_head(tgt).sigmoid()
        
        return class_logits, bbox_pred
