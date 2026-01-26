"""
Deformable Convolutions v2 (DCNv2) Model

Implements DCNv2 for recommendation systems:
- Learned sampling offsets
- Modulation mechanism
- Cross network with deformable convolutions

Variants from Table 5:
- DCNv2-Criteo: For CTR prediction

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, List

# Import level1 operators (used directly - no wrapping needed)
from ..level1.normalization._1_BatchNorm import Model as BatchNorm
from ..level1.normalization._6_LayerNorm import Model as LayerNorm
from ..level1.activations._1_ReLU import Model as ReLU
from ..level1.activations._3_Sigmoid import Model as Sigmoid
from ..level1.matmul._1_MatMul import Model as MatMul


# ============================================================================
# Model Variants - configs loaded from config_loader
# ============================================================================

VARIANTS: Dict[str, str] = {
    "Criteo": "dcnv2-base",  # Use hardcoded config
}


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class CrossNetworkV2(nn.Module):
    """Cross Network V2 for explicit feature interactions using level1 operators."""
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)
        ])
        
        self.matmul = MatMul()

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for i in range(self.num_layers):
            # x_{l+1} = x0 * (W_l @ x_l + b_l) + x_l
            xl = self.cross_layers[i](x)
            x = x0 * (xl + self.biases[i]) + x
        return x


class DeepNetwork(nn.Module):
    """Deep network (MLP) for implicit feature interactions using level1 operators."""
    def __init__(self, input_dim: int, layer_sizes: List[int]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for size in layer_sizes:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(ReLU())
            prev_dim = size
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class EmbeddingLayer(nn.Module):
    """Sparse feature embedding layer."""
    def __init__(self, num_sparse_features: int, embedding_dim: int, vocab_sizes: List[int]):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim) for vocab_size in vocab_sizes
        ])

    def forward(self, sparse_features: torch.Tensor) -> torch.Tensor:
        # sparse_features: (batch, num_sparse_features)
        embedded = []
        for i, emb in enumerate(self.embeddings):
            embedded.append(emb(sparse_features[:, i]))
        return torch.cat(embedded, dim=-1)


class DCNv2Block(nn.Module):
    """DCNv2-style deformable cross layer using level1 operators."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Linear(input_dim, input_dim, bias=False)
        self.offset = nn.Linear(input_dim, input_dim)
        self.modulation = nn.Linear(input_dim, input_dim)
        self.bias = nn.Parameter(torch.zeros(input_dim))
        
        self.sigmoid = Sigmoid()

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Compute offset and modulation
        offset = self.offset(x)
        mod = self.sigmoid(self.modulation(x))
        
        # Apply deformable transformation
        x_shifted = x + offset
        x_modulated = x_shifted * mod
        
        # Cross interaction
        out = x0 * (self.weight(x_modulated) + self.bias) + x
        return out


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    DCNv2 (Deep & Cross Network v2) for recommendation systems.
    
    Uses level1 operators from KernelBench:
    - LayerNorm from level1/normalization/6_LayerNorm
    - ReLU from level1/activations/1_ReLU
    - Sigmoid from level1/activations/3_Sigmoid
    - MatMul from level1/matmul/1_MatMul
    
    Supports variants: Criteo (configs from config_loader)
    """
    
    VARIANTS = VARIANTS
    
    @classmethod
    def from_pretrained(cls, variant: str = "Criteo", operator_level: Optional[OperatorLevel] = None, **kwargs):
        """Create model with config loaded from config_loader."""
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
        sparse_feature_dim = kwargs.get('sparse_feature_dim', kwargs.get('num_sparse_features', 26))
        dense_feature_dim = kwargs.get('dense_feature_dim', kwargs.get('num_dense_features', 13))
        embedding_dim = kwargs.get('embedding_dim', 16)
        cross_layers = kwargs.get('cross_layers', 6)
        deep_layers = kwargs.get('deep_layers', kwargs.get('deep_dims', [1024, 512, 256]))
        vocab_sizes = kwargs.get('vocab_sizes', None)
        num_classes = kwargs.get('num_classes', 1)
        
        if config is None:
            config = ModelConfig(
                hidden_size=embedding_dim * sparse_feature_dim + dense_feature_dim,
            )
        
        super().__init__()
        
        if vocab_sizes is None:
            vocab_sizes = [1000] * sparse_feature_dim
        
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        
        # Embedding layer for sparse features
        self.embedding = EmbeddingLayer(sparse_feature_dim, embedding_dim, vocab_sizes)
        
        # Input dimension after embedding
        input_dim = embedding_dim * sparse_feature_dim + dense_feature_dim
        
        # Cross network
        self.cross_network = CrossNetworkV2(input_dim, cross_layers)
        
        # Deep network
        self.deep_network = DeepNetwork(input_dim, deep_layers)
        
        # Output layer
        self.output = nn.Linear(input_dim + deep_layers[-1], num_classes)
        self.sigmoid = Sigmoid()

    def forward(
        self,
        sparse_features: torch.Tensor,
        dense_features: torch.Tensor,
    ) -> torch.Tensor:
        # Embed sparse features
        sparse_emb = self.embedding(sparse_features)
        
        # Concatenate with dense features
        x0 = torch.cat([sparse_emb, dense_features], dim=-1)
        
        # Cross network
        cross_out = self.cross_network(x0)
        
        # Deep network
        deep_out = self.deep_network(x0)
        
        # Combine and predict
        combined = torch.cat([cross_out, deep_out], dim=-1)
        logits = self.output(combined)
        
        return self.sigmoid(logits)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 1024
sparse_feature_dim = 26
dense_feature_dim = 13
embedding_dim = 16


def get_inputs():
    sparse_features = torch.randint(0, 1000, (batch_size, sparse_feature_dim))
    dense_features = torch.randn(batch_size, dense_feature_dim)
    return [sparse_features, dense_features]


def get_init_inputs():
    return [{
        'sparse_feature_dim': sparse_feature_dim,
        'dense_feature_dim': dense_feature_dim,
        'embedding_dim': embedding_dim,
        'cross_layers': 6,
        'deep_layers': [256, 128, 64],
        'num_classes': 1,
    }]
