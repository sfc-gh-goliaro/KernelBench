"""
DLRMv2 (Deep Learning Recommendation Model v2)

Implements Meta's DLRM architecture for click-through rate prediction:
- Bottom MLP: transforms dense (continuous) features
- Embedding tables: lookup for sparse (categorical) features via EmbeddingBag
- Dot interaction: pairwise dot products between all feature vectors
- Top MLP: final prediction from interaction features
- Sigmoid output: click probability

Reference: https://github.com/facebookresearch/dlrm
Paper: https://arxiv.org/abs/1906.00091
MLPerf benchmark: https://docs.mlcommons.org/inference/benchmarks/recommendation/dlrm-v2/

Architecture:
    dense_features -> BottomMLP -> [bot_out]
    sparse_features -> EmbeddingBag(sum) -> [emb_1, ..., emb_k]
    [bot_out, emb_1, ..., emb_k] -> DotInteraction (pairwise dots, lower triangle)
    [bot_out, interaction_flat] -> TopMLP -> Sigmoid -> click_probability

Variants (Criteo dataset standard: 13 dense + 26 sparse features):
- Criteo-small: small embedding dim + shallow MLPs for testing
- Criteo-base: MLPerf-standard dimensions
- Criteo-large: larger embeddings + deeper MLPs

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, List


# ============================================================================
# Model Variants
# ============================================================================

VARIANTS: Dict[str, Dict[str, Any]] = {
    "Criteo-small": {
        "num_dense_features": 13,
        "num_sparse_features": 26,
        "sparse_feature_size": 16,
        "vocab_sizes": [1000] * 26,
        "bot_mlp_sizes": [13, 512, 256, 16],
        "top_mlp_sizes": [367, 256, 1],
        "interaction_op": "dot",
    },
    "Criteo-base": {
        "num_dense_features": 13,
        "num_sparse_features": 26,
        "sparse_feature_size": 64,
        "vocab_sizes": [
            40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63,
            40000000, 3067956, 405282, 10, 2209, 11938, 155, 4, 976,
            14, 40000000, 40000000, 40000000, 590152, 12973, 108, 36,
        ],
        "bot_mlp_sizes": [13, 512, 256, 64],
        "top_mlp_sizes": [415, 512, 256, 1],
        "interaction_op": "dot",
    },
    "Criteo-large": {
        "num_dense_features": 13,
        "num_sparse_features": 26,
        "sparse_feature_size": 128,
        "vocab_sizes": [
            40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63,
            40000000, 3067956, 405282, 10, 2209, 11938, 155, 4, 976,
            14, 40000000, 40000000, 40000000, 590152, 12973, 108, 36,
        ],
        "bot_mlp_sizes": [13, 512, 256, 128],
        "top_mlp_sizes": [479, 1024, 512, 256, 1],
        "interaction_op": "dot",
    },
}


# ============================================================================
# Component Modules
# ============================================================================

class DLRM_MLP(nn.Module):
    """
    MLP used for both bottom and top networks in DLRM.

    Follows the reference facebookresearch/dlrm implementation:
    - Sequential Linear + ReLU layers
    - Final layer uses Sigmoid (for top MLP) or ReLU (for bottom MLP)
    """
    def __init__(self, layer_sizes: List[int], sigmoid_layer: int = -1):
        super().__init__()
        layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            n = layer_sizes[i]
            m = layer_sizes[i + 1]
            layers.append(nn.Linear(n, m))
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DotInteraction(nn.Module):
    """
    DLRM dot-product feature interaction.

    Given the bottom MLP output and a list of embedding vectors (all same dim),
    concatenates them into a matrix T of shape (batch, num_vectors, dim),
    computes T @ T^T, extracts the strict lower triangular entries,
    and concatenates with the bottom MLP output.

    This matches the `interact_features` method with `arch_interaction_op="dot"`
    from facebookresearch/dlrm.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self, bottom_mlp_output: torch.Tensor, embedding_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        batch_size, d = bottom_mlp_output.shape
        # Stack all vectors: (batch, num_vectors, d)
        T = torch.cat(
            [bottom_mlp_output.unsqueeze(1)]
            + [e.unsqueeze(1) for e in embedding_outputs],
            dim=1,
        )
        # Pairwise dot products: (batch, num_vectors, num_vectors)
        Z = torch.bmm(T, T.transpose(1, 2))
        # Extract strict lower triangular (excluding diagonal)
        _, ni, nj = Z.shape
        li = torch.tensor(
            [i for i in range(ni) for j in range(i)], device=Z.device
        )
        lj = torch.tensor(
            [j for i in range(nj) for j in range(i)], device=Z.device
        )
        Zflat = Z[:, li, lj]
        # Concatenate bottom MLP output with interaction features
        R = torch.cat([bottom_mlp_output, Zflat], dim=1)
        return R


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    DLRMv2 (Deep Learning Recommendation Model v2).

    Faithfully implements the DLRM architecture from facebookresearch/dlrm:
    1. Bottom MLP processes dense features
    2. EmbeddingBag(sum) lookups for sparse features
    3. Dot-product interaction between all feature vectors
    4. Top MLP produces final click probability

    Uses level1 operators from KernelBench where applicable.

    Supports variants: Criteo-small, Criteo-base, Criteo-large
    """

    VARIANTS = VARIANTS

    def __init__(self, **kwargs):
        super().__init__()

        self.num_dense_features = kwargs.get("num_dense_features", 13)
        self.num_sparse_features = kwargs.get("num_sparse_features", 26)
        self.sparse_feature_size = kwargs.get("sparse_feature_size", 16)
        vocab_sizes = kwargs.get("vocab_sizes", [1000] * self.num_sparse_features)
        bot_mlp_sizes = kwargs.get("bot_mlp_sizes", [13, 512, 256, 16])
        top_mlp_sizes = kwargs.get("top_mlp_sizes", [367, 256, 1])
        self.interaction_op = kwargs.get("interaction_op", "dot")

        # Bottom MLP: dense features -> embedding-sized vector
        self.bot_mlp = DLRM_MLP(bot_mlp_sizes)

        # Embedding tables: one EmbeddingBag per sparse feature
        self.emb_l = nn.ModuleList(
            [
                nn.EmbeddingBag(vocab_sizes[i], self.sparse_feature_size, mode="sum")
                for i in range(self.num_sparse_features)
            ]
        )

        # Dot interaction
        self.interaction = DotInteraction()

        # Top MLP: interaction output -> click probability
        # sigmoid on the last layer
        self.top_mlp = DLRM_MLP(top_mlp_sizes, sigmoid_layer=len(top_mlp_sizes) - 2)

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_indices: List[torch.Tensor],
        sparse_offsets: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass of DLRM.

        Args:
            dense_features: (batch_size, num_dense_features) continuous features
            sparse_indices: list of num_sparse_features tensors, each containing
                           indices into the corresponding embedding table
            sparse_offsets: list of num_sparse_features tensors, each containing
                           offsets for EmbeddingBag pooling

        Returns:
            (batch_size, 1) click probabilities
        """
        # Bottom MLP
        x = self.bot_mlp(dense_features)

        # Embedding lookups
        ly = []
        for k in range(len(self.emb_l)):
            V = self.emb_l[k](sparse_indices[k], sparse_offsets[k])
            ly.append(V)

        # Feature interaction
        z = self.interaction(x, ly)

        # Top MLP
        p = self.top_mlp(z)

        return p
