"""
DCN-V2 (Deep & Cross Network V2) Recommendation Model

A recommendation model implementing DCN-V2 architecture:
- Sparse embedding layer for categorical features
- Dense embedding layer for numerical features
- Cross Network V2 with matrix decomposition
- Deep MLP for non-linear interactions
- Stacked or parallel structure

Reference: DCN-V2: Improved Deep & Cross Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SparseEmbedding(nn.Module):
    """Embedding layer for sparse categorical features."""
    def __init__(
        self,
        num_embeddings_list: List[int],
        embedding_dim: int,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_emb, embedding_dim)
            for num_emb in num_embeddings_list
        ])

    def forward(self, sparse_inputs: torch.Tensor) -> torch.Tensor:
        # sparse_inputs: (batch_size, num_sparse_features)
        embedded = [emb(sparse_inputs[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(embedded, dim=1)  # (batch_size, num_sparse * embedding_dim)


class DenseEmbedding(nn.Module):
    """Embedding layer for dense numerical features."""
    def __init__(self, num_dense_features: int, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(num_dense_features, embedding_dim)

    def forward(self, dense_inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(dense_inputs)


class CrossNetworkV2(nn.Module):
    """
    Cross Network V2 with matrix decomposition.
    
    Implements: x_{l+1} = x_0 * (U @ (V @ x_l) + b) + x_l
    Where U, V are low-rank matrices for efficiency.
    """
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        low_rank: int = 64,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.V_list = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, low_rank) * 0.01)
            for _ in range(num_layers)
        ])
        self.U_list = nn.ParameterList([
            nn.Parameter(torch.randn(low_rank, input_dim) * 0.01)
            for _ in range(num_layers)
        ])
        self.bias_list = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for i in range(self.num_layers):
            # x_{l+1} = x_0 * (U @ (V @ x_l) + b) + x_l
            v_out = torch.matmul(x, self.V_list[i])  # (batch, low_rank)
            u_out = torch.matmul(v_out, self.U_list[i])  # (batch, input_dim)
            cross = x0 * (u_out + self.bias_list[i])
            x = cross + x
        return x


class DeepNetwork(nn.Module):
    """Deep MLP for non-linear feature interactions."""
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Model(nn.Module):
    """DCN-V2 Recommendation Model."""
    def __init__(
        self,
        # Sparse features
        num_sparse_features: int = 26,
        sparse_vocab_sizes: Optional[List[int]] = None,
        # Dense features
        num_dense_features: int = 13,
        # Embedding
        embedding_dim: int = 16,
        # Cross Network
        cross_num_layers: int = 3,
        cross_low_rank: int = 64,
        # Deep Network
        deep_hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.1,
        # Structure: "stacked" or "parallel"
        structure: str = "stacked",
    ):
        super().__init__()
        self.structure = structure

        # Default vocab sizes
        if sparse_vocab_sizes is None:
            sparse_vocab_sizes = [10000] * num_sparse_features

        # Embedding layers
        self.sparse_embedding = SparseEmbedding(sparse_vocab_sizes, embedding_dim)
        self.dense_embedding = DenseEmbedding(num_dense_features, embedding_dim)

        # Total input dimension
        input_dim = num_sparse_features * embedding_dim + embedding_dim

        # Cross Network
        self.cross_network = CrossNetworkV2(input_dim, cross_num_layers, cross_low_rank)

        # Deep Network
        if structure == "stacked":
            # Stacked: Cross -> Deep
            self.deep_network = DeepNetwork(input_dim, deep_hidden_dims, dropout)
            final_dim = deep_hidden_dims[-1]
        else:
            # Parallel: Cross and Deep run in parallel
            self.deep_network = DeepNetwork(input_dim, deep_hidden_dims, dropout)
            final_dim = input_dim + deep_hidden_dims[-1]

        # Output layer
        self.output_layer = nn.Linear(final_dim, 1)

    def forward(
        self,
        sparse_inputs: torch.Tensor,
        dense_inputs: torch.Tensor,
    ) -> torch.Tensor:
        # Embed features
        sparse_emb = self.sparse_embedding(sparse_inputs)  # (B, num_sparse * emb_dim)
        dense_emb = self.dense_embedding(dense_inputs)  # (B, emb_dim)

        # Concatenate embeddings
        x0 = torch.cat([sparse_emb, dense_emb], dim=1)

        if self.structure == "stacked":
            # Stacked: Cross -> Deep
            cross_out = self.cross_network(x0)
            deep_out = self.deep_network(cross_out)
            final = deep_out
        else:
            # Parallel: Cross and Deep
            cross_out = self.cross_network(x0)
            deep_out = self.deep_network(x0)
            final = torch.cat([cross_out, deep_out], dim=1)

        # Output (logit for CTR prediction)
        logit = self.output_layer(final)
        return torch.sigmoid(logit).squeeze(-1)


# Configuration
batch_size = 1024
num_sparse_features = 26
num_dense_features = 13
embedding_dim = 16

sparse_vocab_sizes = [1000 + i * 100 for i in range(num_sparse_features)]

cross_num_layers = 3
cross_low_rank = 64
deep_hidden_dims = [512, 256, 128]


def get_inputs():
    sparse_inputs = torch.stack([
        torch.randint(0, vocab_size, (batch_size,))
        for vocab_size in sparse_vocab_sizes
    ], dim=1)
    dense_inputs = torch.randn(batch_size, num_dense_features)
    return [sparse_inputs, dense_inputs]


def get_init_inputs():
    return [{
        'num_sparse_features': num_sparse_features,
        'sparse_vocab_sizes': sparse_vocab_sizes,
        'num_dense_features': num_dense_features,
        'embedding_dim': embedding_dim,
        'cross_num_layers': cross_num_layers,
        'cross_low_rank': cross_low_rank,
        'deep_hidden_dims': deep_hidden_dims,
        'structure': 'stacked',
    }]

