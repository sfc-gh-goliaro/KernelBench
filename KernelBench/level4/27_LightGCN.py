"""
LightGCN (Light Graph Convolution Network)

Implements the LightGCN model for collaborative filtering recommendation:
- Learns user/item embeddings by linearly propagating on a bipartite graph
- No feature transformation or nonlinear activation (unlike GCN)
- Final embedding = weighted sum of embeddings at all layers
- Scoring via dot product between user and item embeddings

Reference: https://arxiv.org/abs/2002.02126
PyG implementation: torch_geometric.nn.models.LightGCN

Architecture:
    x^(0) = Embedding(num_nodes, dim)
    x^(l+1)_i = sum_{j in N(i)} 1/sqrt(deg(i)*deg(j)) * x^(l)_j   (LGConv)
    x_final_i = sum_{l=0}^{L} alpha_l * x^(l)_i
    score(u, v) = x_final_u . x_final_v

Variants:
- MovieLens-small: small graph for testing
- Gowalla: larger graph matching Gowalla benchmark scale

This model is dependency-free (no PyG required). Graph convolution is
implemented via sparse matrix multiplication using torch.sparse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union


# ============================================================================
# Model Variants
# ============================================================================

VARIANTS: Dict[str, Dict[str, Any]] = {
    "MovieLens-small": {
        "num_nodes": 3000,       # ~1000 users + ~2000 items
        "embedding_dim": 64,
        "num_layers": 3,
    },
    "Gowalla": {
        "num_nodes": 107092,     # 29858 users + 40981 items (standard split)
        "embedding_dim": 64,
        "num_layers": 3,
    },
    "Amazon-Book": {
        "num_nodes": 105283,     # 52643 users + 91599 items
        "embedding_dim": 64,
        "num_layers": 3,
    },
}


# ============================================================================
# Component Modules
# ============================================================================

def gcn_norm(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute symmetric normalization coefficients for GCN-style propagation.

    Given edge_index of shape (2, E), computes:
        norm_weight[e] = edge_weight[e] / sqrt(deg(src[e]) * deg(dst[e]))

    This matches PyG's gcn_norm with add_self_loops=False.

    Args:
        edge_index: (2, num_edges) source and target node indices
        num_nodes: total number of nodes in the graph
        edge_weight: (num_edges,) optional edge weights, defaults to 1.0

    Returns:
        edge_index: unchanged
        norm_weight: (num_edges,) normalized edge weights
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    row, col = edge_index[0], edge_index[1]

    # Compute degree: deg[i] = sum of edge_weight for edges with target i
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, row, edge_weight)

    # Symmetric normalization: 1 / sqrt(deg(src) * deg(dst))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

    norm_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, norm_weight


class LGConvLayer(nn.Module):
    """
    Light Graph Convolution layer.

    Performs one step of neighborhood aggregation with symmetric normalization:
        x'_i = sum_{j in N(i)} (1 / sqrt(deg(i) * deg(j))) * x_j

    Implemented via sparse matrix multiplication for efficiency.
    No learnable parameters (pure aggregation).

    This matches PyG's LGConv operator.
    """
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (num_nodes, embedding_dim) node features
            edge_index: (2, num_edges) graph connectivity
            edge_weight: (num_edges,) optional edge weights

        Returns:
            (num_nodes, embedding_dim) updated node features
        """
        num_nodes = x.size(0)

        if self.normalize:
            edge_index, edge_weight = gcn_norm(
                edge_index, num_nodes, edge_weight
            )

        # Build sparse adjacency matrix and multiply
        adj = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(num_nodes, num_nodes)
        ).coalesce()
        return torch.sparse.mm(adj, x)


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    LightGCN for collaborative filtering recommendation.

    Learns embeddings by linearly propagating them on the underlying
    user-item interaction graph. Uses the weighted sum of embeddings
    at all layers as the final embedding.

    Supports link prediction (forward) and top-K recommendation (recommend).

    This implementation is dependency-free (no PyG) and matches the behavior
    of torch_geometric.nn.models.LightGCN exactly.

    Supports variants: MovieLens-small, Gowalla, Amazon-Book
    """

    VARIANTS = VARIANTS

    def __init__(self, **kwargs):
        super().__init__()

        self.num_nodes = kwargs.get("num_nodes", 3000)
        self.embedding_dim = kwargs.get("embedding_dim", 64)
        self.num_layers = kwargs.get("num_layers", 3)
        alpha = kwargs.get("alpha", None)

        # Alpha weights for layer aggregation
        if alpha is None:
            alpha = 1.0 / (self.num_layers + 1)
        if isinstance(alpha, torch.Tensor):
            assert alpha.size(0) == self.num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (self.num_layers + 1))
        self.register_buffer("alpha", alpha)

        # Node embeddings
        self.embedding = nn.Embedding(self.num_nodes, self.embedding_dim)

        # LGConv layers (no learnable params, but kept as ModuleList for structure)
        self.convs = nn.ModuleList(
            [LGConvLayer(normalize=True) for _ in range(self.num_layers)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset all learnable parameters (Xavier uniform for embeddings)."""
        nn.init.xavier_uniform_(self.embedding.weight)

    def get_embedding(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute final node embeddings after all layers of propagation.

        Args:
            edge_index: (2, num_edges) graph connectivity
            edge_weight: (num_edges,) optional edge weights

        Returns:
            (num_nodes, embedding_dim) final node embeddings
        """
        x = self.embedding.weight
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            out = out + x * self.alpha[i + 1]

        return out

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_label_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute link prediction scores for pairs of nodes.

        Args:
            edge_index: (2, num_edges) graph connectivity for propagation
            edge_label_index: (2, num_eval_edges) node pairs to score.
                             If None, scores all edges in edge_index.
            edge_weight: (num_edges,) optional edge weights

        Returns:
            (num_eval_edges,) dot-product scores for each pair
        """
        if edge_label_index is None:
            edge_label_index = edge_index

        out = self.get_embedding(edge_index, edge_weight)

        out_src = out[edge_label_index[0]]
        out_dst = out[edge_label_index[1]]

        return (out_src * out_dst).sum(dim=-1)

    def recommend(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        src_index: Optional[torch.Tensor] = None,
        dst_index: Optional[torch.Tensor] = None,
        k: int = 1,
    ) -> torch.Tensor:
        """
        Get top-k recommendations for source nodes.

        Args:
            edge_index: (2, num_edges) graph connectivity
            edge_weight: (num_edges,) optional edge weights
            src_index: node indices to get recommendations for (default: all)
            dst_index: candidate node indices (default: all)
            k: number of recommendations per source node

        Returns:
            (num_src, k) indices of top-k recommended nodes
        """
        out_src = out_dst = self.get_embedding(edge_index, edge_weight)

        if src_index is not None:
            out_src = out_src[src_index]
        if dst_index is not None:
            out_dst = out_dst[dst_index]

        pred = out_src @ out_dst.t()
        top_index = pred.topk(k, dim=-1, sorted=True).indices

        if dst_index is not None:
            top_index = dst_index[top_index.view(-1)].view(*top_index.size())

        return top_index
