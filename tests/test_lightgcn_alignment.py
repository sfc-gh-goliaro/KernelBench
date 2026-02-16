"""
Test LightGCN alignment with PyTorch Geometric reference implementation.

This test validates that the KernelBench LightGCN implementation produces
outputs matching PyG's torch_geometric.nn.models.LightGCN using shared
random weights on synthetic bipartite user-item graphs.

Tests:
1. GCN normalization alignment
2. Single LGConv layer alignment
3. get_embedding (multi-layer propagation + alpha aggregation) alignment
4. forward (link prediction scores) alignment
5. recommend (top-K) alignment

Usage:
    pytest tests/test_lightgcn_alignment.py -v

Requires: torch_geometric (pip install torch_geometric)
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add paths for imports
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(TEST_DIR, "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "KernelBench"))

# Skip all tests if torch_geometric is not available
pyg = pytest.importorskip("torch_geometric")

from torch_geometric.nn.models import LightGCN as PyGLightGCN
from torch_geometric.nn.conv import LGConv as PyGLGConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm as pyg_gcn_norm

import importlib
lightgcn_module = importlib.import_module("KernelBench.level4.36_LightGCN")

# ============================================================================
# Test Configuration
# ============================================================================

DEVICE = "cpu"
DTYPE = torch.float32

# Our sparse-mm implementation matches PyG's message-passing to within
# floating-point epsilon (~1.5e-8 max diff). We use tight tolerances.
ATOL = 1e-7
RTOL = 1e-7

# Synthetic graph config
NUM_USERS = 200
NUM_ITEMS = 300
NUM_NODES = NUM_USERS + NUM_ITEMS
NUM_EDGES = 5000  # user-item interactions
EMBEDDING_DIM = 64
NUM_LAYERS = 3


# ============================================================================
# Helpers
# ============================================================================

def _create_synthetic_bipartite_graph(
    num_users, num_items, num_edges, device=DEVICE, seed=42
):
    """
    Create a synthetic user-item bipartite graph.

    Returns edge_index as (2, 2*num_edges) -- bidirectional edges for
    an undirected bipartite graph, which is the standard for LightGCN.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Random user-item edges
    users = torch.randint(0, num_users, (num_edges,), device=device)
    items = torch.randint(num_users, num_users + num_items, (num_edges,), device=device)

    # Make undirected (add reverse edges)
    edge_index = torch.stack(
        [torch.cat([users, items]), torch.cat([items, users])], dim=0
    )

    return edge_index


def _copy_weights_pyg_to_kb(pyg_model, kb_model):
    """Copy embedding weights from PyG model to KernelBench model."""
    kb_model.embedding.weight.data.copy_(pyg_model.embedding.weight.data)


@pytest.fixture(scope="module")
def graph_and_models():
    """Create PyG and KernelBench LightGCN models with shared weights."""
    torch.manual_seed(42)
    np.random.seed(42)

    edge_index = _create_synthetic_bipartite_graph(
        NUM_USERS, NUM_ITEMS, NUM_EDGES
    )

    # Create evaluation edge pairs (subset of edges)
    eval_size = 500
    eval_users = torch.randint(0, NUM_USERS, (eval_size,), device=DEVICE)
    eval_items = torch.randint(NUM_USERS, NUM_NODES, (eval_size,), device=DEVICE)
    edge_label_index = torch.stack([eval_users, eval_items], dim=0)

    # PyG reference model
    pyg_model = PyGLightGCN(
        num_nodes=NUM_NODES,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
    ).to(DEVICE).eval()

    # KernelBench model
    kb_model = lightgcn_module.Model(
        num_nodes=NUM_NODES,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
    ).to(DEVICE).eval()

    # Copy weights
    _copy_weights_pyg_to_kb(pyg_model, kb_model)

    return pyg_model, kb_model, edge_index, edge_label_index


# ============================================================================
# Tests
# ============================================================================

class TestLightGCNNormalization:
    """Test that our GCN normalization matches PyG's."""

    def test_gcn_norm_alignment(self, graph_and_models):
        """Our gcn_norm should produce the same normalized weights as PyG."""
        _, _, edge_index, _ = graph_and_models

        # PyG gcn_norm
        pyg_ei, pyg_ew = pyg_gcn_norm(
            edge_index, num_nodes=NUM_NODES, add_self_loops=False
        )

        # Our gcn_norm
        kb_ei, kb_ew = lightgcn_module.gcn_norm(edge_index, NUM_NODES)

        # Edge indices should be identical
        assert torch.equal(pyg_ei, kb_ei), "Edge indices differ"

        max_diff = (pyg_ew - kb_ew).abs().max().item()
        print(f"GCN norm max diff: {max_diff:.2e}")
        assert torch.allclose(pyg_ew, kb_ew, atol=ATOL, rtol=RTOL), (
            f"GCN norm weight mismatch: max_diff={max_diff:.2e}"
        )


class TestLightGCNLayerAlignment:
    """Test single LGConv layer alignment."""

    def test_single_lgconv_layer(self, graph_and_models):
        """Single LGConv layer should match PyG's LGConv."""
        pyg_model, kb_model, edge_index, _ = graph_and_models

        x = pyg_model.embedding.weight.detach().clone()

        with torch.no_grad():
            # PyG single layer
            pyg_conv = pyg_model.convs[0]
            pyg_out = pyg_conv(x, edge_index)

            # KernelBench single layer
            kb_conv = kb_model.convs[0]
            kb_out = kb_conv(x, edge_index)

        max_diff = (pyg_out - kb_out).abs().max().item()
        print(f"Single LGConv layer max diff: {max_diff:.2e}")
        assert torch.allclose(pyg_out, kb_out, atol=ATOL, rtol=RTOL), (
            f"LGConv mismatch: max_diff={max_diff:.2e}"
        )


class TestLightGCNEmbeddingAlignment:
    """Test multi-layer embedding propagation alignment."""

    def test_get_embedding(self, graph_and_models):
        """get_embedding (all layers + alpha aggregation) should match."""
        pyg_model, kb_model, edge_index, _ = graph_and_models

        with torch.no_grad():
            pyg_emb = pyg_model.get_embedding(edge_index)
            kb_emb = kb_model.get_embedding(edge_index)

        max_diff = (pyg_emb - kb_emb).abs().max().item()
        mean_diff = (pyg_emb - kb_emb).abs().mean().item()
        print(f"get_embedding max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e}")
        print(f"Embedding shape: {pyg_emb.shape}")
        assert torch.allclose(pyg_emb, kb_emb, atol=ATOL, rtol=RTOL), (
            f"get_embedding mismatch: max_diff={max_diff:.2e}"
        )

    def test_layer_by_layer_propagation(self, graph_and_models):
        """Check each layer's output individually."""
        pyg_model, kb_model, edge_index, _ = graph_and_models

        x_pyg = pyg_model.embedding.weight.detach().clone()
        x_kb = kb_model.embedding.weight.detach().clone()

        assert torch.equal(x_pyg, x_kb), "Initial embeddings differ"

        with torch.no_grad():
            for layer_idx in range(NUM_LAYERS):
                x_pyg = pyg_model.convs[layer_idx](x_pyg, edge_index)
                x_kb = kb_model.convs[layer_idx](x_kb, edge_index)

                max_diff = (x_pyg - x_kb).abs().max().item()
                print(f"  Layer {layer_idx}: max_diff={max_diff:.2e}")
                assert torch.allclose(x_pyg, x_kb, atol=ATOL, rtol=RTOL), (
                    f"Layer {layer_idx} mismatch: max_diff={max_diff:.2e}"
                )

    def test_alpha_weights_match(self, graph_and_models):
        """Alpha weights should be identical."""
        pyg_model, kb_model, _, _ = graph_and_models
        assert torch.allclose(pyg_model.alpha, kb_model.alpha, atol=1e-7), (
            f"Alpha mismatch: pyg={pyg_model.alpha}, kb={kb_model.alpha}"
        )


class TestLightGCNForwardAlignment:
    """Test forward pass (link prediction) alignment."""

    def test_forward_scores(self, graph_and_models):
        """forward() link prediction scores should match."""
        pyg_model, kb_model, edge_index, edge_label_index = graph_and_models

        with torch.no_grad():
            pyg_scores = pyg_model(edge_index, edge_label_index)
            kb_scores = kb_model(edge_index, edge_label_index)

        max_diff = (pyg_scores - kb_scores).abs().max().item()
        print(f"Forward scores max diff: {max_diff:.2e}")
        print(f"Scores shape: {pyg_scores.shape}")
        print(f"Score range: [{pyg_scores.min().item():.4f}, {pyg_scores.max().item():.4f}]")
        assert torch.allclose(pyg_scores, kb_scores, atol=ATOL, rtol=RTOL), (
            f"Forward scores mismatch: max_diff={max_diff:.2e}"
        )

    def test_forward_default_edge_label(self, graph_and_models):
        """forward() with edge_label_index=None should match."""
        pyg_model, kb_model, edge_index, _ = graph_and_models

        with torch.no_grad():
            pyg_scores = pyg_model(edge_index)
            kb_scores = kb_model(edge_index)

        max_diff = (pyg_scores - kb_scores).abs().max().item()
        print(f"Forward (default edge_label) max diff: {max_diff:.2e}")
        assert torch.allclose(pyg_scores, kb_scores, atol=ATOL, rtol=RTOL), (
            f"Forward (default) mismatch: max_diff={max_diff:.2e}"
        )


class TestLightGCNRecommendAlignment:
    """Test recommend (top-K) alignment."""

    def test_recommend_top_k(self, graph_and_models):
        """recommend() top-K indices should match."""
        pyg_model, kb_model, edge_index, _ = graph_and_models

        src_index = torch.arange(0, 20, device=DEVICE)  # first 20 users
        dst_index = torch.arange(NUM_USERS, NUM_NODES, device=DEVICE)  # all items
        k = 10

        with torch.no_grad():
            pyg_top = pyg_model.recommend(
                edge_index, src_index=src_index, dst_index=dst_index, k=k
            )
            kb_top = kb_model.recommend(
                edge_index, src_index=src_index, dst_index=dst_index, k=k
            )

        print(f"Recommend top-{k} shape: {pyg_top.shape}")
        match = (pyg_top == kb_top).all().item()
        if not match:
            mismatches = (pyg_top != kb_top).sum().item()
            print(f"  Mismatches: {mismatches}/{pyg_top.numel()}")
        else:
            print(f"  All top-{k} recommendations match exactly")
        assert match, "Top-K recommendations differ"


class TestLightGCNEdgeCases:
    """Test edge cases and different configurations."""

    def test_different_num_layers(self):
        """Test alignment with different layer counts."""
        torch.manual_seed(123)
        edge_index = _create_synthetic_bipartite_graph(50, 80, 500, seed=123)

        for n_layers in [1, 2, 4, 5]:
            pyg_m = PyGLightGCN(
                num_nodes=130, embedding_dim=32, num_layers=n_layers
            ).eval()
            kb_m = lightgcn_module.Model(
                num_nodes=130, embedding_dim=32, num_layers=n_layers
            ).eval()
            _copy_weights_pyg_to_kb(pyg_m, kb_m)

            with torch.no_grad():
                pyg_emb = pyg_m.get_embedding(edge_index)
                kb_emb = kb_m.get_embedding(edge_index)

            max_diff = (pyg_emb - kb_emb).abs().max().item()
            print(f"  num_layers={n_layers}: max_diff={max_diff:.2e}")
            assert torch.allclose(pyg_emb, kb_emb, atol=ATOL, rtol=RTOL), (
                f"Mismatch at num_layers={n_layers}: max_diff={max_diff:.2e}"
            )

    def test_custom_alpha(self):
        """Test alignment with custom alpha weights."""
        torch.manual_seed(456)
        edge_index = _create_synthetic_bipartite_graph(50, 80, 500, seed=456)
        n_layers = 3
        alpha = torch.tensor([0.5, 0.3, 0.15, 0.05])

        pyg_m = PyGLightGCN(
            num_nodes=130, embedding_dim=32, num_layers=n_layers, alpha=alpha
        ).eval()
        kb_m = lightgcn_module.Model(
            num_nodes=130, embedding_dim=32, num_layers=n_layers, alpha=alpha
        ).eval()
        _copy_weights_pyg_to_kb(pyg_m, kb_m)

        with torch.no_grad():
            pyg_emb = pyg_m.get_embedding(edge_index)
            kb_emb = kb_m.get_embedding(edge_index)

        max_diff = (pyg_emb - kb_emb).abs().max().item()
        print(f"Custom alpha max diff: {max_diff:.2e}")
        assert torch.allclose(pyg_emb, kb_emb, atol=ATOL, rtol=RTOL), (
            f"Custom alpha mismatch: max_diff={max_diff:.2e}"
        )

    def test_parameter_count_match(self, graph_and_models):
        """Both models should have the same number of parameters."""
        pyg_model, kb_model, _, _ = graph_and_models

        pyg_params = sum(p.numel() for p in pyg_model.parameters())
        kb_params = sum(p.numel() for p in kb_model.parameters())

        print(f"PyG params: {pyg_params:,}")
        print(f"KernelBench params: {kb_params:,}")
        assert pyg_params == kb_params, (
            f"Parameter count mismatch: pyg={pyg_params}, kb={kb_params}"
        )
