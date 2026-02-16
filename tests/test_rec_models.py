"""
Test alignment for recommendation models: DLRMv2 and LightGCN.

DLRMv2:
  Validates that the KernelBench DLRMv2 implementation produces outputs
  matching a reference DLRM forward pass built from first principles,
  following the architecture from facebookresearch/dlrm.

  The reference is a minimal, self-contained DLRM implementation written
  directly in this file that matches the canonical DLRM_Net from
  facebookresearch/dlrm (sequential_forward path). We use shared random
  weights to verify numerical alignment.

LightGCN:
  Validates that the KernelBench LightGCN implementation produces outputs
  matching PyG's torch_geometric.nn.models.LightGCN using shared random
  weights on synthetic bipartite user-item graphs.

All tests run on GPU when available, falling back to CPU otherwise.

Usage:
    pytest tests/test_rec_models.py -v

Requires: torch_geometric (pip install torch_geometric) for LightGCN tests
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import importlib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(TEST_DIR, "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "KernelBench"))

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# Both implementations share the same PyTorch ops and code paths as their
# references, so outputs are bit-exact (max diff = 0.0). We use very tight
# tolerances to catch any regressions.
ATOL = 1e-7
RTOL = 1e-7


# ============================================================================
#  DLRMv2 alignment tests
# ============================================================================

dlrmv2_module = importlib.import_module("KernelBench.level4.26_DLRMv2")

# Small config for fast testing
DLRM_TEST_CONFIG = {
    "num_dense_features": 13,
    "num_sparse_features": 26,
    "sparse_feature_size": 16,
    "vocab_sizes": [1000] * 26,
    "bot_mlp_sizes": [13, 512, 256, 16],
    "top_mlp_sizes": [367, 256, 1],
    "interaction_op": "dot",
}
DLRM_BATCH_SIZE = 32


# -- Reference DLRM ---------------------------------------------------------

class ReferenceDLRM(nn.Module):
    """
    Minimal reference DLRM matching facebookresearch/dlrm DLRM_Net.

    This is a clean reimplementation of the sequential_forward path
    from dlrm_s_pytorch.py, stripped of all distributed/quantization code.
    """

    def __init__(self, bot_mlp_sizes, top_mlp_sizes, sparse_feature_size, vocab_sizes):
        super().__init__()
        bot_layers = []
        for i in range(len(bot_mlp_sizes) - 1):
            bot_layers.append(nn.Linear(bot_mlp_sizes[i], bot_mlp_sizes[i + 1]))
            bot_layers.append(nn.ReLU())
        self.bot_mlp = nn.Sequential(*bot_layers)

        self.emb_l = nn.ModuleList(
            [nn.EmbeddingBag(vs, sparse_feature_size, mode="sum") for vs in vocab_sizes]
        )

        top_layers = []
        for i in range(len(top_mlp_sizes) - 1):
            top_layers.append(nn.Linear(top_mlp_sizes[i], top_mlp_sizes[i + 1]))
            if i == len(top_mlp_sizes) - 2:
                top_layers.append(nn.Sigmoid())
            else:
                top_layers.append(nn.ReLU())
        self.top_mlp = nn.Sequential(*top_layers)

    def interact_features(self, x, ly):
        batch_size, d = x.shape
        T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
        Z = torch.bmm(T, torch.transpose(T, 1, 2))
        _, ni, nj = Z.shape
        li = torch.tensor([i for i in range(ni) for j in range(i)], device=Z.device)
        lj = torch.tensor([j for i in range(nj) for j in range(i)], device=Z.device)
        Zflat = Z[:, li, lj]
        R = torch.cat([x] + [Zflat], dim=1)
        return R

    def forward(self, dense_x, sparse_indices, sparse_offsets):
        x = self.bot_mlp(dense_x)
        ly = [self.emb_l[k](sparse_indices[k], sparse_offsets[k]) for k in range(len(self.emb_l))]
        z = self.interact_features(x, ly)
        return self.top_mlp(z)

    def get_bot_mlp_output(self, dense_x):
        return self.bot_mlp(dense_x)

    def get_embeddings(self, sparse_indices, sparse_offsets):
        return [self.emb_l[k](sparse_indices[k], sparse_offsets[k]) for k in range(len(self.emb_l))]

    def get_interaction_output(self, x, ly):
        return self.interact_features(x, ly)


# -- Helpers -----------------------------------------------------------------

def _dlrm_create_input(batch_size, config, device=DEVICE):
    """Create synthetic Criteo-like input data on *device*."""
    dense = torch.randn(batch_size, config["num_dense_features"], device=device)
    sparse_indices, sparse_offsets = [], []
    for i in range(config["num_sparse_features"]):
        indices = torch.randint(0, config["vocab_sizes"][i], (batch_size,), device=device)
        offsets = torch.arange(batch_size, device=device)
        sparse_indices.append(indices)
        sparse_offsets.append(offsets)
    return dense, sparse_indices, sparse_offsets


def _dlrm_copy_weights(ref_model, kb_model):
    """Copy all weights from reference model to KernelBench model."""
    kb_model.bot_mlp.layers.load_state_dict(ref_model.bot_mlp.state_dict())
    for k in range(len(ref_model.emb_l)):
        kb_model.emb_l[k].load_state_dict(ref_model.emb_l[k].state_dict())
    kb_model.top_mlp.layers.load_state_dict(ref_model.top_mlp.state_dict())


@pytest.fixture(scope="module")
def dlrm_models_and_input():
    """Create reference and KernelBench DLRMv2 models with shared weights."""
    torch.manual_seed(42)
    np.random.seed(42)

    cfg = DLRM_TEST_CONFIG
    ref = ReferenceDLRM(
        bot_mlp_sizes=cfg["bot_mlp_sizes"],
        top_mlp_sizes=cfg["top_mlp_sizes"],
        sparse_feature_size=cfg["sparse_feature_size"],
        vocab_sizes=cfg["vocab_sizes"],
    ).to(DEVICE).eval()

    kb = dlrmv2_module.Model(**cfg).to(DEVICE).eval()
    _dlrm_copy_weights(ref, kb)

    dense, sparse_idx, sparse_off = _dlrm_create_input(DLRM_BATCH_SIZE, cfg)
    return ref, kb, dense, sparse_idx, sparse_off, cfg


# -- Test class --------------------------------------------------------------

class TestDLRMv2Alignment:
    """Numerical alignment between KernelBench and reference DLRM."""

    def test_bottom_mlp_alignment(self, dlrm_models_and_input):
        ref, kb, dense, *_ = dlrm_models_and_input
        with torch.no_grad():
            ref_out = ref.get_bot_mlp_output(dense)
            kb_out = kb.bot_mlp(dense)
        max_diff = (ref_out - kb_out).abs().max().item()
        print(f"Bottom MLP max diff: {max_diff:.2e}")
        assert torch.allclose(ref_out, kb_out, atol=ATOL, rtol=RTOL), (
            f"Bottom MLP mismatch: max_diff={max_diff:.2e}"
        )

    def test_embedding_alignment(self, dlrm_models_and_input):
        ref, kb, _, sparse_idx, sparse_off, cfg = dlrm_models_and_input
        with torch.no_grad():
            ref_embs = ref.get_embeddings(sparse_idx, sparse_off)
            kb_embs = [kb.emb_l[k](sparse_idx[k], sparse_off[k])
                       for k in range(cfg["num_sparse_features"])]
        for k in range(cfg["num_sparse_features"]):
            max_diff = (ref_embs[k] - kb_embs[k]).abs().max().item()
            assert torch.allclose(ref_embs[k], kb_embs[k], atol=ATOL, rtol=RTOL), (
                f"Embedding {k} mismatch: max_diff={max_diff:.2e}"
            )
        print(f"All {cfg['num_sparse_features']} embeddings aligned")

    def test_dot_interaction_alignment(self, dlrm_models_and_input):
        ref, kb, dense, sparse_idx, sparse_off, _ = dlrm_models_and_input
        with torch.no_grad():
            bot_out = ref.get_bot_mlp_output(dense)
            embs = ref.get_embeddings(sparse_idx, sparse_off)
            ref_interact = ref.get_interaction_output(bot_out, embs)
            kb_interact = kb.interaction(bot_out, embs)
        max_diff = (ref_interact - kb_interact).abs().max().item()
        print(f"Dot interaction max diff: {max_diff:.2e}")
        assert torch.allclose(ref_interact, kb_interact, atol=ATOL, rtol=RTOL), (
            f"Dot interaction mismatch: max_diff={max_diff:.2e}"
        )

    def test_full_forward_alignment(self, dlrm_models_and_input):
        ref, kb, dense, sparse_idx, sparse_off, _ = dlrm_models_and_input
        with torch.no_grad():
            ref_out = ref(dense, sparse_idx, sparse_off)
            kb_out = kb(dense, sparse_idx, sparse_off)
        max_diff = (ref_out - kb_out).abs().max().item()
        print(f"Full forward max diff: {max_diff:.2e}")
        assert torch.allclose(ref_out, kb_out, atol=ATOL, rtol=RTOL), (
            f"Full forward mismatch: max_diff={max_diff:.2e}"
        )

    def test_multiple_batch_sizes(self, dlrm_models_and_input):
        ref, kb, _, _, _, cfg = dlrm_models_and_input
        for bs in [1, 8, 64, 128]:
            torch.manual_seed(bs)
            dense, sparse_idx, sparse_off = _dlrm_create_input(bs, cfg)
            with torch.no_grad():
                ref_out = ref(dense, sparse_idx, sparse_off)
                kb_out = kb(dense, sparse_idx, sparse_off)
            max_diff = (ref_out - kb_out).abs().max().item()
            print(f"  batch_size={bs}: max_diff={max_diff:.2e}")
            assert torch.allclose(ref_out, kb_out, atol=ATOL, rtol=RTOL), (
                f"Mismatch at batch_size={bs}: max_diff={max_diff:.2e}"
            )

    def test_multi_hot_sparse_features(self, dlrm_models_and_input):
        ref, kb, _, _, _, cfg = dlrm_models_and_input
        batch_size = 16
        num_indices_per_lookup = 3
        dense = torch.randn(batch_size, cfg["num_dense_features"], device=DEVICE)
        sparse_indices, sparse_offsets = [], []
        for i in range(cfg["num_sparse_features"]):
            indices = torch.randint(
                0, cfg["vocab_sizes"][i],
                (batch_size * num_indices_per_lookup,), device=DEVICE,
            )
            offsets = torch.arange(
                0, batch_size * num_indices_per_lookup,
                num_indices_per_lookup, device=DEVICE,
            )
            sparse_indices.append(indices)
            sparse_offsets.append(offsets)
        with torch.no_grad():
            ref_out = ref(dense, sparse_indices, sparse_offsets)
            kb_out = kb(dense, sparse_indices, sparse_offsets)
        max_diff = (ref_out - kb_out).abs().max().item()
        print(f"Multi-hot (k={num_indices_per_lookup}) max diff: {max_diff:.2e}")
        assert torch.allclose(ref_out, kb_out, atol=ATOL, rtol=RTOL), (
            f"Multi-hot mismatch: max_diff={max_diff:.2e}"
        )

    def test_output_range(self, dlrm_models_and_input):
        _, kb, dense, sparse_idx, sparse_off, _ = dlrm_models_and_input
        with torch.no_grad():
            out = kb(dense, sparse_idx, sparse_off)
        assert (out >= 0.0).all() and (out <= 1.0).all(), (
            f"Output out of [0,1] range: [{out.min().item()}, {out.max().item()}]"
        )
        print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")

    def test_parameter_count_match(self, dlrm_models_and_input):
        ref, kb, *_ = dlrm_models_and_input
        ref_params = sum(p.numel() for p in ref.parameters())
        kb_params = sum(p.numel() for p in kb.parameters())
        print(f"Reference params: {ref_params:,}  KernelBench params: {kb_params:,}")
        assert ref_params == kb_params, (
            f"Parameter count mismatch: ref={ref_params}, kb={kb_params}"
        )


# ============================================================================
#  LightGCN alignment tests
# ============================================================================

# Skip the entire LightGCN section if torch_geometric is not installed.
pyg = pytest.importorskip("torch_geometric")

from torch_geometric.nn.models import LightGCN as PyGLightGCN  # noqa: E402
from torch_geometric.nn.conv import LGConv as PyGLGConv  # noqa: E402
from torch_geometric.nn.conv.gcn_conv import gcn_norm as pyg_gcn_norm  # noqa: E402

lightgcn_module = importlib.import_module("KernelBench.level4.27_LightGCN")

# Synthetic graph config
GCN_NUM_USERS = 200
GCN_NUM_ITEMS = 300
GCN_NUM_NODES = GCN_NUM_USERS + GCN_NUM_ITEMS
GCN_NUM_EDGES = 5000
GCN_EMBEDDING_DIM = 64
GCN_NUM_LAYERS = 3


# -- Helpers -----------------------------------------------------------------

def _gcn_create_bipartite_graph(num_users, num_items, num_edges, device=DEVICE, seed=42):
    """Create a synthetic user-item bipartite graph on *device*."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    users = torch.randint(0, num_users, (num_edges,), device=device)
    items = torch.randint(num_users, num_users + num_items, (num_edges,), device=device)
    edge_index = torch.stack(
        [torch.cat([users, items]), torch.cat([items, users])], dim=0,
    )
    return edge_index


def _gcn_copy_weights(pyg_model, kb_model):
    """Copy embedding weights from PyG model to KernelBench model."""
    kb_model.embedding.weight.data.copy_(pyg_model.embedding.weight.data)


@pytest.fixture(scope="module")
def gcn_graph_and_models():
    """Create PyG and KernelBench LightGCN models with shared weights on GPU."""
    torch.manual_seed(42)
    np.random.seed(42)

    edge_index = _gcn_create_bipartite_graph(GCN_NUM_USERS, GCN_NUM_ITEMS, GCN_NUM_EDGES)

    eval_size = 500
    eval_users = torch.randint(0, GCN_NUM_USERS, (eval_size,), device=DEVICE)
    eval_items = torch.randint(GCN_NUM_USERS, GCN_NUM_NODES, (eval_size,), device=DEVICE)
    edge_label_index = torch.stack([eval_users, eval_items], dim=0)

    pyg_model = PyGLightGCN(
        num_nodes=GCN_NUM_NODES,
        embedding_dim=GCN_EMBEDDING_DIM,
        num_layers=GCN_NUM_LAYERS,
    ).to(DEVICE).eval()

    kb_model = lightgcn_module.Model(
        num_nodes=GCN_NUM_NODES,
        embedding_dim=GCN_EMBEDDING_DIM,
        num_layers=GCN_NUM_LAYERS,
    ).to(DEVICE).eval()

    _gcn_copy_weights(pyg_model, kb_model)

    return pyg_model, kb_model, edge_index, edge_label_index


# -- Test classes ------------------------------------------------------------

class TestLightGCNNormalization:
    """Test that our GCN normalization matches PyG's."""

    def test_gcn_norm_alignment(self, gcn_graph_and_models):
        _, _, edge_index, _ = gcn_graph_and_models
        pyg_ei, pyg_ew = pyg_gcn_norm(edge_index, num_nodes=GCN_NUM_NODES, add_self_loops=False)
        kb_ei, kb_ew = lightgcn_module.gcn_norm(edge_index, GCN_NUM_NODES)
        assert torch.equal(pyg_ei, kb_ei), "Edge indices differ"
        max_diff = (pyg_ew - kb_ew).abs().max().item()
        print(f"GCN norm max diff: {max_diff:.2e}")
        assert torch.allclose(pyg_ew, kb_ew, atol=ATOL, rtol=RTOL), (
            f"GCN norm weight mismatch: max_diff={max_diff:.2e}"
        )


class TestLightGCNLayerAlignment:
    """Test single LGConv layer alignment."""

    def test_single_lgconv_layer(self, gcn_graph_and_models):
        pyg_model, kb_model, edge_index, _ = gcn_graph_and_models
        x = pyg_model.embedding.weight.detach().clone()
        with torch.no_grad():
            pyg_out = pyg_model.convs[0](x, edge_index)
            kb_out = kb_model.convs[0](x, edge_index)
        max_diff = (pyg_out - kb_out).abs().max().item()
        print(f"Single LGConv layer max diff: {max_diff:.2e}")
        assert torch.allclose(pyg_out, kb_out, atol=ATOL, rtol=RTOL), (
            f"LGConv mismatch: max_diff={max_diff:.2e}"
        )


class TestLightGCNEmbeddingAlignment:
    """Test multi-layer embedding propagation alignment."""

    def test_get_embedding(self, gcn_graph_and_models):
        pyg_model, kb_model, edge_index, _ = gcn_graph_and_models
        with torch.no_grad():
            pyg_emb = pyg_model.get_embedding(edge_index)
            kb_emb = kb_model.get_embedding(edge_index)
        max_diff = (pyg_emb - kb_emb).abs().max().item()
        mean_diff = (pyg_emb - kb_emb).abs().mean().item()
        print(f"get_embedding max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e}")
        assert torch.allclose(pyg_emb, kb_emb, atol=ATOL, rtol=RTOL), (
            f"get_embedding mismatch: max_diff={max_diff:.2e}"
        )

    def test_layer_by_layer_propagation(self, gcn_graph_and_models):
        pyg_model, kb_model, edge_index, _ = gcn_graph_and_models
        x_pyg = pyg_model.embedding.weight.detach().clone()
        x_kb = kb_model.embedding.weight.detach().clone()
        assert torch.equal(x_pyg, x_kb), "Initial embeddings differ"
        with torch.no_grad():
            for idx in range(GCN_NUM_LAYERS):
                x_pyg = pyg_model.convs[idx](x_pyg, edge_index)
                x_kb = kb_model.convs[idx](x_kb, edge_index)
                max_diff = (x_pyg - x_kb).abs().max().item()
                print(f"  Layer {idx}: max_diff={max_diff:.2e}")
                assert torch.allclose(x_pyg, x_kb, atol=ATOL, rtol=RTOL), (
                    f"Layer {idx} mismatch: max_diff={max_diff:.2e}"
                )

    def test_alpha_weights_match(self, gcn_graph_and_models):
        pyg_model, kb_model, _, _ = gcn_graph_and_models
        assert torch.allclose(pyg_model.alpha, kb_model.alpha, atol=1e-7), (
            f"Alpha mismatch: pyg={pyg_model.alpha}, kb={kb_model.alpha}"
        )


class TestLightGCNForwardAlignment:
    """Test forward pass (link prediction) alignment."""

    def test_forward_scores(self, gcn_graph_and_models):
        pyg_model, kb_model, edge_index, edge_label_index = gcn_graph_and_models
        with torch.no_grad():
            pyg_scores = pyg_model(edge_index, edge_label_index)
            kb_scores = kb_model(edge_index, edge_label_index)
        max_diff = (pyg_scores - kb_scores).abs().max().item()
        print(f"Forward scores max diff: {max_diff:.2e}")
        assert torch.allclose(pyg_scores, kb_scores, atol=ATOL, rtol=RTOL), (
            f"Forward scores mismatch: max_diff={max_diff:.2e}"
        )

    def test_forward_default_edge_label(self, gcn_graph_and_models):
        pyg_model, kb_model, edge_index, _ = gcn_graph_and_models
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

    def test_recommend_top_k(self, gcn_graph_and_models):
        pyg_model, kb_model, edge_index, _ = gcn_graph_and_models
        src_index = torch.arange(0, 20, device=DEVICE)
        dst_index = torch.arange(GCN_NUM_USERS, GCN_NUM_NODES, device=DEVICE)
        k = 10
        with torch.no_grad():
            pyg_top = pyg_model.recommend(
                edge_index, src_index=src_index, dst_index=dst_index, k=k,
            )
            kb_top = kb_model.recommend(
                edge_index, src_index=src_index, dst_index=dst_index, k=k,
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
        torch.manual_seed(123)
        edge_index = _gcn_create_bipartite_graph(50, 80, 500, seed=123)
        for n_layers in [1, 2, 4, 5]:
            pyg_m = PyGLightGCN(
                num_nodes=130, embedding_dim=32, num_layers=n_layers,
            ).to(DEVICE).eval()
            kb_m = lightgcn_module.Model(
                num_nodes=130, embedding_dim=32, num_layers=n_layers,
            ).to(DEVICE).eval()
            _gcn_copy_weights(pyg_m, kb_m)
            with torch.no_grad():
                pyg_emb = pyg_m.get_embedding(edge_index)
                kb_emb = kb_m.get_embedding(edge_index)
            max_diff = (pyg_emb - kb_emb).abs().max().item()
            print(f"  num_layers={n_layers}: max_diff={max_diff:.2e}")
            assert torch.allclose(pyg_emb, kb_emb, atol=ATOL, rtol=RTOL), (
                f"Mismatch at num_layers={n_layers}: max_diff={max_diff:.2e}"
            )

    def test_custom_alpha(self):
        torch.manual_seed(456)
        edge_index = _gcn_create_bipartite_graph(50, 80, 500, seed=456)
        n_layers = 3
        alpha = torch.tensor([0.5, 0.3, 0.15, 0.05])
        pyg_m = PyGLightGCN(
            num_nodes=130, embedding_dim=32, num_layers=n_layers, alpha=alpha,
        ).to(DEVICE).eval()
        kb_m = lightgcn_module.Model(
            num_nodes=130, embedding_dim=32, num_layers=n_layers, alpha=alpha,
        ).to(DEVICE).eval()
        _gcn_copy_weights(pyg_m, kb_m)
        with torch.no_grad():
            pyg_emb = pyg_m.get_embedding(edge_index)
            kb_emb = kb_m.get_embedding(edge_index)
        max_diff = (pyg_emb - kb_emb).abs().max().item()
        print(f"Custom alpha max diff: {max_diff:.2e}")
        assert torch.allclose(pyg_emb, kb_emb, atol=ATOL, rtol=RTOL), (
            f"Custom alpha mismatch: max_diff={max_diff:.2e}"
        )

    def test_parameter_count_match(self, gcn_graph_and_models):
        pyg_model, kb_model, _, _ = gcn_graph_and_models
        pyg_params = sum(p.numel() for p in pyg_model.parameters())
        kb_params = sum(p.numel() for p in kb_model.parameters())
        print(f"PyG params: {pyg_params:,}  KernelBench params: {kb_params:,}")
        assert pyg_params == kb_params, (
            f"Parameter count mismatch: pyg={pyg_params}, kb={kb_params}"
        )
