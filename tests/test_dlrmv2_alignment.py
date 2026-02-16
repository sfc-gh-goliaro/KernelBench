"""
Test DLRMv2 alignment with a minimal reference implementation.

This test validates that the KernelBench DLRMv2 implementation produces
outputs matching a reference DLRM forward pass built from first principles,
following the architecture from facebookresearch/dlrm.

The reference is a minimal, self-contained DLRM implementation written
directly in this file (~50 lines) that matches the canonical DLRM_Net
from facebookresearch/dlrm (sequential_forward path). We use shared
random weights to verify numerical alignment.

Tests:
1. Bottom MLP alignment
2. Embedding lookup alignment
3. Dot interaction alignment
4. Full forward pass (end-to-end) alignment

Usage:
    pytest tests/test_dlrmv2_alignment.py -v
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

import importlib
dlrmv2_module = importlib.import_module("KernelBench.level4.35_DLRMv2")

# ============================================================================
# Test Configuration
# ============================================================================

DEVICE = "cpu"
DTYPE = torch.float32

# Our implementation shares the same PyTorch ops and code paths as the
# reference, so outputs are bit-exact (max diff = 0.0). We use very tight
# tolerances to catch any regressions.
ATOL = 1e-7
RTOL = 1e-7

# Small config for fast testing
TEST_CONFIG = {
    "num_dense_features": 13,
    "num_sparse_features": 26,
    "sparse_feature_size": 16,
    "vocab_sizes": [1000] * 26,
    "bot_mlp_sizes": [13, 512, 256, 16],
    "top_mlp_sizes": [367, 256, 1],
    "interaction_op": "dot",
}

BATCH_SIZE = 32
NUM_INDICES_PER_LOOKUP = 1  # single index per sparse feature per sample


# ============================================================================
# Minimal Reference DLRM Implementation
# ============================================================================

class ReferenceDLRM(nn.Module):
    """
    Minimal reference DLRM matching facebookresearch/dlrm DLRM_Net.

    This is a clean reimplementation of the sequential_forward path
    from dlrm_s_pytorch.py, stripped of all distributed/quantization code.
    """

    def __init__(
        self,
        bot_mlp_sizes,
        top_mlp_sizes,
        sparse_feature_size,
        vocab_sizes,
    ):
        super().__init__()
        # Bottom MLP
        bot_layers = []
        for i in range(len(bot_mlp_sizes) - 1):
            bot_layers.append(nn.Linear(bot_mlp_sizes[i], bot_mlp_sizes[i + 1]))
            bot_layers.append(nn.ReLU())
        self.bot_mlp = nn.Sequential(*bot_layers)

        # Embedding tables
        self.emb_l = nn.ModuleList(
            [nn.EmbeddingBag(vs, sparse_feature_size, mode="sum") for vs in vocab_sizes]
        )

        # Top MLP (sigmoid on last layer)
        top_layers = []
        for i in range(len(top_mlp_sizes) - 1):
            top_layers.append(nn.Linear(top_mlp_sizes[i], top_mlp_sizes[i + 1]))
            if i == len(top_mlp_sizes) - 2:
                top_layers.append(nn.Sigmoid())
            else:
                top_layers.append(nn.ReLU())
        self.top_mlp = nn.Sequential(*top_layers)

    def interact_features(self, x, ly):
        """Dot interaction: exactly matches facebookresearch/dlrm."""
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
        ly = []
        for k in range(len(self.emb_l)):
            V = self.emb_l[k](sparse_indices[k], sparse_offsets[k])
            ly.append(V)
        z = self.interact_features(x, ly)
        p = self.top_mlp(z)
        return p

    def get_bot_mlp_output(self, dense_x):
        return self.bot_mlp(dense_x)

    def get_embeddings(self, sparse_indices, sparse_offsets):
        ly = []
        for k in range(len(self.emb_l)):
            V = self.emb_l[k](sparse_indices[k], sparse_offsets[k])
            ly.append(V)
        return ly

    def get_interaction_output(self, x, ly):
        return self.interact_features(x, ly)


# ============================================================================
# Fixtures
# ============================================================================

def _create_synthetic_input(batch_size, config, device=DEVICE):
    """Create synthetic Criteo-like input data."""
    dense = torch.randn(batch_size, config["num_dense_features"], device=device)

    sparse_indices = []
    sparse_offsets = []
    for i in range(config["num_sparse_features"]):
        # Single index per sample (num_indices_per_lookup=1)
        indices = torch.randint(
            0, config["vocab_sizes"][i], (batch_size,), device=device
        )
        offsets = torch.arange(batch_size, device=device)
        sparse_indices.append(indices)
        sparse_offsets.append(offsets)

    return dense, sparse_indices, sparse_offsets


def _copy_weights_ref_to_kb(ref_model, kb_model):
    """Copy all weights from reference model to KernelBench model."""
    # Bottom MLP
    ref_bot_state = ref_model.bot_mlp.state_dict()
    kb_model.bot_mlp.layers.load_state_dict(ref_bot_state)

    # Embeddings
    for k in range(len(ref_model.emb_l)):
        kb_model.emb_l[k].load_state_dict(ref_model.emb_l[k].state_dict())

    # Top MLP
    ref_top_state = ref_model.top_mlp.state_dict()
    kb_model.top_mlp.layers.load_state_dict(ref_top_state)


@pytest.fixture(scope="module")
def models_and_input():
    """Create reference and KernelBench models with shared weights and input."""
    torch.manual_seed(42)
    np.random.seed(42)

    config = TEST_CONFIG

    ref_model = ReferenceDLRM(
        bot_mlp_sizes=config["bot_mlp_sizes"],
        top_mlp_sizes=config["top_mlp_sizes"],
        sparse_feature_size=config["sparse_feature_size"],
        vocab_sizes=config["vocab_sizes"],
    ).to(DEVICE).eval()

    kb_model = dlrmv2_module.Model(**config).to(DEVICE).eval()

    _copy_weights_ref_to_kb(ref_model, kb_model)

    dense, sparse_indices, sparse_offsets = _create_synthetic_input(BATCH_SIZE, config)

    return ref_model, kb_model, dense, sparse_indices, sparse_offsets, config


# ============================================================================
# Tests
# ============================================================================

class TestDLRMv2Alignment:
    """Tests for numerical alignment between KernelBench and reference DLRM."""

    def test_bottom_mlp_alignment(self, models_and_input):
        """Bottom MLP should produce identical outputs."""
        ref_model, kb_model, dense, _, _, _ = models_and_input

        with torch.no_grad():
            ref_out = ref_model.get_bot_mlp_output(dense)
            kb_out = kb_model.bot_mlp(dense)

        max_diff = (ref_out - kb_out).abs().max().item()
        print(f"Bottom MLP max diff: {max_diff:.2e}")
        assert torch.allclose(ref_out, kb_out, atol=ATOL, rtol=RTOL), (
            f"Bottom MLP mismatch: max_diff={max_diff:.2e}"
        )

    def test_embedding_alignment(self, models_and_input):
        """Embedding lookups should produce identical outputs."""
        ref_model, kb_model, _, sparse_indices, sparse_offsets, config = models_and_input

        with torch.no_grad():
            ref_embs = ref_model.get_embeddings(sparse_indices, sparse_offsets)
            kb_embs = []
            for k in range(config["num_sparse_features"]):
                V = kb_model.emb_l[k](sparse_indices[k], sparse_offsets[k])
                kb_embs.append(V)

        for k in range(config["num_sparse_features"]):
            max_diff = (ref_embs[k] - kb_embs[k]).abs().max().item()
            assert torch.allclose(ref_embs[k], kb_embs[k], atol=ATOL, rtol=RTOL), (
                f"Embedding {k} mismatch: max_diff={max_diff:.2e}"
            )

        print(f"All {config['num_sparse_features']} embeddings aligned (max_diff=0.0)")

    def test_dot_interaction_alignment(self, models_and_input):
        """Dot interaction should produce identical outputs."""
        ref_model, kb_model, dense, sparse_indices, sparse_offsets, _ = models_and_input

        with torch.no_grad():
            # Get intermediate values
            bot_out = ref_model.get_bot_mlp_output(dense)
            embs = ref_model.get_embeddings(sparse_indices, sparse_offsets)

            ref_interact = ref_model.get_interaction_output(bot_out, embs)
            kb_interact = kb_model.interaction(bot_out, embs)

        max_diff = (ref_interact - kb_interact).abs().max().item()
        print(f"Dot interaction max diff: {max_diff:.2e}")
        print(f"Interaction output shape: {ref_interact.shape}")
        assert torch.allclose(ref_interact, kb_interact, atol=ATOL, rtol=RTOL), (
            f"Dot interaction mismatch: max_diff={max_diff:.2e}"
        )

    def test_full_forward_alignment(self, models_and_input):
        """Full forward pass should produce identical outputs."""
        ref_model, kb_model, dense, sparse_indices, sparse_offsets, _ = models_and_input

        with torch.no_grad():
            ref_out = ref_model(dense, sparse_indices, sparse_offsets)
            kb_out = kb_model(dense, sparse_indices, sparse_offsets)

        max_diff = (ref_out - kb_out).abs().max().item()
        print(f"Full forward max diff: {max_diff:.2e}")
        print(f"Output shape: {ref_out.shape}")
        print(f"Output range: [{ref_out.min().item():.4f}, {ref_out.max().item():.4f}]")
        assert torch.allclose(ref_out, kb_out, atol=ATOL, rtol=RTOL), (
            f"Full forward mismatch: max_diff={max_diff:.2e}"
        )

    def test_multiple_batch_sizes(self, models_and_input):
        """Test alignment across different batch sizes."""
        ref_model, kb_model, _, _, _, config = models_and_input

        for bs in [1, 8, 64, 128]:
            torch.manual_seed(bs)
            dense, sparse_indices, sparse_offsets = _create_synthetic_input(bs, config)

            with torch.no_grad():
                ref_out = ref_model(dense, sparse_indices, sparse_offsets)
                kb_out = kb_model(dense, sparse_indices, sparse_offsets)

            max_diff = (ref_out - kb_out).abs().max().item()
            print(f"  batch_size={bs}: max_diff={max_diff:.2e}")
            assert torch.allclose(ref_out, kb_out, atol=ATOL, rtol=RTOL), (
                f"Mismatch at batch_size={bs}: max_diff={max_diff:.2e}"
            )

    def test_multi_hot_sparse_features(self, models_and_input):
        """Test with multi-hot sparse features (multiple indices per lookup)."""
        ref_model, kb_model, _, _, _, config = models_and_input

        batch_size = 16
        num_indices_per_lookup = 3

        dense = torch.randn(batch_size, config["num_dense_features"], device=DEVICE)
        sparse_indices = []
        sparse_offsets = []
        for i in range(config["num_sparse_features"]):
            # Multiple indices per sample
            indices = torch.randint(
                0,
                config["vocab_sizes"][i],
                (batch_size * num_indices_per_lookup,),
                device=DEVICE,
            )
            offsets = torch.arange(0, batch_size * num_indices_per_lookup,
                                   num_indices_per_lookup, device=DEVICE)
            sparse_indices.append(indices)
            sparse_offsets.append(offsets)

        with torch.no_grad():
            ref_out = ref_model(dense, sparse_indices, sparse_offsets)
            kb_out = kb_model(dense, sparse_indices, sparse_offsets)

        max_diff = (ref_out - kb_out).abs().max().item()
        print(f"Multi-hot (k={num_indices_per_lookup}) max diff: {max_diff:.2e}")
        assert torch.allclose(ref_out, kb_out, atol=ATOL, rtol=RTOL), (
            f"Multi-hot mismatch: max_diff={max_diff:.2e}"
        )

    def test_output_range(self, models_and_input):
        """Output should be in [0, 1] (sigmoid output)."""
        _, kb_model, dense, sparse_indices, sparse_offsets, _ = models_and_input

        with torch.no_grad():
            out = kb_model(dense, sparse_indices, sparse_offsets)

        assert (out >= 0.0).all() and (out <= 1.0).all(), (
            f"Output out of [0,1] range: [{out.min().item()}, {out.max().item()}]"
        )
        print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")

    def test_parameter_count_match(self, models_and_input):
        """Both models should have the same number of parameters."""
        ref_model, kb_model, _, _, _, _ = models_and_input

        ref_params = sum(p.numel() for p in ref_model.parameters())
        kb_params = sum(p.numel() for p in kb_model.parameters())

        print(f"Reference params: {ref_params:,}")
        print(f"KernelBench params: {kb_params:,}")
        assert ref_params == kb_params, (
            f"Parameter count mismatch: ref={ref_params}, kb={kb_params}"
        )
