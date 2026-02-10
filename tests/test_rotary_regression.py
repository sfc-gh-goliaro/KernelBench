"""
Regression tests for _1_RotaryEmbedding.

Verifies that the refactored RotaryEmbedding (unified inv_freq storage,
extracted _apply_rotary_half, new apply_rotary method) produces numerically
identical results to a reference implementation of the original code.

Tests cover:
1. forward() in half_rotate mode (default, llama3, yarn) with bshd/bhsd layouts
2. forward() in complex mode
3. apply_rotary() equivalence with forward() cos/sin
4. inv_freq float32 preservation after .to(bfloat16)
5. Exact numerical match against a standalone reference rotate_half
"""

import pytest
import math
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'KernelBench'))

from level1.embeddings._1_RotaryEmbedding import Model as RotaryEmbedding

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATOL = 1e-5
RTOL = 1e-5


# ============================================================================
# Reference implementations (standalone, no dependency on the module under test)
# ============================================================================

def ref_rotate_half(x):
    """Reference rotate_half: split last dim in half, negate-swap."""
    d = x.shape[-1]
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]
    return torch.cat((-x2, x1), dim=-1)


def ref_apply_rotary_half(q, k, cos, sin):
    """Reference RoPE application using rotate_half."""
    q_rot = q * cos + ref_rotate_half(q) * sin
    k_rot = k * cos + ref_rotate_half(k) * sin
    return q_rot, k_rot


def ref_compute_cos_sin(head_dim, seq_len, base=10000.0, position_ids=None, device="cpu"):
    """Reference cos/sin computation matching the original code."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim)
    )
    if position_ids is None:
        t = torch.arange(seq_len, device=device, dtype=torch.float)
    else:
        t = position_ids.float()
    freqs = torch.outer(t, inv_freq) if position_ids is None or position_ids.dim() == 1 else None
    if freqs is None:
        # batched position_ids
        inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        pos_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ pos_expanded).transpose(1, 2).squeeze(0)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


# ============================================================================
# Tests: forward() in half_rotate mode
# ============================================================================

class TestForwardHalfRotate:
    """Test that forward() in half_rotate mode matches reference."""

    @pytest.mark.parametrize("layout", ["bshd", "bhsd"])
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_forward_default_matches_reference(self, layout, head_dim):
        """forward() with default rope should match reference rotate_half."""
        batch, seq, heads = 2, 16, 4
        rope = RotaryEmbedding(head_dim, max_seq_len=seq, base=10000.0, layout=layout).to(DEVICE)
        rope.eval()

        if layout == "bshd":
            q = torch.randn(batch, seq, heads, head_dim, device=DEVICE)
            k = torch.randn(batch, seq, heads, head_dim, device=DEVICE)
        else:
            q = torch.randn(batch, heads, seq, head_dim, device=DEVICE)
            k = torch.randn(batch, heads, seq, head_dim, device=DEVICE)

        with torch.no_grad():
            q_rot, k_rot = rope(q, k)

        # Reference
        cos, sin = ref_compute_cos_sin(head_dim, seq, device=DEVICE)
        if layout == "bshd":
            cos_ref = cos.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim)
            sin_ref = sin.unsqueeze(0).unsqueeze(2)
        else:
            cos_ref = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
            sin_ref = sin.unsqueeze(0).unsqueeze(0)

        q_ref, k_ref = ref_apply_rotary_half(q, k, cos_ref, sin_ref)

        assert torch.allclose(q_rot, q_ref, atol=ATOL, rtol=RTOL), \
            f"q mismatch: max diff = {(q_rot - q_ref).abs().max()}"
        assert torch.allclose(k_rot, k_ref, atol=ATOL, rtol=RTOL), \
            f"k mismatch: max diff = {(k_rot - k_ref).abs().max()}"

    @pytest.mark.parametrize("layout", ["bshd", "bhsd"])
    def test_forward_with_position_ids(self, layout):
        """forward() with explicit position_ids should match reference."""
        batch, seq, heads, head_dim = 2, 16, 4, 64
        rope = RotaryEmbedding(head_dim, max_seq_len=32, base=10000.0, layout=layout).to(DEVICE)
        rope.eval()

        # Non-contiguous position_ids
        position_ids = torch.tensor([[0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15],
                                      [1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14]],
                                     device=DEVICE)

        if layout == "bshd":
            q = torch.randn(batch, seq, heads, head_dim, device=DEVICE)
            k = torch.randn(batch, seq, heads, head_dim, device=DEVICE)
        else:
            q = torch.randn(batch, heads, seq, head_dim, device=DEVICE)
            k = torch.randn(batch, heads, seq, head_dim, device=DEVICE)

        with torch.no_grad():
            q_rot, k_rot = rope(q, k, position_ids)

        # Reference: gather cos/sin by position_ids
        cos_full, sin_full = ref_compute_cos_sin(head_dim, 32, device=DEVICE)
        cos_gathered = cos_full[position_ids]  # (batch, seq, head_dim)
        sin_gathered = sin_full[position_ids]

        if layout == "bshd":
            cos_ref = cos_gathered.unsqueeze(2)  # (batch, seq, 1, head_dim)
            sin_ref = sin_gathered.unsqueeze(2)
        else:
            cos_ref = cos_gathered.unsqueeze(1)  # (batch, 1, seq, head_dim)
            sin_ref = sin_gathered.unsqueeze(1)

        q_ref, k_ref = ref_apply_rotary_half(q, k, cos_ref, sin_ref)

        assert torch.allclose(q_rot, q_ref, atol=ATOL, rtol=RTOL), \
            f"q mismatch: max diff = {(q_rot - q_ref).abs().max()}"
        assert torch.allclose(k_rot, k_ref, atol=ATOL, rtol=RTOL), \
            f"k mismatch: max diff = {(k_rot - k_ref).abs().max()}"

    def test_forward_llama3_rope(self):
        """forward() with llama3 rope_scaling should produce finite, non-trivial output."""
        head_dim = 128
        batch, seq, heads = 2, 16, 4
        rope_scaling = {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
        rope = RotaryEmbedding(
            head_dim, max_seq_len=seq, base=500000.0,
            layout="bhsd", rope_scaling=rope_scaling,
        ).to(DEVICE)
        rope.eval()

        q = torch.randn(batch, heads, seq, head_dim, device=DEVICE)
        k = torch.randn(batch, heads, seq, head_dim, device=DEVICE)

        with torch.no_grad():
            q_rot, k_rot = rope(q, k)

        assert torch.isfinite(q_rot).all(), "llama3 q_rot has non-finite values"
        assert torch.isfinite(k_rot).all(), "llama3 k_rot has non-finite values"
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        # Should not be identity
        assert not torch.allclose(q_rot, q, atol=1e-3), "llama3 RoPE should modify q"

    def test_forward_yarn_rope(self):
        """forward() with yarn rope_scaling should produce finite, non-trivial output."""
        head_dim = 128
        batch, seq, heads = 2, 16, 4
        rope_scaling = {
            "rope_type": "yarn",
            "factor": 40.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 0.707,
        }
        rope = RotaryEmbedding(
            head_dim, max_seq_len=seq, base=10000.0,
            layout="bshd", rope_scaling=rope_scaling,
            mode="complex",
        ).to(DEVICE)
        rope.eval()

        q = torch.randn(batch, heads, seq, head_dim, device=DEVICE)
        k = torch.randn(batch, heads, seq, head_dim, device=DEVICE)

        with torch.no_grad():
            q_rot, k_rot = rope(q, k)

        assert torch.isfinite(q_rot).all(), "yarn q_rot has non-finite values"
        assert torch.isfinite(k_rot).all(), "yarn k_rot has non-finite values"
        assert q_rot.shape == q.shape
        assert not torch.allclose(q_rot, q, atol=1e-3), "yarn RoPE should modify q"


# ============================================================================
# Tests: apply_rotary() method
# ============================================================================

class TestApplyRotary:
    """Test the new apply_rotary() method."""

    def test_apply_rotary_matches_forward(self):
        """apply_rotary(q, k, cos, sin) should match forward() when given the same cos/sin."""
        head_dim = 64
        batch, seq, heads = 2, 16, 4
        rope = RotaryEmbedding(head_dim, max_seq_len=seq, base=10000.0, layout="bhsd").to(DEVICE)
        rope.eval()

        q = torch.randn(batch, heads, seq, head_dim, device=DEVICE)
        k = torch.randn(batch, heads, seq, head_dim, device=DEVICE)

        # Get cos/sin from the module's cache
        cos = rope.cos_cached[:seq].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
        sin = rope.sin_cached[:seq].unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            q_fwd, k_fwd = rope(q, k)
            q_apply, k_apply = rope.apply_rotary(q, k, cos, sin)

        assert torch.allclose(q_fwd, q_apply, atol=ATOL, rtol=RTOL), \
            f"q mismatch: max diff = {(q_fwd - q_apply).abs().max()}"
        assert torch.allclose(k_fwd, k_apply, atol=ATOL, rtol=RTOL), \
            f"k mismatch: max diff = {(k_fwd - k_apply).abs().max()}"

    def test_apply_rotary_matches_reference(self):
        """apply_rotary() should match standalone reference rotate_half."""
        head_dim = 128
        seq, heads = 32, 8
        rope = RotaryEmbedding(head_dim, max_seq_len=seq, base=10000.0).to(DEVICE)
        rope.eval()

        # Arbitrary shapes (no batch dim) - mimics Qwen2VL vision usage
        q = torch.randn(seq, heads, head_dim, device=DEVICE)
        k = torch.randn(seq, heads, head_dim, device=DEVICE)

        cos, sin = ref_compute_cos_sin(head_dim, seq, device=DEVICE)
        cos = cos.unsqueeze(1)  # (seq, 1, head_dim)
        sin = sin.unsqueeze(1)

        with torch.no_grad():
            q_rot, k_rot = rope.apply_rotary(q, k, cos, sin)

        q_ref, k_ref = ref_apply_rotary_half(q, k, cos, sin)

        assert torch.allclose(q_rot, q_ref, atol=ATOL, rtol=RTOL), \
            f"q mismatch: max diff = {(q_rot - q_ref).abs().max()}"
        assert torch.allclose(k_rot, k_ref, atol=ATOL, rtol=RTOL), \
            f"k mismatch: max diff = {(k_rot - k_ref).abs().max()}"

    def test_apply_rotary_external_cos_sin(self):
        """apply_rotary() with externally computed cos/sin (Qwen2VL-like pattern)."""
        head_dim = 128
        batch, seq, heads = 1, 64, 8
        rope = RotaryEmbedding(head_dim, max_seq_len=1, base=10000.0).to(DEVICE)
        rope.eval()

        q = torch.randn(batch, heads, seq, head_dim, device=DEVICE)
        k = torch.randn(batch, heads, seq, head_dim, device=DEVICE)

        # Externally compute cos/sin (like Qwen2VL M-RoPE does)
        inv_freq = rope.inv_freq.float()
        positions = torch.arange(seq, device=DEVICE, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_ext = emb.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
        sin_ext = emb.sin().unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            q_rot, k_rot = rope.apply_rotary(q, k, cos_ext, sin_ext)

        q_ref, k_ref = ref_apply_rotary_half(q, k, cos_ext, sin_ext)

        assert torch.allclose(q_rot, q_ref, atol=ATOL, rtol=RTOL)
        assert torch.allclose(k_rot, k_ref, atol=ATOL, rtol=RTOL)


# ============================================================================
# Tests: inv_freq float32 preservation
# ============================================================================

class TestInvFreqPreservation:
    """Test that inv_freq stays float32 after dtype conversions."""

    def test_inv_freq_survives_bfloat16(self):
        """inv_freq should remain float32 after model.to(bfloat16)."""
        rope = RotaryEmbedding(64, max_seq_len=16, base=10000.0).to(DEVICE)
        rope = rope.to(torch.bfloat16)

        assert rope.inv_freq.dtype == torch.float32, \
            f"inv_freq dtype is {rope.inv_freq.dtype}, expected float32"
        assert rope._inv_freq_float32.dtype == torch.float32

    def test_inv_freq_survives_float16(self):
        """inv_freq should remain float32 after model.to(float16)."""
        rope = RotaryEmbedding(64, max_seq_len=16, base=10000.0).to(DEVICE)
        rope = rope.to(torch.float16)

        assert rope.inv_freq.dtype == torch.float32, \
            f"inv_freq dtype is {rope.inv_freq.dtype}, expected float32"

    def test_forward_after_bfloat16_conversion(self):
        """forward() should still work correctly after bfloat16 conversion."""
        head_dim = 64
        batch, seq, heads = 2, 16, 4
        rope = RotaryEmbedding(head_dim, max_seq_len=seq, base=10000.0, layout="bhsd").to(DEVICE)

        q = torch.randn(batch, heads, seq, head_dim, device=DEVICE)
        k = torch.randn(batch, heads, seq, head_dim, device=DEVICE)

        # Get reference in float32
        with torch.no_grad():
            q_ref, k_ref = rope(q, k)

        # Convert to bfloat16 and run with bfloat16 inputs
        rope_bf16 = rope.to(torch.bfloat16)
        q_bf16 = q.to(torch.bfloat16)
        k_bf16 = k.to(torch.bfloat16)

        with torch.no_grad():
            q_rot_bf16, k_rot_bf16 = rope_bf16(q_bf16, k_bf16)

        # Should be close (bfloat16 has lower precision)
        assert torch.allclose(q_rot_bf16.float(), q_ref, atol=0.05, rtol=0.05), \
            f"bfloat16 q max diff = {(q_rot_bf16.float() - q_ref).abs().max()}"


# ============================================================================
# Tests: Norm preservation (RoPE mathematical property)
# ============================================================================

class TestRoPEProperties:
    """Test mathematical properties of RoPE that must hold."""

    def test_norm_preservation(self):
        """RoPE should preserve the L2 norm of vectors."""
        head_dim = 64
        batch, seq, heads = 2, 16, 4
        rope = RotaryEmbedding(head_dim, max_seq_len=seq, base=10000.0, layout="bhsd").to(DEVICE)
        rope.eval()

        q = torch.randn(batch, heads, seq, head_dim, device=DEVICE)
        k = torch.randn(batch, heads, seq, head_dim, device=DEVICE)

        with torch.no_grad():
            q_rot, k_rot = rope(q, k)

        q_norm_before = q.norm(dim=-1)
        q_norm_after = q_rot.norm(dim=-1)
        k_norm_before = k.norm(dim=-1)
        k_norm_after = k_rot.norm(dim=-1)

        assert torch.allclose(q_norm_before, q_norm_after, atol=1e-4, rtol=1e-4), \
            f"q norm changed: max diff = {(q_norm_before - q_norm_after).abs().max()}"
        assert torch.allclose(k_norm_before, k_norm_after, atol=1e-4, rtol=1e-4), \
            f"k norm changed: max diff = {(k_norm_before - k_norm_after).abs().max()}"

    def test_different_positions_give_different_results(self):
        """Different position_ids should produce different rotations."""
        head_dim = 64
        batch, seq, heads = 1, 8, 2
        rope = RotaryEmbedding(head_dim, max_seq_len=32, base=10000.0, layout="bhsd").to(DEVICE)
        rope.eval()

        q = torch.randn(batch, heads, seq, head_dim, device=DEVICE)
        k = torch.randn(batch, heads, seq, head_dim, device=DEVICE)

        pos1 = torch.arange(seq, device=DEVICE).unsqueeze(0)
        pos2 = torch.arange(8, 16, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            q_rot1, _ = rope(q, k, pos1)
            q_rot2, _ = rope(q, k, pos2)

        assert not torch.allclose(q_rot1, q_rot2, atol=1e-3), \
            "Different positions should produce different rotations"

    def test_zero_position_is_identity_for_low_freq(self):
        """At position 0, cos=1 and sin=0, so rotation should be identity."""
        head_dim = 64
        batch, heads = 1, 2
        rope = RotaryEmbedding(head_dim, max_seq_len=16, base=10000.0, layout="bhsd").to(DEVICE)
        rope.eval()

        q = torch.randn(batch, heads, 1, head_dim, device=DEVICE)
        k = torch.randn(batch, heads, 1, head_dim, device=DEVICE)

        pos = torch.zeros(batch, 1, device=DEVICE, dtype=torch.long)

        with torch.no_grad():
            q_rot, k_rot = rope(q, k, pos)

        # At position 0, cos=1 and sin=0, so output should equal input
        assert torch.allclose(q_rot, q, atol=ATOL, rtol=RTOL), \
            f"Position 0 should be identity, max diff = {(q_rot - q).abs().max()}"
        assert torch.allclose(k_rot, k, atol=ATOL, rtol=RTOL)


# ============================================================================
# Tests: Level4 model usage patterns
# ============================================================================

class TestLevel4UsagePatterns:
    """Test usage patterns matching actual level4 models."""

    def test_llama31_pattern(self):
        """Llama31: bhsd layout, llama3 scaling, position_ids."""
        head_dim = 128
        batch, seq, heads, kv_heads = 1, 32, 32, 8
        rope_scaling = {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
        rope = RotaryEmbedding(
            head_dim, max_seq_len=32, base=500000.0,
            layout="bhsd", rope_scaling=rope_scaling,
        ).to(DEVICE)
        rope.eval()

        q = torch.randn(batch, heads, seq, head_dim, device=DEVICE)
        k = torch.randn(batch, kv_heads, seq, head_dim, device=DEVICE)
        position_ids = torch.arange(seq, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            q_rot, k_rot = rope(q, k, position_ids)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert torch.isfinite(q_rot).all()
        assert torch.isfinite(k_rot).all()

    def test_falcon_pattern(self):
        """Falcon: bhsd layout, default rope, MQA (1 kv head)."""
        head_dim = 64
        batch, seq, heads, kv_heads = 1, 32, 71, 1
        rope = RotaryEmbedding(
            head_dim, max_seq_len=32, base=10000.0, layout="bhsd",
        ).to(DEVICE)
        rope.eval()

        q = torch.randn(batch, heads, seq, head_dim, device=DEVICE)
        k = torch.randn(batch, kv_heads, seq, head_dim, device=DEVICE)
        position_ids = torch.arange(seq, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            q_rot, k_rot = rope(q, k, position_ids)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert torch.isfinite(q_rot).all()

    def test_mixtral_pattern(self):
        """Mixtral: bhsd layout, default rope."""
        head_dim = 128
        batch, seq, heads, kv_heads = 1, 32, 32, 8
        rope = RotaryEmbedding(
            head_dim, max_seq_len=32, base=1000000.0, layout="bhsd",
        ).to(DEVICE)
        rope.eval()

        q = torch.randn(batch, heads, seq, head_dim, device=DEVICE)
        k = torch.randn(batch, kv_heads, seq, head_dim, device=DEVICE)
        position_ids = torch.arange(seq, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            q_rot, k_rot = rope(q, k, position_ids)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert torch.isfinite(q_rot).all()

    def test_qwen2vl_vision_pattern(self):
        """Qwen2VL vision: apply_rotary with externally computed cos/sin, no batch dim."""
        head_dim = 128  # 1280 // 16 = 80 in real model, but 128 for test
        seq, heads = 64, 16
        rope = RotaryEmbedding(head_dim, max_seq_len=1, base=10000.0).to(DEVICE)
        rope.eval()

        q = torch.randn(seq, heads, head_dim, device=DEVICE, dtype=torch.float32)
        k = torch.randn(seq, heads, head_dim, device=DEVICE, dtype=torch.float32)

        # Simulate Qwen2VL vision cos/sin computation
        cos = torch.randn(seq, head_dim, device=DEVICE, dtype=torch.float32)
        sin = torch.randn(seq, head_dim, device=DEVICE, dtype=torch.float32)
        cos = cos.unsqueeze(-2)  # (seq, 1, head_dim)
        sin = sin.unsqueeze(-2)

        with torch.no_grad():
            q_rot, k_rot = rope.apply_rotary(q, k, cos, sin)

        q_ref, k_ref = ref_apply_rotary_half(q, k, cos, sin)

        assert torch.allclose(q_rot, q_ref, atol=ATOL, rtol=RTOL)
        assert torch.allclose(k_rot, k_ref, atol=ATOL, rtol=RTOL)

    def test_qwen2vl_llm_pattern(self):
        """Qwen2VL LLM: apply_rotary with M-RoPE assembled cos/sin, bhsd layout."""
        head_dim = 128
        batch, seq, heads = 1, 32, 28
        rope = RotaryEmbedding(head_dim, max_seq_len=1, base=1000000.0).to(DEVICE)
        rope.eval()

        q = torch.randn(batch, heads, seq, head_dim, device=DEVICE)
        k = torch.randn(batch, heads, seq, head_dim, device=DEVICE)

        # Simulate M-RoPE assembled cos/sin (already interleaved)
        cos = torch.randn(1, batch, 1, seq, head_dim, device=DEVICE)
        sin = torch.randn(1, batch, 1, seq, head_dim, device=DEVICE)

        with torch.no_grad():
            q_rot, k_rot = rope.apply_rotary(q, k, cos, sin)

        q_ref, k_ref = ref_apply_rotary_half(q, k, cos, sin)

        assert torch.allclose(q_rot, q_ref, atol=ATOL, rtol=RTOL)
        assert torch.allclose(k_rot, k_ref, atol=ATOL, rtol=RTOL)

    def test_eagle3_pattern(self):
        """EAGLE3: bhsd layout, default rope, no position_ids."""
        head_dim = 128
        batch, seq, heads, kv_heads = 1, 16, 32, 8
        rope = RotaryEmbedding(
            head_dim, max_seq_len=16, base=500000.0, layout="bhsd",
        ).to(DEVICE)
        rope.eval()

        q = torch.randn(batch, heads, seq, head_dim, device=DEVICE)
        k = torch.randn(batch, kv_heads, seq, head_dim, device=DEVICE)

        with torch.no_grad():
            q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert torch.isfinite(q_rot).all()
        assert not torch.allclose(q_rot, q, atol=1e-3)


# ============================================================================
# Tests: Cache extension
# ============================================================================

class TestCacheExtension:
    """Test that cos/sin cache extends correctly."""

    def test_cache_extends_for_longer_seq(self):
        """Cache should auto-extend when seq_len exceeds max_seq_len."""
        head_dim = 64
        rope = RotaryEmbedding(head_dim, max_seq_len=8, base=10000.0, layout="bhsd").to(DEVICE)
        rope.eval()

        # First call within cache
        q1 = torch.randn(1, 2, 8, head_dim, device=DEVICE)
        k1 = torch.randn(1, 2, 8, head_dim, device=DEVICE)
        with torch.no_grad():
            q_rot1, k_rot1 = rope(q1, k1)
        assert q_rot1.shape == q1.shape

        # Second call exceeding cache
        q2 = torch.randn(1, 2, 32, head_dim, device=DEVICE)
        k2 = torch.randn(1, 2, 32, head_dim, device=DEVICE)
        with torch.no_grad():
            q_rot2, k_rot2 = rope(q2, k2)
        assert q_rot2.shape == q2.shape
        assert rope.max_seq_len >= 32

    def test_cache_extends_for_large_position_ids(self):
        """Cache should extend when position_ids exceed max_seq_len."""
        head_dim = 64
        rope = RotaryEmbedding(head_dim, max_seq_len=8, base=10000.0, layout="bhsd").to(DEVICE)
        rope.eval()

        q = torch.randn(1, 2, 4, head_dim, device=DEVICE)
        k = torch.randn(1, 2, 4, head_dim, device=DEVICE)
        position_ids = torch.tensor([[100, 200, 300, 400]], device=DEVICE)

        with torch.no_grad():
            q_rot, k_rot = rope(q, k, position_ids)

        assert q_rot.shape == q.shape
        assert rope.max_seq_len >= 401


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
