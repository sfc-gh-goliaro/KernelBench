"""
Unit tests for KernelBench level1 operators.

These tests validate expected mathematical properties of operator implementations.
Each test loads the Model class from the operator file and validates that it
produces outputs satisfying known mathematical invariants.

Tests are designed to detect reward hacking by RL-trained LLMs that generate
CUDA kernels. Common hacks include: returning zeros, constants, identity,
or ignoring some inputs.
"""

import pytest
import torch
import torch.nn as nn
import importlib.util
import os
import sys
import glob

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
ATOL = 1e-4
RTOL = 1e-4

# Base path for level1 operators
LEVEL1_PATH = os.path.join(os.path.dirname(__file__), '..', 'KernelBench', 'level1')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'KernelBench'))


def load_operator_module(category: str, filename: str):
    """Dynamically load an operator module."""
    filepath = os.path.join(LEVEL1_PATH, category, filename)
    spec = importlib.util.spec_from_file_location(f"{category}.{filename[:-3]}", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_model_and_inputs(module, param_idx=0, device=DEVICE, dtype=DTYPE):
    """Create a model instance and its inputs from an operator module."""
    # Get initialization inputs
    init_inputs = module.get_init_inputs(param_idx=param_idx)

    # Create model
    model = module.Model(*init_inputs)
    model = model.to(device)
    if hasattr(model, 'eval'):
        model.eval()

    # Get forward inputs - use the module's supported distributions
    if hasattr(module, 'SUPPORTED_DISTRIBUTIONS') and module.SUPPORTED_DISTRIBUTIONS:
        dist_name = module.SUPPORTED_DISTRIBUTIONS[0]
        inputs = module.get_inputs(param_idx=param_idx, dist_name=dist_name, dtype=dtype, device=device)
    else:
        inputs = module.get_inputs(param_idx=param_idx, dtype=dtype, device=device)

    return model, inputs


def get_operators(category: str):
    """Get all .py operator files in a category."""
    pattern = os.path.join(LEVEL1_PATH, category, "*.py")
    files = glob.glob(pattern)
    return [os.path.basename(f) for f in files if not f.endswith("__init__.py")]


# =============================================================================
# ACTIVATION TESTS
# =============================================================================

class TestActivations:
    """Tests for activation operators."""

    # --- ReLU ---
    def test_relu_output_non_negative(self):
        """ReLU output should be non-negative."""
        module = load_operator_module("activations", "1_ReLU.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert (output >= 0).all(), "ReLU output should be non-negative"

    def test_relu_preserves_positive(self):
        """ReLU should preserve positive values."""
        module = load_operator_module("activations", "1_ReLU.py")
        model, _ = create_model_and_inputs(module)
        x = torch.abs(torch.randn(32, 64, device=DEVICE, dtype=DTYPE)) + 0.1
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, x, atol=ATOL), "ReLU should preserve positive values"

    def test_relu_zeros_negative(self):
        """ReLU should zero out negative values."""
        module = load_operator_module("activations", "1_ReLU.py")
        model, _ = create_model_and_inputs(module)
        x = -torch.abs(torch.randn(32, 64, device=DEVICE, dtype=DTYPE)) - 0.1
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, torch.zeros_like(output), atol=ATOL), "ReLU should zero negative"

    # --- LeakyReLU ---
    def test_leaky_relu_positive_preserved(self):
        """LeakyReLU should preserve positive values."""
        module = load_operator_module("activations", "2_LeakyReLU.py")
        model, _ = create_model_and_inputs(module)
        x = torch.abs(torch.randn(32, 64, device=DEVICE, dtype=DTYPE)) + 0.1
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, x, atol=ATOL), "LeakyReLU should preserve positive"

    def test_leaky_relu_negative_scaled(self):
        """LeakyReLU should scale negative values."""
        module = load_operator_module("activations", "2_LeakyReLU.py")
        model, _ = create_model_and_inputs(module)
        x = -torch.abs(torch.randn(32, 64, device=DEVICE, dtype=DTYPE)) - 0.1
        with torch.no_grad():
            output = model(x)
        # Output should be negative but smaller in magnitude than input
        assert (output < 0).all(), "LeakyReLU of negative should be negative"
        assert (output.abs() < x.abs()).all(), "LeakyReLU should reduce magnitude of negative"

    # --- Sigmoid ---
    def test_sigmoid_bounded(self):
        """Sigmoid output should be in (0, 1)."""
        module = load_operator_module("activations", "3_Sigmoid.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert (output > 0).all() and (output < 1).all(), "Sigmoid should be in (0, 1)"

    def test_sigmoid_zero_gives_half(self):
        """Sigmoid(0) should equal 0.5."""
        module = load_operator_module("activations", "3_Sigmoid.py")
        model, _ = create_model_and_inputs(module)
        x = torch.zeros(32, 64, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, torch.full_like(output, 0.5), atol=ATOL)

    def test_sigmoid_symmetry(self):
        """Sigmoid should satisfy: sigmoid(-x) = 1 - sigmoid(x)."""
        module = load_operator_module("activations", "3_Sigmoid.py")
        model, _ = create_model_and_inputs(module)
        x = torch.randn(32, 64, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            output_pos = model(x)
            output_neg = model(-x)
        assert torch.allclose(output_neg, 1 - output_pos, atol=ATOL)

    # --- Tanh ---
    def test_tanh_bounded(self):
        """Tanh output should be in (-1, 1)."""
        module = load_operator_module("activations", "4_Tanh.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert (output > -1).all() and (output < 1).all(), "Tanh should be in (-1, 1)"

    def test_tanh_zero_gives_zero(self):
        """Tanh(0) should equal 0."""
        module = load_operator_module("activations", "4_Tanh.py")
        model, _ = create_model_and_inputs(module)
        x = torch.zeros(32, 64, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, torch.zeros_like(output), atol=ATOL)

    def test_tanh_odd_function(self):
        """Tanh should be odd: tanh(-x) = -tanh(x)."""
        module = load_operator_module("activations", "4_Tanh.py")
        model, _ = create_model_and_inputs(module)
        x = torch.randn(32, 64, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            output_pos = model(x)
            output_neg = model(-x)
        assert torch.allclose(output_neg, -output_pos, atol=ATOL)

    # --- Softmax ---
    def test_softmax_sums_to_one(self):
        """Softmax output should sum to 1 along reduction dimension."""
        module = load_operator_module("activations", "5_Softmax.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        row_sums = output.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=ATOL)

    def test_softmax_non_negative(self):
        """Softmax output should be non-negative."""
        module = load_operator_module("activations", "5_Softmax.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert (output >= 0).all(), "Softmax should be non-negative"

    def test_softmax_preserves_order(self):
        """Softmax should preserve relative ordering."""
        module = load_operator_module("activations", "5_Softmax.py")
        model, _ = create_model_and_inputs(module)
        x = torch.arange(64, device=DEVICE, dtype=DTYPE).unsqueeze(0)
        with torch.no_grad():
            output = model(x)
        # Output should be monotonically increasing
        diffs = output[0, 1:] - output[0, :-1]
        assert (diffs >= -1e-6).all(), "Softmax should preserve ordering"

    # --- LogSoftmax ---
    def test_logsoftmax_exp_sums_to_one(self):
        """exp(LogSoftmax(x)) should sum to 1."""
        module = load_operator_module("activations", "6_LogSoftmax.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        exp_sums = torch.exp(output).sum(dim=1)
        assert torch.allclose(exp_sums, torch.ones_like(exp_sums), atol=ATOL)

    def test_logsoftmax_non_positive(self):
        """LogSoftmax output should be non-positive."""
        module = load_operator_module("activations", "6_LogSoftmax.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert (output <= 0).all(), "LogSoftmax should be non-positive"

    # --- Swish ---
    def test_swish_zero_gives_zero(self):
        """Swish(0) should equal 0."""
        module = load_operator_module("activations", "7_Swish.py")
        model, _ = create_model_and_inputs(module)
        x = torch.zeros(32, 64, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, torch.zeros_like(output), atol=ATOL)

    def test_swish_positive_for_large_positive(self):
        """Swish should be positive for large positive input."""
        module = load_operator_module("activations", "7_Swish.py")
        model, _ = create_model_and_inputs(module)
        x = torch.ones(32, 64, device=DEVICE, dtype=DTYPE) * 5.0
        with torch.no_grad():
            output = model(x)
        assert (output > 4.9).all(), "Swish of large positive should be ~ x"

    # --- GELU ---
    def test_gelu_zero_gives_zero(self):
        """GELU(0) should equal 0."""
        module = load_operator_module("activations", "8_GELU.py")
        model, _ = create_model_and_inputs(module)
        x = torch.zeros(32, 64, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, torch.zeros_like(output), atol=ATOL)

    def test_gelu_large_positive_preserved(self):
        """GELU should approximately preserve large positive values."""
        module = load_operator_module("activations", "8_GELU.py")
        model, _ = create_model_and_inputs(module)
        x = torch.ones(32, 64, device=DEVICE, dtype=DTYPE) * 5.0
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, x, atol=0.1), "GELU of large positive should be ~ x"

    # --- SELU ---
    def test_selu_zero_gives_zero(self):
        """SELU(0) should equal 0."""
        module = load_operator_module("activations", "9_SELU.py")
        model, _ = create_model_and_inputs(module)
        x = torch.zeros(32, 64, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, torch.zeros_like(output), atol=ATOL)

    def test_selu_positive_scaled(self):
        """SELU of positive should be scaled by lambda."""
        module = load_operator_module("activations", "9_SELU.py")
        model, _ = create_model_and_inputs(module)
        x = torch.ones(32, 64, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            output = model(x)
        # SELU scale is approximately 1.0507
        assert torch.allclose(output, x * 1.0507, atol=0.01)

    # --- HardSigmoid ---
    def test_hardsigmoid_bounded(self):
        """HardSigmoid output should be in [0, 1]."""
        module = load_operator_module("activations", "10_HardSigmoid.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert (output >= 0).all() and (output <= 1).all()

    def test_hardsigmoid_zero_gives_half(self):
        """HardSigmoid(0) should equal 0.5."""
        module = load_operator_module("activations", "10_HardSigmoid.py")
        model, _ = create_model_and_inputs(module)
        x = torch.zeros(32, 64, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, torch.full_like(output, 0.5), atol=ATOL)

    # --- Softplus ---
    def test_softplus_non_negative(self):
        """Softplus output should be non-negative."""
        module = load_operator_module("activations", "11_Softplus.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert (output >= 0).all(), "Softplus should be non-negative"

    def test_softplus_large_positive_approx_identity(self):
        """Softplus should approximate identity for large positive."""
        module = load_operator_module("activations", "11_Softplus.py")
        model, _ = create_model_and_inputs(module)
        x = torch.ones(32, 64, device=DEVICE, dtype=DTYPE) * 10.0
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, x, atol=0.01)

    # --- Softsign ---
    def test_softsign_bounded(self):
        """Softsign output should be in (-1, 1)."""
        module = load_operator_module("activations", "12_Softsign.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert (output > -1).all() and (output < 1).all()

    def test_softsign_zero_gives_zero(self):
        """Softsign(0) should equal 0."""
        module = load_operator_module("activations", "12_Softsign.py")
        model, _ = create_model_and_inputs(module)
        x = torch.zeros(32, 64, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, torch.zeros_like(output), atol=ATOL)

    # --- ELU ---
    def test_elu_positive_preserved(self):
        """ELU should preserve positive values."""
        module = load_operator_module("activations", "13_ELU.py")
        model, _ = create_model_and_inputs(module)
        x = torch.abs(torch.randn(32, 64, device=DEVICE, dtype=DTYPE)) + 0.1
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, x, atol=ATOL)

    def test_elu_negative_bounded(self):
        """ELU of negative should be bounded below by -alpha."""
        module = load_operator_module("activations", "13_ELU.py")
        model, _ = create_model_and_inputs(module)
        x = -torch.abs(torch.randn(32, 64, device=DEVICE, dtype=DTYPE)) - 0.1
        with torch.no_grad():
            output = model(x)
        # ELU with default alpha=1.0 is bounded below by -1
        assert (output > -1.1).all(), "ELU should be bounded below"
        assert (output < 0).all(), "ELU of negative should be negative"

    # --- HardTanh ---
    def test_hardtanh_bounded(self):
        """HardTanh output should be in [-1, 1]."""
        module = load_operator_module("activations", "14_HardTanh.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert (output >= -1).all() and (output <= 1).all()

    def test_hardtanh_saturates(self):
        """HardTanh should saturate at boundaries."""
        module = load_operator_module("activations", "14_HardTanh.py")
        model, _ = create_model_and_inputs(module)
        x = torch.tensor([[-10.0, 10.0, 0.0]], device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            output = model(x)
        assert output[0, 0] == -1.0 and output[0, 1] == 1.0 and output[0, 2] == 0.0

    # --- HardSwish ---
    def test_hardswish_zero_gives_zero(self):
        """HardSwish(0) should equal 0."""
        module = load_operator_module("activations", "15_HardSwish.py")
        model, _ = create_model_and_inputs(module)
        x = torch.zeros(32, 64, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, torch.zeros_like(output), atol=ATOL)

    def test_hardswish_large_positive(self):
        """HardSwish should approach identity for large positive."""
        module = load_operator_module("activations", "15_HardSwish.py")
        model, _ = create_model_and_inputs(module)
        x = torch.ones(32, 64, device=DEVICE, dtype=DTYPE) * 10.0
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, x, atol=0.1)

    # --- SiluAndMul ---
    def test_siluandmul_output_shape(self):
        """SiluAndMul should halve the last dimension."""
        module = load_operator_module("activations", "16_SiluAndMul.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert output.shape[-1] == inputs[0].shape[-1] // 2

    def test_siluandmul_zero_input_zero_output(self):
        """SiluAndMul of zeros should be zero."""
        module = load_operator_module("activations", "16_SiluAndMul.py")
        model, inputs = create_model_and_inputs(module)
        x = torch.zeros_like(inputs[0])
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, torch.zeros_like(output), atol=ATOL)

    # --- GeluAndMul ---
    def test_geluandmul_output_shape(self):
        """GeluAndMul should halve the last dimension."""
        module = load_operator_module("activations", "17_GeluAndMul.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert output.shape[-1] == inputs[0].shape[-1] // 2

    def test_geluandmul_zero_input_zero_output(self):
        """GeluAndMul of zeros should be zero."""
        module = load_operator_module("activations", "17_GeluAndMul.py")
        model, inputs = create_model_and_inputs(module)
        x = torch.zeros_like(inputs[0])
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, torch.zeros_like(output), atol=ATOL)


# =============================================================================
# NORMALIZATION TESTS
# =============================================================================

class TestNormalization:
    """Tests for normalization operators."""

    # --- BatchNorm ---
    def test_batchnorm_shape_preserved(self):
        """BatchNorm should preserve input shape."""
        module = load_operator_module("normalization", "1_BatchNorm.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert output.shape == inputs[0].shape

    def test_batchnorm_finite_output(self):
        """BatchNorm should produce finite output."""
        module = load_operator_module("normalization", "1_BatchNorm.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert torch.isfinite(output).all()

    # --- InstanceNorm ---
    def test_instancenorm_shape_preserved(self):
        """InstanceNorm should preserve input shape."""
        module = load_operator_module("normalization", "2_InstanceNorm.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert output.shape == inputs[0].shape

    def test_instancenorm_finite(self):
        """InstanceNorm output should be finite."""
        module = load_operator_module("normalization", "2_InstanceNorm.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert torch.isfinite(output).all()

    # --- GroupNorm ---
    def test_groupnorm_shape_preserved(self):
        """GroupNorm should preserve input shape."""
        module = load_operator_module("normalization", "3_GroupNorm.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert output.shape == inputs[0].shape

    def test_groupnorm_finite_output(self):
        """GroupNorm output should be finite."""
        module = load_operator_module("normalization", "3_GroupNorm.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert torch.isfinite(output).all()

    # --- RMSNorm ---
    def test_rmsnorm_shape_preserved(self):
        """RMSNorm should preserve input shape."""
        module = load_operator_module("normalization", "4_RMSNorm.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert output.shape == inputs[0].shape

    def test_rmsnorm_finite(self):
        """RMSNorm output should be finite."""
        module = load_operator_module("normalization", "4_RMSNorm.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert torch.isfinite(output).all()


# =============================================================================
# MATMUL TESTS
# =============================================================================

class TestMatMul:
    """Tests for matrix multiplication operators."""

    # --- MatMul ---
    def test_matmul_output_shape(self):
        """MatMul output shape should be (M, N)."""
        module = load_operator_module("matmul", "1_MatMul.py")
        model, inputs = create_model_and_inputs(module)
        A, B = inputs
        with torch.no_grad():
            output = model(A, B)
        assert output.shape == (A.shape[0], B.shape[1])

    def test_matmul_identity(self):
        """MatMul with identity should return original."""
        module = load_operator_module("matmul", "1_MatMul.py")
        model, _ = create_model_and_inputs(module)
        n = 64
        A = torch.randn(n, n, device=DEVICE, dtype=DTYPE)
        I = torch.eye(n, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            output = model(A, I)
        assert torch.allclose(output, A, atol=ATOL)

    def test_matmul_not_zero(self):
        """MatMul of non-zero matrices should not be zero."""
        module = load_operator_module("matmul", "1_MatMul.py")
        model, _ = create_model_and_inputs(module)
        A = torch.ones(32, 64, device=DEVICE, dtype=DTYPE)
        B = torch.ones(64, 48, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            output = model(A, B)
        expected = torch.full((32, 48), 64.0, device=DEVICE, dtype=DTYPE)
        assert torch.allclose(output, expected, atol=ATOL)

    # --- BatchedMatMul ---
    def test_batched_matmul_output_shape(self):
        """BatchedMatMul output shape should be (B, M, N)."""
        module = load_operator_module("matmul", "2_BatchedMatMul.py")
        model, inputs = create_model_and_inputs(module)
        A, B = inputs
        with torch.no_grad():
            output = model(A, B)
        assert output.shape == (A.shape[0], A.shape[1], B.shape[2])

    def test_batched_matmul_batch_independence(self):
        """Each batch should be computed independently."""
        module = load_operator_module("matmul", "2_BatchedMatMul.py")
        model, _ = create_model_and_inputs(module)
        B, M, K, N = 4, 8, 16, 12
        A = torch.randn(B, M, K, device=DEVICE, dtype=DTYPE)
        B_mat = torch.randn(B, K, N, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            output = model(A, B_mat)
        # Verify each batch
        for i in range(B):
            expected = torch.matmul(A[i], B_mat[i])
            assert torch.allclose(output[i], expected, atol=ATOL)

    # --- DiagonalMatMul ---
    def test_diagonal_matmul_shape(self):
        """DiagonalMatMul should produce (N, M) output."""
        module = load_operator_module("matmul", "3_DiagonalMatMul.py")
        model, inputs = create_model_and_inputs(module)
        A, B = inputs
        with torch.no_grad():
            output = model(A, B)
        assert output.shape == B.shape

    def test_diagonal_matmul_scales_rows(self):
        """DiagonalMatMul should scale each row by diagonal element."""
        module = load_operator_module("matmul", "3_DiagonalMatMul.py")
        model, _ = create_model_and_inputs(module)
        N, M = 8, 16
        diag = torch.arange(1, N + 1, device=DEVICE, dtype=DTYPE)
        B = torch.ones(N, M, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            output = model(diag, B)
        for i in range(N):
            assert torch.allclose(output[i], torch.full((M,), float(i + 1), device=DEVICE, dtype=DTYPE), atol=ATOL)

    # --- Transposed variants ---
    def test_transposed_a_output_shape(self):
        """TransposedA should produce correct output shape."""
        module = load_operator_module("matmul", "7_TransposedA.py")
        model, inputs = create_model_and_inputs(module)
        A, B = inputs
        with torch.no_grad():
            output = model(A, B)
        assert output.shape[0] == A.shape[1]  # A^T has shape (K, M) -> output M

    def test_transposed_b_output_shape(self):
        """TransposedB should produce correct output shape."""
        module = load_operator_module("matmul", "8_TransposedB.py")
        model, inputs = create_model_and_inputs(module)
        A, B = inputs
        with torch.no_grad():
            output = model(A, B)
        assert output.shape[1] == B.shape[0]  # B^T has shape (N, K) -> output N

    def test_transposed_both_output_shape(self):
        """TransposedBoth should produce correct output shape."""
        module = load_operator_module("matmul", "9_TransposedBoth.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert len(output.shape) == 2


# =============================================================================
# POOLING TESTS
# =============================================================================

class TestPooling:
    """Tests for pooling operators."""

    # --- MaxPool1d ---
    def test_maxpool1d_reduces_size(self):
        """MaxPool1d should reduce the temporal dimension."""
        module = load_operator_module("pooling", "1_MaxPool1d.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert output.shape[-1] < inputs[0].shape[-1]

    def test_maxpool1d_bounded(self):
        """MaxPool1d output should be bounded by input range."""
        module = load_operator_module("pooling", "1_MaxPool1d.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert output.max() <= inputs[0].max() + ATOL
        assert output.min() >= inputs[0].min() - ATOL

    # --- MaxPool2d ---
    def test_maxpool2d_reduces_spatial(self):
        """MaxPool2d should reduce spatial dimensions."""
        module = load_operator_module("pooling", "2_MaxPool2d.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert output.shape[2] <= inputs[0].shape[2]
        assert output.shape[3] <= inputs[0].shape[3]

    def test_maxpool2d_bounded(self):
        """MaxPool2d output should be bounded by input range."""
        module = load_operator_module("pooling", "2_MaxPool2d.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert output.max() <= inputs[0].max() + ATOL

    # --- MaxPool3d ---
    def test_maxpool3d_reduces_spatial(self):
        """MaxPool3d should reduce spatial dimensions."""
        module = load_operator_module("pooling", "3_MaxPool3d.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert output.numel() <= inputs[0].numel()

    def test_maxpool3d_bounded(self):
        """MaxPool3d output should be bounded by input range."""
        module = load_operator_module("pooling", "3_MaxPool3d.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert output.max() <= inputs[0].max() + ATOL

    # --- AdaptiveAvgPool2d ---
    def test_adaptiveavgpool2d_output_size(self):
        """AdaptiveAvgPool2d should produce target output size."""
        module = load_operator_module("pooling", "7_AdaptiveAvgPool2d.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        # Should reduce to target size
        assert output.shape[0] == inputs[0].shape[0]
        assert output.shape[1] == inputs[0].shape[1]

    def test_adaptiveavgpool2d_constant_input(self):
        """AdaptiveAvgPool2d of constant should preserve constant."""
        module = load_operator_module("pooling", "7_AdaptiveAvgPool2d.py")
        model, inputs = create_model_and_inputs(module)
        x = torch.full_like(inputs[0], 2.71)
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, torch.full_like(output, 2.71), atol=ATOL)

    # --- GlobalAveragePooling ---
    def test_global_avgpool_output_shape(self):
        """GlobalAveragePooling should reduce to (B, C)."""
        module = load_operator_module("pooling", "8_GlobalAveragePooling.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert len(output.shape) == 2
        assert output.shape[0] == inputs[0].shape[0]
        assert output.shape[1] == inputs[0].shape[1]

    def test_global_avgpool_constant_input(self):
        """GlobalAveragePooling of constant should preserve constant."""
        module = load_operator_module("pooling", "8_GlobalAveragePooling.py")
        model, inputs = create_model_and_inputs(module)
        x = torch.full_like(inputs[0], 1.41)
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, torch.full_like(output, 1.41), atol=ATOL)

    # --- MeanPooling ---
    def test_meanpooling_output_shape(self):
        """MeanPooling should reduce sequence dimension."""
        module = load_operator_module("pooling", "9_MeanPooling.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert output.numel() < inputs[0].numel()

    def test_meanpooling_constant_input(self):
        """MeanPooling of constant should preserve constant."""
        module = load_operator_module("pooling", "9_MeanPooling.py")
        model, inputs = create_model_and_inputs(module)
        x = torch.full_like(inputs[0], 1.73)
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, torch.full_like(output, 1.73), atol=ATOL)

    # --- LastTokenPooling ---
    def test_lasttokenpooling_output_shape(self):
        """LastTokenPooling should reduce sequence dimension."""
        module = load_operator_module("pooling", "10_LastTokenPooling.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert len(output.shape) == len(inputs[0].shape) - 1 or output.shape[1] == 1


# =============================================================================
# REDUCTION TESTS
# =============================================================================

class TestReductions:
    """Tests for reduction operators."""

    # --- Sum ---
    def test_sum_shape(self):
        """Sum should reduce along specified dimension."""
        module = load_operator_module("reductions", "1_Sum.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        # One dimension should be 1 (keepdim=True)
        assert 1 in output.shape

    def test_sum_zeros_give_zero(self):
        """Sum of zeros should be zero."""
        module = load_operator_module("reductions", "1_Sum.py")
        model, inputs = create_model_and_inputs(module)
        x = torch.zeros_like(inputs[0])
        with torch.no_grad():
            output = model(x)
        assert torch.allclose(output, torch.zeros_like(output), atol=ATOL)

    def test_sum_ones(self):
        """Sum of ones should equal dimension size."""
        module = load_operator_module("reductions", "1_Sum.py")
        model, inputs = create_model_and_inputs(module)
        x = torch.ones_like(inputs[0])
        with torch.no_grad():
            output = model(x)
        # Sum should be positive and significant
        assert output.mean() > 0.5

    # --- MinMax ---
    def test_minmax_bounded_by_input(self):
        """MinMax output should be within input range."""
        module = load_operator_module("reductions", "2_MinMax.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert output.max() <= inputs[0].max() + ATOL
        assert output.min() >= inputs[0].min() - ATOL

    def test_minmax_reduces_dimension(self):
        """MinMax should reduce one dimension."""
        module = load_operator_module("reductions", "2_MinMax.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert output.numel() < inputs[0].numel()

    # --- ArgMinMax ---
    def test_argminmax_valid_indices(self):
        """ArgMinMax should return valid indices."""
        module = load_operator_module("reductions", "3_ArgMinMax.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        # Indices should be non-negative
        assert (output >= 0).all()

    def test_argminmax_selects_extremum(self):
        """ArgMinMax should select index of extremum."""
        module = load_operator_module("reductions", "3_ArgMinMax.py")
        model, _ = create_model_and_inputs(module)
        # Create tensor with known argmax
        x = torch.zeros(4, 10, 8, device=DEVICE, dtype=DTYPE)
        x[:, 5, :] = 1.0  # Max at position 5
        with torch.no_grad():
            output = model(x)
        assert (output == 5).all()


# =============================================================================
# ATTENTION TESTS
# =============================================================================

class TestAttention:
    """Tests for attention operators."""

    # --- ScaledDotProductAttention ---
    def test_sdpa_output_shape(self):
        """Scaled dot-product attention should preserve shape."""
        module = load_operator_module("attention", "1_ScaledDotProductAttention.py")
        model, inputs = create_model_and_inputs(module)
        Q, K, V = inputs
        with torch.no_grad():
            output = model(Q, K, V)
        assert output.shape == V.shape

    def test_sdpa_not_zero(self):
        """Attention output should not be all zeros."""
        module = load_operator_module("attention", "1_ScaledDotProductAttention.py")
        model, inputs = create_model_and_inputs(module)
        with torch.no_grad():
            output = model(*inputs)
        assert output.abs().mean() > ATOL

    def test_sdpa_uniform_qk_returns_avg_v(self):
        """Uniform Q,K should return average of V."""
        module = load_operator_module("attention", "1_ScaledDotProductAttention.py")
        model, _ = create_model_and_inputs(module)
        B, H, S, D = 2, 4, 8, 16
        Q = torch.ones(B, H, S, D, device=DEVICE, dtype=DTYPE)
        K = torch.ones(B, H, S, D, device=DEVICE, dtype=DTYPE)
        V = torch.ones(B, H, S, D, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            output = model(Q, K, V)
        assert torch.allclose(output, V, atol=0.01)


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
