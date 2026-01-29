"""
FP8 Block-wise GEMM

Implements FP8 (E4M3) matrix multiplication with block-wise quantization,
matching the implementation used by DeepSeek-R1/V3 models.

Features:
- Block-wise weight quantization (typically 128x128 blocks)
- Dynamic activation quantization per 128-element groups
- Triton-accelerated kernels for efficient FP8 matmul
- Accumulation in FP32 for numerical stability

Based on transformers.integrations.finegrained_fp8 and DeepSeek's inference kernel.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# FP8 constants
try:
    _FP8_DTYPE = torch.float8_e4m3fn
    _FP8_MIN = torch.finfo(_FP8_DTYPE).min
    _FP8_MAX = torch.finfo(_FP8_DTYPE).max
except AttributeError:
    _FP8_DTYPE = None
    _FP8_MIN, _FP8_MAX = -448, 448


# ============================================================================
# Triton Kernels
# ============================================================================

if HAS_TRITON:
    @triton.jit
    def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
        """Quantize activations to FP8 with per-block scaling."""
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs).to(tl.float32)
        s = tl.max(tl.abs(x)) / 448.0
        # Avoid division by zero
        s = tl.where(s > 0, s, 1.0)
        y = x / s
        y = y.to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + offs, y)
        tl.store(s_ptr + pid, s)

    @triton.jit
    def _w8a8_block_fp8_matmul(
        # Pointers to inputs and output
        A, B, C, As, Bs,
        # Shape for matmul
        M, N, K,
        # Block size for block-wise quantization
        group_n, group_k,
        # Stride for inputs and output
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_As_m, stride_As_k,
        stride_Bs_k, stride_Bs_n,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """Block-wise FP8 matmul with per-block scaling."""
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        As_ptrs = As + offs_am * stride_As_m
        offs_bsn = offs_bn // group_n
        Bs_ptrs = Bs + offs_bsn * stride_Bs_n

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

            k_start = k * BLOCK_SIZE_K
            offs_ks = k_start // group_k
            a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
            b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if C.dtype.element_ty == tl.bfloat16:
            c = accumulator.to(tl.bfloat16)
        elif C.dtype.element_ty == tl.float16:
            c = accumulator.to(tl.float16)
        else:
            c = accumulator.to(tl.float32)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    @triton.jit
    def _w8a8_block_fp8_matmul_per_tensor(
        A, B, C, As, Bs,
        M, N, K,
        group_n, group_k,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """Per-tensor quantized FP8 matmul."""
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        scale_a = tl.load(As)
        scale_b = tl.load(Bs)
        
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator += tl.dot(a, b) * scale_a * scale_b
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if C.dtype.element_ty == tl.bfloat16:
            c = accumulator.to(tl.bfloat16)
        elif C.dtype.element_ty == tl.float16:
            c = accumulator.to(tl.float16)
        else:
            c = accumulator.to(tl.float32)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


# ============================================================================
# Python Functions
# ============================================================================

def act_quant(x: torch.Tensor, block_size: int = 128) -> tuple:
    """
    Quantize activations to FP8 with per-block scaling.
    
    Args:
        x: Input tensor, must be contiguous and last dim divisible by block_size
        block_size: Block size for quantization (default 128)
        
    Returns:
        Tuple of (quantized tensor, scale tensor)
    """
    assert x.is_contiguous(), "Input must be contiguous"
    assert x.shape[-1] % block_size == 0, f"Last dim {x.shape[-1]} must be divisible by {block_size}"
    
    if HAS_TRITON:
        y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)

        def grid(meta):
            return (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)

        act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
        return y, s
    else:
        # Fallback: manual implementation
        shape = x.shape
        x_flat = x.view(-1, shape[-1])
        n_blocks = shape[-1] // block_size
        x_blocks = x_flat.view(-1, n_blocks, block_size)
        
        # Compute per-block scales
        max_abs = x_blocks.abs().amax(dim=-1, keepdim=True)
        scales = max_abs / 448.0
        scales = torch.where(scales > 0, scales, torch.ones_like(scales))
        
        # Quantize
        x_scaled = x_blocks / scales
        x_quantized = x_scaled.clamp(-448, 448).to(torch.float8_e4m3fn)
        
        y = x_quantized.view(shape)
        s = scales.squeeze(-1).view(*shape[:-1], n_blocks)
        return y, s


def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: tuple = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Block-wise FP8 matrix multiplication.
    
    Args:
        A: Quantized input tensor (FP8)
        B: Quantized weight tensor (FP8), shape (out_features, in_features)
        As: Per-block scales for A
        Bs: Per-block scales for B (weight_scale_inv)
        block_size: (block_n, block_k) for weight blocks
        output_dtype: Output dtype (default bfloat16)
        
    Returns:
        Output tensor in output_dtype
    """
    if block_size is None:
        block_n, block_k = 128, 128
    else:
        block_n, block_k = block_size[0], block_size[1]

    # For per-tensor quantization, use 128x128 blocks
    if block_n == B.shape[-2] and block_k == B.shape[-1]:
        block_n = 128
        block_k = 128

    assert A.shape[-1] == B.shape[-1], f"Dimension mismatch: A={A.shape}, B={B.shape}"

    if As.numel() != 1:
        assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    assert B.ndim == 2 and B.is_contiguous()

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    if not HAS_TRITON:
        # Fallback: dequantize and use regular matmul
        A_deq = A.to(torch.float32)
        B_deq = B.to(torch.float32)
        
        # Apply scales (simplified - full implementation would tile properly)
        if As.numel() > 1:
            As_expanded = As.repeat_interleave(block_k, dim=-1)
            As_expanded = As_expanded[..., :A.shape[-1]]
            A_deq = A_deq * As_expanded.unsqueeze(-2) if A_deq.ndim > 2 else A_deq * As_expanded
        
        if Bs.numel() > 1:
            Bs_expanded = Bs.repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1)
            Bs_expanded = Bs_expanded[:B.shape[0], :B.shape[1]]
            B_deq = B_deq * Bs_expanded
        
        C = F.linear(A_deq.to(output_dtype), B_deq.to(output_dtype))
        return C

    BLOCK_SIZE_M = 128
    if M < BLOCK_SIZE_M:
        BLOCK_SIZE_M = triton.next_power_of_2(M)
        BLOCK_SIZE_M = max(BLOCK_SIZE_M, 16)
    BLOCK_SIZE_K = block_k
    BLOCK_SIZE_N = block_n

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    if As.numel() == 1 and Bs.numel() == 1:
        _w8a8_block_fp8_matmul_per_tensor[grid](
            A, B, C, As, Bs,
            M, N, K,
            block_n, block_k,
            A.stride(-2), A.stride(-1),
            B.stride(1), B.stride(0),
            C.stride(-2), C.stride(-1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=8,
        )
    else:
        _w8a8_block_fp8_matmul[grid](
            A, B, C, As, Bs,
            M, N, K,
            block_n, block_k,
            A.stride(-2), A.stride(-1),
            B.stride(1), B.stride(0),
            C.stride(-2), C.stride(-1),
            As.stride(-2), As.stride(-1),
            Bs.stride(1), Bs.stride(0),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=8,
        )

    return C


# ============================================================================
# FP8 Linear Module
# ============================================================================

class Model(nn.Module):
    """
    FP8 Linear Layer with Block-wise Quantization
    
    Implements nn.Linear equivalent using FP8 weights with block-wise scaling,
    matching the format used by DeepSeek-R1/V3 models.
    
    Features:
    - Weights stored in FP8 (float8_e4m3fn) format
    - Block-wise scaling (default 128x128 blocks)
    - Dynamic activation quantization
    - Triton-accelerated matmul
    
    Shapes:
        Input: (batch, seq_len, in_features)
        Output: (batch, seq_len, out_features)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        block_size: tuple = (128, 128),
        dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        """
        Initialize FP8 Linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to use bias (default False)
            block_size: (block_m, block_n) for weight quantization
            dtype: Weight dtype (default float8_e4m3fn)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        
        # FP8 weight
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype),
            requires_grad=False,  # FP8 weights are typically frozen
        )
        
        # Block-wise scale (weight_scale_inv)
        scale_out = (out_features + block_size[0] - 1) // block_size[0]
        scale_in = (in_features + block_size[1] - 1) // block_size[1]
        self.weight_scale_inv = nn.Parameter(
            torch.ones(scale_out, scale_in, dtype=torch.float32),
            requires_grad=False,
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights (for non-pretrained use)."""
        # Initialize to small random values
        with torch.no_grad():
            # Initialize in FP32, then quantize
            w = torch.randn(self.out_features, self.in_features) * 0.02
            self._quantize_weight(w)

    def _quantize_weight(self, weight: torch.Tensor):
        """Quantize a weight tensor to FP8 with block-wise scaling."""
        block_m, block_n = self.block_size
        rows, cols = weight.shape
        
        weight_fp32 = weight.to(torch.float32)
        
        # Handle case where dimensions are smaller than block size
        # Use effective block sizes that evenly divide the dimensions
        eff_block_m = min(block_m, rows)
        eff_block_n = min(block_n, cols)
        
        # For dimensions not divisible by block size, use the dimension as block
        if rows % eff_block_m != 0:
            eff_block_m = rows
        if cols % eff_block_n != 0:
            eff_block_n = cols
            
        rows_tiles = rows // eff_block_m
        cols_tiles = cols // eff_block_n
        
        if rows_tiles == 0:
            rows_tiles = 1
            eff_block_m = rows
        if cols_tiles == 0:
            cols_tiles = 1
            eff_block_n = cols
        
        # Reshape to blocks
        reshaped = weight_fp32.view(rows_tiles, eff_block_m, cols_tiles, eff_block_n)
        
        # Per-block max-abs
        max_abs = reshaped.abs().amax(dim=(1, 3))  # (rows_tiles, cols_tiles)
        safe_max_abs = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
        
        # Compute scales
        scales = 448.0 / safe_max_abs
        scales = torch.where(max_abs > 0, scales, torch.ones_like(scales))
        
        # Quantize
        scales_broadcast = scales[:, None, :, None]
        scaled = reshaped * scales_broadcast
        quantized = torch.clamp(scaled, -448, 448).to(torch.float8_e4m3fn)
        quantized = quantized.view(rows, cols)
        
        # Store weight_scale_inv (inverse of scales)
        inv_scales = 1.0 / scales
        
        # Update weight_scale_inv shape if needed
        scale_out = (self.out_features + self.block_size[0] - 1) // self.block_size[0]
        scale_in = (self.in_features + self.block_size[1] - 1) // self.block_size[1]
        
        # Pad inv_scales to expected shape if needed
        if inv_scales.shape != (scale_out, scale_in):
            padded = torch.ones(scale_out, scale_in, dtype=torch.float32, device=inv_scales.device)
            padded[:inv_scales.shape[0], :inv_scales.shape[1]] = inv_scales
            inv_scales = padded
        
        self.weight.data = quantized
        self.weight_scale_inv.data = inv_scales.to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FP8 computation.
        
        Args:
            x: Input tensor (batch, seq_len, in_features) in bf16/fp16
            
        Returns:
            Output tensor (batch, seq_len, out_features)
        """
        input_dtype = x.dtype
        
        # If weight is not FP8 (e.g., during development), use regular linear
        if self.weight.element_size() > 1:
            return F.linear(x, self.weight, self.bias)
        
        # Ensure contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        weight_scale = self.weight_scale_inv.contiguous()
        
        # Quantize activations
        x_quant, x_scale = act_quant(x, block_size=self.block_size[1])
        
        # FP8 matmul
        output = w8a8_block_fp8_matmul(
            x_quant,
            weight,
            x_scale,
            weight_scale,
            block_size=self.block_size,
            output_dtype=input_dtype,
        )
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
        
        return output

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, block_size={self.block_size}")


# Alias for convenience
FP8Linear = Model


# ============================================================================
# Benchmark Configuration
# ============================================================================
