import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused Dequantize + GEMM
    
    Used by: All quantized LLM inference (GPTQ, AWQ, INT8, FP8)
    
    Fuses weight dequantization with matrix multiplication.
    Avoids materializing full-precision weights in memory.
    
    Found in: vLLM (Marlin kernels), TensorRT-LLM, AutoAWQ
    
    Shapes:
        Input: (batch_size, seq_len, in_features) FP16/BF16
        Weights: (out_features, in_features) INT4/INT8 packed
        Output: (batch_size, seq_len, out_features) FP16/BF16
    """
    
    def __init__(self, in_features: int, out_features: int, bits: int = 4, group_size: int = 128):
        """
        Initialize fused Dequant + GEMM.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bits: Quantization bits (4 or 8)
            group_size: Quantization group size
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        
        # Quantized weights (simulated as INT8 storage)
        # In real impl, INT4 would be packed 2 per byte
        self.qweight = nn.Parameter(
            torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8),
            requires_grad=False
        )
        
        # Scales per group
        num_groups = (in_features + group_size - 1) // group_size
        self.scales = nn.Parameter(torch.ones(out_features, num_groups))
        self.zeros = nn.Parameter(torch.zeros(out_features, num_groups))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fused dequantize + GEMM.
        
        Args:
            x: Input tensor (batch_size, seq_len, in_features)
            
        Returns:
            Output tensor (batch_size, seq_len, out_features)
        """
        # Dequantize weights on-the-fly
        # In real fused kernels, this happens during GEMM tile loading
        weight_fp = self.qweight.float()
        
        # Apply per-group scales and zeros
        for g in range(self.scales.shape[1]):
            start = g * self.group_size
            end = min(start + self.group_size, self.in_features)
            weight_fp[:, start:end] = (
                (weight_fp[:, start:end] - self.zeros[:, g:g+1]) * 
                self.scales[:, g:g+1]
            )
        
        # GEMM
        weight_fp = weight_fp.to(x.dtype)
        return F.linear(x, weight_fp)


import torch.nn.functional as F

# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
in_features = 4096
out_features = 4096

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, in_features, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [in_features, out_features]

