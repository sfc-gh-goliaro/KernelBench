import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused AllReduce + RMSNorm + FP8 Quantization
    
    Used by: Tensor-parallel LLM inference with FP8 (H100, MI300X)
    
    Fuses all-reduce collective, RMSNorm, and FP8 quantization.
    This is the optimal fusion for FP8 inference in TP settings,
    producing quantized output directly for the next GEMM.
    
    Found in: TensorRT-LLM (kARResidualRMSNormFP8Quant)
    
    Shapes:
        Input: (batch_size, seq_len, hidden_size) 
        Residual: (batch_size, seq_len, hidden_size)
        Output: (batch_size, seq_len, hidden_size) quantized + scale
    """
    
    def __init__(self, hidden_size: int, tp_size: int = 1, eps: float = 1e-6):
        """
        Initialize fused AllReduce + RMSNorm + Quant.
        
        Args:
            hidden_size: Hidden dimension
            tp_size: Tensor parallel size
            eps: Epsilon for numerical stability
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
        # FP8 E4M3 range (simulated)
        self.fp8_max = 448.0  # E4M3 max value
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fused AllReduce + Residual + RMSNorm + FP8 Quantization.
        
        Args:
            x: Input tensor (local partition result)
            residual: Residual tensor
            
        Returns:
            Tuple of (quantized output, scale, updated residual)
        """
        # Simulate all-reduce
        reduced = x * self.tp_size
        
        # Fused residual add
        residual = reduced + residual
        
        # RMSNorm
        rms = torch.sqrt(residual.pow(2).mean(-1, keepdim=True) + self.eps)
        normed = residual / rms * self.weight
        
        # FP8 Quantization (simulated - real impl uses torch.float8_e4m3fn)
        # Compute per-tensor scale
        amax = normed.abs().max()
        scale = self.fp8_max / amax.clamp(min=1e-12)
        
        # Quantize (clamp to FP8 range and simulate reduced precision)
        quantized = (normed * scale).clamp(-self.fp8_max, self.fp8_max)
        # Round to simulate reduced precision
        quantized = quantized.round()
        
        return quantized, scale, residual


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
hidden_size = 4096
tp_size = 1

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    residual = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    return [x, residual]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, tp_size]

