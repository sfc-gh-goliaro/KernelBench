import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused GEMM + Quantize
    
    Used by: FP8 LLM inference (H100, MI300X)
    
    Fuses matrix multiplication with output quantization.
    Produces quantized output directly for the next layer,
    avoiding FP16/BF16 intermediate storage.
    
    Found in: TensorRT-LLM FP8 flow, vLLM FP8
    
    Shapes:
        Input: (batch_size, seq_len, in_features) 
        Output: (batch_size, seq_len, out_features) quantized + scale
    """
    
    def __init__(self, in_features: int, out_features: int, output_bits: int = 8):
        """
        Initialize fused GEMM + Quantize.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            output_bits: Output quantization bits (8 for FP8/INT8)
        """
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.output_bits = output_bits
        
        # FP8 E4M3 range
        self.fp8_max = 448.0 if output_bits == 8 else 127.0
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fused GEMM + Quantize.
        
        Args:
            x: Input tensor (batch_size, seq_len, in_features)
            
        Returns:
            Tuple of (quantized output, scale)
        """
        # GEMM
        output = self.linear(x)
        
        # Quantize output (fused in real kernels)
        # Compute per-tensor scale
        amax = output.abs().max()
        scale = self.fp8_max / amax.clamp(min=1e-12)
        
        # Quantize
        quantized = (output * scale).clamp(-self.fp8_max, self.fp8_max)
        quantized = quantized.round()
        
        return quantized, scale


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

