import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Dynamic Quantization
    
    Used by: INT8 inference optimization
    
    Runtime dynamic quantization of activations based on
    observed min/max values. Used for inference optimization.
    
    Shapes:
        Input: any shape tensor
        Output: same shape (quantized and dequantized)
    """
    
    def __init__(self, num_bits: int = 8, symmetric: bool = True):
        """
        Initialize dynamic quantization.
        
        Args:
            num_bits: Number of quantization bits
            symmetric: Whether to use symmetric quantization
        """
        super(Model, self).__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        
        if symmetric:
            self.qmax = 2 ** (num_bits - 1) - 1
            self.qmin = -(2 ** (num_bits - 1))
        else:
            self.qmax = 2 ** num_bits - 1
            self.qmin = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dynamically quantize and dequantize tensor.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Quantized and dequantized tensor (simulates INT8 compute)
        """
        if self.symmetric:
            # Symmetric quantization
            x_max = x.abs().max()
            scale = x_max / self.qmax
            scale = torch.clamp(scale, min=1e-10)
            
            # Quantize
            x_quant = torch.round(x / scale).clamp(self.qmin, self.qmax)
            
            # Dequantize
            x_deq = x_quant * scale
        else:
            # Asymmetric quantization
            x_min, x_max = x.min(), x.max()
            scale = (x_max - x_min) / (self.qmax - self.qmin)
            scale = torch.clamp(scale, min=1e-10)
            zero_point = torch.round(-x_min / scale)
            
            # Quantize
            x_quant = torch.round(x / scale + zero_point).clamp(self.qmin, self.qmax)
            
            # Dequantize
            x_deq = (x_quant - zero_point) * scale
        
        return x_deq


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
hidden_size = 4096

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [8, True]  # 8-bit symmetric

