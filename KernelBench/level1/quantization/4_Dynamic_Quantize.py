import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
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


PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096},
    # Llama-3.1-70B: Dynamic INT8 activation quantization
    {"batch_size": 2, "seq_length": 4096, "hidden_size": 8192},
    # Mistral-Nemo-12B: Runtime quantization for memory efficiency
    {"batch_size": 8, "seq_length": 8192, "hidden_size": 5120},
    # Phi-3-medium: Dynamic quantization for edge deployment
    {"batch_size": 16, "seq_length": 1024, "hidden_size": 5120},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("quantization", "4_Dynamic_Quantize")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["hidden_size"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return [8, True]
