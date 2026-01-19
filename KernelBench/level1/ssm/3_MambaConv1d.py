import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Mamba Causal Conv1d
    
    Used by: Mamba, Mamba-2
    
    Causal depthwise 1D convolution for Mamba input preprocessing
    before selective scan. Provides local context to each position.
    
    Shapes:
        Input: (batch, seq_len, d_inner)
        Output: (batch, seq_len, d_inner)
    """
    
    def __init__(self, d_inner: int, kernel_size: int = 4):
        """
        Initialize Mamba Conv1d.
        
        Args:
            d_inner: Number of channels (processed depthwise)
            kernel_size: Convolution kernel size
        """
        super(Model, self).__init__()
        self.d_inner = d_inner
        self.kernel_size = kernel_size
        
        # Depthwise causal convolution
        self.conv = nn.Conv1d(
            d_inner, d_inner, kernel_size,
            padding=kernel_size - 1,  # Causal padding
            groups=d_inner,  # Depthwise
            bias=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply causal depthwise convolution.
        
        Args:
            x: Input tensor (batch, seq_len, d_inner)
            
        Returns:
            Output tensor (batch, seq_len, d_inner)
        """
        seq_len = x.shape[1]
        
        # Transpose for conv1d: (batch, d_inner, seq_len)
        x = x.transpose(1, 2)
        
        # Apply convolution
        x = self.conv(x)
        
        # Remove extra padding to maintain causality
        x = x[:, :, :seq_len]
        
        # Apply SiLU activation (as in Mamba)
        x = F.silu(x)
        
        # Transpose back: (batch, seq_len, d_inner)
        x = x.transpose(1, 2)
        
        return x


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "d_inner": 4096, "kernel_size": 4},
    # Mamba-2-1.3B: d_model=2048, expand=2, d_inner=4096
    {"batch_size": 8, "seq_length": 4096, "d_inner": 4096, "kernel_size": 4},
    # Mamba-2-2.7B: d_model=2560, expand=2, d_inner=5120
    {"batch_size": 8, "seq_length": 2048, "d_inner": 5120, "kernel_size": 4},
    # Mamba-370M: d_model=1024, expand=2, d_inner=2048
    {"batch_size": 16, "seq_length": 2048, "d_inner": 2048, "kernel_size": 4},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("ssm", "3_MambaConv1d")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["d_inner"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["d_inner"], p["kernel_size"]]
