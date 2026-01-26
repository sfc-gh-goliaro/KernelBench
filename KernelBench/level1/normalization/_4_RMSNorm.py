import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
from typing import Optional

class Model(nn.Module):
    """
    RMS Normalization with optional learnable weight.
    
    Used by: Llama, Mistral, Qwen, Gemma, Yi, DeepSeek, Phi
    
    Supports two layouts:
        - dim=1: Input shape (batch_size, num_features, *) - normalize along dim 1
        - dim=-1: Input shape (batch_size, seq_len, num_features) - normalize along last dim (Llama style)
    """
    def __init__(
        self, 
        num_features: int, 
        eps: float = 1e-5, 
        learnable_weight: bool = False,
        dim: int = 1
    ):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
            learnable_weight (bool, optional): If True, adds a learnable weight parameter. Defaults to False.
            dim (int, optional): Dimension along which to normalize. Use 1 for (batch, features, *) 
                                 or -1 for (batch, seq, features). Defaults to 1.
        """
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.dim = dim
        self.learnable_weight = learnable_weight
        
        if learnable_weight:
            self.weight = nn.Parameter(torch.ones(num_features))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor. Shape depends on dim parameter:
                - dim=1: (batch_size, num_features, *)
                - dim=-1: (batch_size, *, num_features)

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        # Calculate the RMS along the specified dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=self.dim, keepdim=True) + self.eps)

        # Normalize the input by dividing by the RMS
        normalized = x / rms
        
        # Apply learnable weight if enabled
        if self.weight is not None:
            normalized = normalized * self.weight
        
        return normalized


PARAMETERS = [
    {"batch_size": 112, "features": 64, "dim1": 512, "dim2": 512},
    # Llama-3.1-8B: hidden_size=4096
    {"batch_size": 8, "features": 4096, "dim1": 1, "dim2": 2048},
    # Llama-3.1-70B: hidden_size=8192
    {"batch_size": 4, "features": 8192, "dim1": 1, "dim2": 2048},
    # Mistral-7B-v0.3: hidden_size=4096
    {"batch_size": 8, "features": 4096, "dim1": 1, "dim2": 4096},
    # DeepSeek-V2-Lite: hidden_size=2048
    {"batch_size": 8, "features": 2048, "dim1": 1, "dim2": 4096},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("normalization", "4_RMSNorm")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["features"], p["dim1"], p["dim2"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["features"]]
