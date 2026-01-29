import os
import sys
import torch
import torch.nn as nn
from typing import Optional

class Model(nn.Module):
    """
    RMS Normalization with optional learnable weight.
    
    Used by: Llama, Mistral, Qwen, Gemma, Yi, DeepSeek, Phi
    
    Implements exact HuggingFace LlamaRMSNorm behavior:
    - Casts to float32 for numerical stability (important for bf16)
    - Uses rsqrt for computation
    - Returns result in input dtype
    
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
        
        Matches HuggingFace's LlamaRMSNorm exactly for numerical precision.

        Args:
            x (torch.Tensor): Input tensor. Shape depends on dim parameter:
                - dim=1: (batch_size, num_features, *)
                - dim=-1: (batch_size, *, num_features)

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        input_dtype = x.dtype
        # Cast to float32 for numerical stability (matches HuggingFace)
        x = x.to(torch.float32)
        
        # Compute variance = mean(x^2) along the normalization dimension
        variance = x.pow(2).mean(dim=self.dim, keepdim=True)
        
        # Normalize using rsqrt (matches HuggingFace exactly)
        x = x * torch.rsqrt(variance + self.eps)
        
        # Apply learnable weight if enabled, then cast back to input dtype
        if self.weight is not None:
            x = self.weight * x.to(input_dtype)
        else:
            x = x.to(input_dtype)
        
        return x
