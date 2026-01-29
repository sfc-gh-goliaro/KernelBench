import os
import sys
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
