import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    W4A16 GEMM (4-bit Weight, 16-bit Activation)
    
    Used by: GPTQ, AWQ quantized models
    
    Weight-only 4-bit quantized GEMM with on-the-fly dequantization.
    Weights stored as INT4, activations remain FP16.
    
    Shapes:
        Input: (batch, seq_len, in_features)
        Output: (batch, seq_len, out_features)
    """
    
    def __init__(self, in_features: int, out_features: int, group_size: int = 128):
        """
        Initialize W4A16 linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            group_size: Quantization group size
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        # Number of groups
        self.num_groups = in_features // group_size
        
        # Quantized weights (packed 4-bit, 2 values per byte)
        # Shape: (out_features, in_features // 2) as uint8
        self.register_buffer('qweight', torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        
        # Scales per group: (num_groups, out_features)
        self.register_buffer('scales', torch.ones(self.num_groups, out_features))
        
        # Zero points per group: (num_groups, out_features)
        self.register_buffer('zeros', torch.zeros(self.num_groups, out_features, dtype=torch.int8))
    
    def _dequantize(self) -> torch.Tensor:
        """Dequantize 4-bit weights to FP16."""
        # Unpack 4-bit values
        w_low = self.qweight & 0x0F
        w_high = self.qweight >> 4
        
        # Interleave low and high bits
        weights = torch.stack([w_low, w_high], dim=-1).view(self.out_features, self.in_features)
        weights = weights.to(torch.float16)
        
        # Apply dequantization per group
        # weights = (qweight - zeros) * scales
        for g in range(self.num_groups):
            start = g * self.group_size
            end = start + self.group_size
            weights[:, start:end] = (weights[:, start:end] - self.zeros[g].unsqueeze(1).float()) * self.scales[g].unsqueeze(1)
        
        return weights.T  # Return (in_features, out_features) for matmul
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dequantization.
        
        Args:
            x: Input tensor (batch, seq_len, in_features)
            
        Returns:
            Output tensor (batch, seq_len, out_features)
        """
        # Dequantize weights
        weight = self._dequantize()
        
        # Matrix multiplication
        return torch.matmul(x, weight)


# ============================================================================
# Benchmark Configuration
# ============================================================================
