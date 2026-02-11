import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    KV Cache Quantization
    
    Used by: Memory-constrained serving
    
    Quantize KV cache to INT8 or FP8 for memory reduction.
    Dequantize during attention computation.
    
    Shapes:
        Input K, V: (batch, num_heads, seq_len, head_dim)
        Output: quantized tensors + scales
    """
    
    def __init__(self, num_heads: int, head_dim: int, num_bits: int = 8):
        """
        Initialize KV cache quantization.
        
        Args:
            num_heads: Number of attention heads
            head_dim: Dimension per head
            num_bits: Quantization bits (8 for INT8)
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_bits = num_bits
        
        # Quantization range
        self.qmax = 2 ** (num_bits - 1) - 1
        self.qmin = -(2 ** (num_bits - 1))
    
    def quantize(self, x: torch.Tensor) -> tuple:
        """
        Quantize tensor to INT8.
        
        Args:
            x: Input tensor (batch, num_heads, seq_len, head_dim)
            
        Returns:
            Tuple of (quantized_tensor, scale)
        """
        # Per-token quantization (compute scale per token)
        x_flat = x.view(-1, self.head_dim)
        
        # Compute scale per token
        x_max = x_flat.abs().max(dim=-1, keepdim=True).values
        scale = x_max / self.qmax
        scale = torch.clamp(scale, min=1e-10)
        
        # Quantize
        x_quant = torch.round(x_flat / scale).clamp(self.qmin, self.qmax).to(torch.int8)
        
        # Reshape back
        x_quant = x_quant.view(x.shape)
        scale = scale.view(x.shape[:-1] + (1,))
        
        return x_quant, scale
    
    def dequantize(self, x_quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Dequantize INT8 tensor back to FP16.
        
        Args:
            x_quant: Quantized tensor (INT8)
            scale: Scale factors
            
        Returns:
            Dequantized FP16 tensor
        """
        return x_quant.float() * scale
    
    def forward(self, k: torch.Tensor, v: torch.Tensor) -> tuple:
        """
        Quantize K and V caches.
        
        Args:
            k: Key cache (batch, num_heads, seq_len, head_dim)
            v: Value cache (batch, num_heads, seq_len, head_dim)
            
        Returns:
            Tuple of ((k_quant, k_scale), (v_quant, v_scale))
        """
        k_quant, k_scale = self.quantize(k)
        v_quant, v_scale = self.quantize(v)
        
        return (k_quant, k_scale), (v_quant, v_scale)
