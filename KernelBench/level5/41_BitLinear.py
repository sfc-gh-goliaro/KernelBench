import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    BitLinear: 1-bit Weight Linear Layer.
    
    Quantizes weights to {-1, +1} during forward pass while
    maintaining full precision gradients for training.
    
    Based on: "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
    """
    def __init__(self, in_features, out_features, bits=1, use_absmax_quant=True):
        """
        :param in_features: Size of input features
        :param out_features: Size of output features
        :param bits: Number of bits (1 for binary, 1.58 for ternary)
        :param use_absmax_quant: Use absmax quantization for activations
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.use_absmax_quant = use_absmax_quant
        
        # Full precision weights (quantized during forward)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        
        # Optional bias
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Layer normalization for input (RMSNorm variant)
        self.input_norm = nn.RMSNorm(in_features) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(in_features)
        
    def weight_quant(self, weight):
        """
        Quantize weights to 1-bit (binary) or 1.58-bit (ternary).
        Uses Straight-Through Estimator (STE) for gradients.
        """
        if self.bits == 1:
            # Binary quantization: {-1, +1}
            # Scale by mean absolute value
            scale = weight.abs().mean()
            weight_quant = torch.sign(weight)
            return weight_quant * scale
        else:
            # Ternary quantization: {-1, 0, +1} (1.58 bits)
            scale = weight.abs().mean()
            threshold = 0.5 * scale
            weight_quant = torch.where(
                weight > threshold, torch.ones_like(weight),
                torch.where(weight < -threshold, -torch.ones_like(weight), 
                           torch.zeros_like(weight))
            )
            return weight_quant * scale
    
    def activation_quant(self, x):
        """
        Quantize activations to 8-bit using absmax scaling.
        """
        if not self.use_absmax_quant:
            return x
        
        # Per-token absmax scaling
        scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        scale = scale / 127  # Scale to INT8 range
        
        # Quantize
        x_quant = (x / scale).round().clamp(-128, 127)
        
        # Dequantize (for computation in higher precision)
        return x_quant * scale
    
    def forward(self, x):
        """
        Forward pass with 1-bit weights and quantized activations.
        
        :param x: Input tensor (..., in_features)
        :return: Output tensor (..., out_features)
        """
        # Normalize input
        x_norm = self.input_norm(x)
        
        # Quantize activations
        x_quant = self.activation_quant(x_norm)
        
        # Quantize weights (with STE gradient)
        w_quant = self.weight_quant(self.weight)
        
        # Linear transformation
        output = F.linear(x_quant, w_quant, self.bias)
        
        return output
    
    def get_quantized_weight(self):
        """Get the quantized weight for inference."""
        with torch.no_grad():
            return self.weight_quant(self.weight)


# Test parameters
batch_size = 32
seq_len = 512
in_features = 4096
out_features = 4096

def get_inputs():
    return [torch.randn(batch_size, seq_len, in_features)]

def get_init_inputs():
    return [in_features, out_features]

