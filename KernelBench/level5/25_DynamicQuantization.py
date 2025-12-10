import torch
import torch.nn as nn
import torch.nn.functional as F


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Dynamic Quantization Layer with Per-Token Scaling.
    
    Dynamically quantizes activations to INT8 with per-token
    or per-channel scaling factors computed at runtime.
    
    Based on: LLM.int8() and SmoothQuant approaches
    """
    def __init__(self, in_features, out_features, quant_bits=8, 
                 symmetric=True, per_token=True):
        """
        :param in_features: Size of input features
        :param out_features: Size of output features
        :param quant_bits: Number of bits for quantization
        :param symmetric: Use symmetric vs asymmetric quantization
        :param per_token: Per-token vs per-tensor scaling
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_bits = quant_bits
        self.symmetric = symmetric
        self.per_token = per_token
        
        # Quantization range
        if symmetric:
            self.qmin = -(2 ** (quant_bits - 1))
            self.qmax = 2 ** (quant_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** quant_bits - 1
        
        # Weight stored in FP32 (would be pre-quantized in production)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Pre-computed weight quantization parameters
        self.register_buffer('weight_scale', torch.ones(out_features))
        self.register_buffer('weight_zp', torch.zeros(out_features, dtype=torch.int32))
        
        # Quantize weights at init
        self._quantize_weights()
    
    def _quantize_weights(self):
        """Pre-quantize weights for efficient inference."""
        with torch.no_grad():
            if self.symmetric:
                # Per-channel symmetric quantization for weights
                amax = self.weight.abs().max(dim=1).values.clamp(min=1e-8)
                scale = amax / self.qmax
                self.weight_scale.copy_(scale)
                self.weight_zp.zero_()
            else:
                # Per-channel asymmetric quantization
                wmin = self.weight.min(dim=1).values
                wmax = self.weight.max(dim=1).values
                scale = (wmax - wmin) / (self.qmax - self.qmin)
                scale = scale.clamp(min=1e-8)
                zp = (self.qmin - wmin / scale).round().to(torch.int32)
                self.weight_scale.copy_(scale)
                self.weight_zp.copy_(zp)
    
    def quantize_activation(self, x):
        """
        Dynamically quantize activations.
        
        :param x: Input tensor (..., in_features)
        :return: Quantized tensor and scales
        """
        shape = x.shape
        x_flat = x.view(-1, self.in_features)
        
        if self.per_token:
            # Per-token quantization
            if self.symmetric:
                amax = x_flat.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
                scale = amax / self.qmax
                x_quant = (x_flat / scale).round().clamp(self.qmin, self.qmax)
                zp = None
            else:
                xmin = x_flat.min(dim=1, keepdim=True).values
                xmax = x_flat.max(dim=1, keepdim=True).values
                scale = (xmax - xmin) / (self.qmax - self.qmin)
                scale = scale.clamp(min=1e-8)
                zp = (self.qmin - xmin / scale).round()
                x_quant = (x_flat / scale + zp).round().clamp(self.qmin, self.qmax)
        else:
            # Per-tensor quantization
            if self.symmetric:
                amax = x_flat.abs().max().clamp(min=1e-8)
                scale = amax / self.qmax
                x_quant = (x_flat / scale).round().clamp(self.qmin, self.qmax)
                zp = None
            else:
                xmin = x_flat.min()
                xmax = x_flat.max()
                scale = (xmax - xmin) / (self.qmax - self.qmin)
                scale = scale.clamp(min=1e-8)
                zp = (self.qmin - xmin / scale).round()
                x_quant = (x_flat / scale + zp).round().clamp(self.qmin, self.qmax)
        
        return x_quant.view(shape), scale, zp
    
    def dequantize_and_matmul(self, x_quant, x_scale, x_zp):
        """
        Perform quantized matmul and dequantize.
        
        In a real implementation, this would use INT8 matmul kernels.
        """
        shape = x_quant.shape
        x_flat = x_quant.view(-1, self.in_features)
        
        # Dequantize activation
        if x_zp is not None:
            x_deq = (x_flat - x_zp) * x_scale
        else:
            x_deq = x_flat * x_scale
        
        # Quantize weight (simulate)
        w_scale = self.weight_scale.unsqueeze(-1)
        w_zp = self.weight_zp.unsqueeze(-1) if not self.symmetric else 0
        w_quant = ((self.weight / w_scale) + w_zp).round().clamp(self.qmin, self.qmax)
        w_deq = (w_quant - w_zp) * w_scale
        
        # Matrix multiplication
        output = F.linear(x_deq, w_deq, self.bias)
        
        return output.view(*shape[:-1], self.out_features)
    
    def forward(self, x):
        """
        Forward pass with dynamic quantization.
        
        :param x: Input tensor (..., in_features)
        :return: Output tensor (..., out_features)
        """
        # Quantize activation dynamically
        x_quant, x_scale, x_zp = self.quantize_activation(x)
        
        # Perform quantized matmul
        output = self.dequantize_and_matmul(x_quant, x_scale, x_zp)
        
        return output


# Test parameters
batch_size = 32
seq_len = 256
in_features = 4096
out_features = 4096

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    return [torch.randn(batch_size, seq_len, in_features)]

def get_init_inputs():
    return [in_features, out_features]

