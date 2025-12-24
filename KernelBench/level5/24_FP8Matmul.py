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
    FP8 (8-bit Floating Point) Matrix Multiplication.
    
    Implements FP8 quantized matrix multiplication with per-tensor
    or per-channel scaling for training and inference.
    
    Based on: "FP8 Formats for Deep Learning" (NVIDIA, ARM, Intel)
    """
    def __init__(self, in_features, out_features, fp8_format='e4m3', 
                 per_channel=False, bias=True):
        """
        :param in_features: Size of input features
        :param out_features: Size of output features
        :param fp8_format: 'e4m3' or 'e5m2' (exponent/mantissa bits)
        :param per_channel: Use per-channel vs per-tensor scaling
        :param bias: Whether to include bias
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fp8_format = fp8_format
        self.per_channel = per_channel
        
        # FP8 format specifications
        if fp8_format == 'e4m3':
            self.max_val = 448.0  # Max representable value
            self.min_val = 2**-9  # Min positive value
        else:  # e5m2
            self.max_val = 57344.0
            self.min_val = 2**-16
        
        # Weight stored in FP32, will be quantized during forward
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        
        # Scaling factors
        if per_channel:
            self.register_buffer('weight_scale', torch.ones(out_features))
            self.register_buffer('input_scale', torch.ones(1))
        else:
            self.register_buffer('weight_scale', torch.ones(1))
            self.register_buffer('input_scale', torch.ones(1))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Cached quantized weight
        self.register_buffer('qweight', None)
    
    def compute_scale(self, tensor, max_val):
        """Compute scaling factor for FP8 quantization."""
        if self.per_channel and tensor.dim() > 1:
            amax = tensor.abs().max(dim=-1).values.clamp(min=1e-12)
        else:
            amax = tensor.abs().max().clamp(min=1e-12)
        return amax / max_val
    
    def quantize_fp8(self, tensor, scale):
        """Simulate FP8 quantization."""
        # Scale down
        scaled = tensor / scale.unsqueeze(-1) if scale.dim() > 0 and tensor.dim() > 1 else tensor / scale
        
        # Clamp to FP8 range
        clamped = scaled.clamp(-self.max_val, self.max_val)
        
        # Simulate reduced precision (round to FP8)
        # For e4m3: 4 exponent bits, 3 mantissa bits
        # For e5m2: 5 exponent bits, 2 mantissa bits
        if self.fp8_format == 'e4m3':
            # Simulate 3-bit mantissa precision
            log_scale = 2 ** torch.floor(torch.log2(clamped.abs().clamp(min=self.min_val)))
            quantized = torch.round(clamped / log_scale * 8) / 8 * log_scale
        else:
            # Simulate 2-bit mantissa precision
            log_scale = 2 ** torch.floor(torch.log2(clamped.abs().clamp(min=self.min_val)))
            quantized = torch.round(clamped / log_scale * 4) / 4 * log_scale
        
        # Handle special values
        quantized = torch.where(clamped == 0, torch.zeros_like(quantized), quantized)
        
        return quantized, scale
    
    def forward(self, x):
        """
        Forward pass with FP8 quantized computation.
        
        :param x: Input tensor (batch, *, in_features)
        :return: Output tensor (batch, *, out_features)
        """
        # Compute input scale dynamically
        input_scale = self.compute_scale(x, self.max_val)
        self.input_scale.copy_(input_scale if input_scale.dim() == 0 else input_scale.mean())
        
        # Quantize input to FP8
        x_fp8, _ = self.quantize_fp8(x, self.input_scale)
        
        # Compute weight scale
        weight_scale = self.compute_scale(self.weight, self.max_val)
        if self.per_channel:
            self.weight_scale.copy_(weight_scale)
        else:
            self.weight_scale.copy_(weight_scale.view(1))
        
        # Quantize weight to FP8
        w_fp8, _ = self.quantize_fp8(self.weight, self.weight_scale.unsqueeze(-1) if self.per_channel else self.weight_scale)
        
        # Matrix multiplication (in FP32/FP16 after dequant)
        output = F.linear(x_fp8 * self.input_scale, w_fp8 * self.weight_scale.unsqueeze(-1) if self.per_channel else w_fp8 * self.weight_scale, self.bias)
        
        return output
    
    def calibrate(self, x):
        """
        Calibrate scaling factors using sample data.
        
        :param x: Sample input tensor for calibration
        """
        with torch.no_grad():
            # Calibrate input scale
            input_scale = self.compute_scale(x, self.max_val)
            self.input_scale.copy_(input_scale.mean() if input_scale.dim() > 0 else input_scale)
            
            # Calibrate weight scale
            weight_scale = self.compute_scale(self.weight, self.max_val)
            if self.per_channel:
                self.weight_scale.copy_(weight_scale)
            else:
                self.weight_scale.copy_(weight_scale.mean().view(1))


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

