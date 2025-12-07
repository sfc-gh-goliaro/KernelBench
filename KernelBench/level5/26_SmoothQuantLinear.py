import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    SmoothQuant Linear Layer.
    
    Migrates quantization difficulty from activations to weights
    using mathematically equivalent per-channel scaling.
    
    Based on: "SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs"
    """
    def __init__(self, in_features, out_features, alpha=0.5, quant_bits=8):
        """
        :param in_features: Size of input features
        :param out_features: Size of output features
        :param alpha: Migration strength (0=all to weight, 1=all to activation)
        :param quant_bits: Number of bits for quantization
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.quant_bits = quant_bits
        
        # Quantization range
        self.qmax = 2 ** (quant_bits - 1) - 1
        self.qmin = -(2 ** (quant_bits - 1))
        
        # Weight matrix
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Smoothing scales (per-channel)
        self.register_buffer('smooth_scale', torch.ones(in_features))
        
        # Calibrated activation statistics
        self.register_buffer('act_scales', torch.ones(in_features))
        self.register_buffer('calibrated', torch.tensor(False))
        
    def calibrate(self, x):
        """
        Calibrate smoothing scales from sample activations.
        
        :param x: Sample activations (..., in_features)
        """
        with torch.no_grad():
            # Compute per-channel activation max
            x_flat = x.view(-1, self.in_features)
            act_max = x_flat.abs().max(dim=0).values.clamp(min=1e-8)
            
            # Compute per-channel weight max
            w_max = self.weight.abs().max(dim=0).values.clamp(min=1e-8)
            
            # Compute smoothing scale
            # s = act_max^alpha / w_max^(1-alpha)
            smooth_scale = (act_max ** self.alpha) / (w_max ** (1 - self.alpha))
            smooth_scale = smooth_scale.clamp(min=1e-8)
            
            self.smooth_scale.copy_(smooth_scale)
            self.act_scales.copy_(act_max)
            self.calibrated.copy_(torch.tensor(True))
    
    def get_smooth_weight(self):
        """Get smoothed weight matrix."""
        # W_smooth = W * diag(s)
        return self.weight * self.smooth_scale.unsqueeze(0)
    
    def quantize_symmetric(self, tensor, scale):
        """Symmetric quantization to INT8."""
        tensor_scaled = tensor / scale
        tensor_quant = tensor_scaled.round().clamp(self.qmin, self.qmax)
        return tensor_quant, scale
    
    def forward(self, x):
        """
        Forward pass with SmoothQuant.
        
        :param x: Input tensor (..., in_features)
        :return: Output tensor (..., out_features)
        """
        shape = x.shape
        x_flat = x.view(-1, self.in_features)
        
        # Apply smoothing to activations
        # X_smooth = X / diag(s)
        x_smooth = x_flat / self.smooth_scale.unsqueeze(0)
        
        # Get smoothed weights
        w_smooth = self.get_smooth_weight()
        
        # Quantize activations (per-token)
        x_max = x_smooth.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        x_scale = x_max / self.qmax
        x_quant, _ = self.quantize_symmetric(x_smooth, x_scale)
        
        # Quantize weights (per-channel)
        w_max = w_smooth.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        w_scale = w_max / self.qmax
        w_quant, _ = self.quantize_symmetric(w_smooth, w_scale)
        
        # Dequantized matmul (simulated INT8 matmul)
        x_deq = x_quant * x_scale
        w_deq = w_quant * w_scale
        
        output = F.linear(x_deq, w_deq, self.bias)
        
        return output.view(*shape[:-1], self.out_features)
    
    def forward_fp32(self, x):
        """Forward pass without quantization (for comparison)."""
        return F.linear(x, self.weight, self.bias)


# Test parameters
batch_size = 32
seq_len = 256
in_features = 4096
out_features = 4096

def get_inputs():
    return [torch.randn(batch_size, seq_len, in_features)]

def get_init_inputs():
    return [in_features, out_features]

