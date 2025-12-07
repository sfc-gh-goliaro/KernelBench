import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Activation-aware Weight Quantization (AWQ) GEMM.
    
    Protects salient weight channels based on activation magnitude
    for better accuracy in weight-only quantization.
    
    Based on: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
    """
    def __init__(self, in_features, out_features, group_size=128, w_bits=4):
        """
        :param in_features: Size of input features
        :param out_features: Size of output features
        :param group_size: Size of quantization groups
        :param w_bits: Number of bits for weight quantization
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.w_bits = w_bits
        self.num_groups = in_features // group_size
        
        # Quantization range
        self.qmax = 2 ** w_bits - 1
        
        # Original weight (would be discarded after quantization in production)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Per-channel scaling (learned from activation statistics)
        self.register_buffer('act_scale', torch.ones(in_features))
        
        # Quantization parameters per group
        self.register_buffer('scales', torch.ones(out_features, self.num_groups))
        self.register_buffer('zeros', torch.zeros(out_features, self.num_groups))
        
    def calibrate(self, sample_activations):
        """
        Calibrate scaling factors from sample activations.
        
        :param sample_activations: Sample activations (..., in_features)
        """
        with torch.no_grad():
            # Compute activation statistics
            x_flat = sample_activations.view(-1, self.in_features)
            act_mean = x_flat.abs().mean(dim=0)
            
            # Salient channels have high average activation
            # Scale protects these channels
            self.act_scale.copy_(act_mean.clamp(min=1e-8))
            
            # Find optimal per-channel scale via grid search
            best_scale = self._search_best_scale()
            
            # Quantize weights with optimal scaling
            self._quantize_weights(best_scale)
    
    def _search_best_scale(self, n_grid=20):
        """Search for optimal scaling factor."""
        # Simplified grid search over scaling factors
        scale_range = torch.linspace(0.1, 1.0, n_grid, device=self.weight.device)
        
        best_scale = torch.ones_like(self.act_scale)
        best_error = float('inf')
        
        for s in scale_range:
            # Apply uniform scaling
            test_scale = self.act_scale * s
            
            # Compute quantization error with this scaling
            w_scaled = self.weight * test_scale.unsqueeze(0)
            w_quant = self._fake_quantize(w_scaled)
            w_deq = w_quant / test_scale.unsqueeze(0)
            
            error = (self.weight - w_deq).pow(2).mean()
            
            if error < best_error:
                best_error = error
                best_scale = test_scale.clone()
        
        return best_scale
    
    def _fake_quantize(self, weight):
        """Fake quantization for scale search."""
        w_reshape = weight.view(self.out_features, self.num_groups, self.group_size)
        
        w_min = w_reshape.min(dim=-1, keepdim=True).values
        w_max = w_reshape.max(dim=-1, keepdim=True).values
        
        scale = (w_max - w_min) / self.qmax
        scale = scale.clamp(min=1e-8)
        
        w_quant = ((w_reshape - w_min) / scale).round().clamp(0, self.qmax)
        w_deq = w_quant * scale + w_min
        
        return w_deq.view(self.out_features, self.in_features)
    
    def _quantize_weights(self, channel_scale):
        """Quantize weights with AWQ scaling."""
        with torch.no_grad():
            # Apply channel scaling
            w_scaled = self.weight * channel_scale.unsqueeze(0)
            
            # Group-wise quantization
            w_reshape = w_scaled.view(self.out_features, self.num_groups, self.group_size)
            
            w_min = w_reshape.min(dim=-1).values
            w_max = w_reshape.max(dim=-1).values
            
            scale = (w_max - w_min) / self.qmax
            scale = scale.clamp(min=1e-8)
            zero = w_min
            
            self.scales.copy_(scale)
            self.zeros.copy_(zero)
            
            # Store channel scale for inference
            self.act_scale.copy_(channel_scale)
    
    def dequantize_weight(self):
        """Dequantize weight for inference."""
        # Reconstruct from quantization params
        w_reshape = self.weight.view(self.out_features, self.num_groups, self.group_size)
        
        # Fake dequant using stored scales
        w_scaled = self.weight * self.act_scale.unsqueeze(0)
        w_quant = self._fake_quantize(w_scaled)
        w_deq = w_quant / self.act_scale.unsqueeze(0)
        
        return w_deq
    
    def forward(self, x):
        """
        Forward pass with AWQ dequantized weights.
        
        :param x: Input tensor (..., in_features)
        :return: Output tensor (..., out_features)
        """
        # Dequantize weights
        w_deq = self.dequantize_weight()
        
        # Matrix multiplication
        output = F.linear(x, w_deq, self.bias)
        
        return output


# Test parameters
batch_size = 32
seq_len = 256
in_features = 4096
out_features = 4096
group_size = 128

def get_inputs():
    return [torch.randn(batch_size, seq_len, in_features)]

def get_init_inputs():
    return [in_features, out_features, group_size]

