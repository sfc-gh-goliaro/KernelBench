import torch
import torch.nn as nn
import torch.nn.functional as F


TASK_CONFIG = {
    'comparison_mode': 'relative',
    'atol': 1e-5,
    'rtol': 1e-3,
}

class Model(nn.Module):
    """
    Fused RMSNorm + FP8 Quantization.
    
    Combines RMSNorm with immediate FP8 quantization:
    1. Compute RMS normalization
    2. Apply learned scale
    3. Quantize to FP8 format
    
    This is critical for FP8 inference where normalized activations
    feed into quantized linear layers.
    
    Reference: vLLM FP8, SGLang, NVIDIA Transformer Engine
    """
    def __init__(self, hidden_dim, eps=1e-6, quant_dtype='fp8_e4m3',
                 static_scale=None):
        """
        :param hidden_dim: Hidden dimension
        :param eps: RMSNorm epsilon
        :param quant_dtype: 'fp8_e4m3' or 'fp8_e5m2'
        :param static_scale: Optional pre-calibrated scale
        """
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.quant_dtype = quant_dtype
        
        # RMSNorm weight
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        
        # FP8 parameters
        if quant_dtype == 'fp8_e4m3':
            self.fp8_max = 448.0
        else:
            self.fp8_max = 57344.0
        
        # Static or dynamic scaling
        if static_scale is not None:
            self.register_buffer('static_scale', torch.tensor(static_scale))
        else:
            self.static_scale = None
    
    def forward(self, x):
        """
        Fused RMSNorm + FP8 Quantization.
        
        :param x: Input (batch, seq, hidden_dim)
        :return: Tuple of (quantized_output, scale)
        """
        # === FUSED KERNEL START ===
        # RMSNorm
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        normalized = x_norm * self.weight
        
        # FP8 Quantization
        if self.static_scale is not None:
            scale = self.static_scale
        else:
            # Dynamic scaling
            absmax = normalized.abs().max()
            scale = self.fp8_max / absmax.clamp(min=1e-12)
        
        # Quantize
        output = (normalized * scale).clamp(-self.fp8_max, self.fp8_max)
        output = output.to(torch.float16)  # Simulate FP8 storage
        # === FUSED KERNEL END ===
        
        return output, 1.0 / scale


# Per-tensor vs Per-token variants
class FusedRMSNormFP8PerToken(nn.Module):
    """
    Fused RMSNorm + Per-Token FP8 Quantization.
    
    Uses per-token scaling for better accuracy at cost of more scales.
    """
    def __init__(self, hidden_dim, eps=1e-6, quant_dtype='fp8_e4m3'):
        super(FusedRMSNormFP8PerToken, self).__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.fp8_max = 448.0 if 'e4m3' in quant_dtype else 57344.0
        
        self.weight = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, x):
        """
        Per-token FP8 quantization.
        
        :return: (quantized, scales) where scales is per-token
        """
        # RMSNorm
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps) * self.weight
        
        # Per-token quantization
        # Shape: (batch, seq, 1)
        absmax = normalized.abs().max(dim=-1, keepdim=True).values
        scales = self.fp8_max / absmax.clamp(min=1e-12)
        
        output = (normalized * scales).clamp(-self.fp8_max, self.fp8_max)
        
        return output.to(torch.float16), (1.0 / scales).squeeze(-1)


# With residual input
class FusedAddRMSNormFP8Quant(nn.Module):
    """
    Fused Residual Add + RMSNorm + FP8 Quantization.
    
    Complete post-sublayer operation with quantization.
    """
    def __init__(self, hidden_dim, eps=1e-6, quant_dtype='fp8_e4m3'):
        super(FusedAddRMSNormFP8Quant, self).__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.fp8_max = 448.0 if 'e4m3' in quant_dtype else 57344.0
        
        self.weight = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, x, residual):
        """
        Fused add + RMSNorm + FP8 quant.
        
        :param x: Sublayer output
        :param residual: Residual input
        :return: (quantized, scale, residual_out)
        """
        # === FUSED KERNEL START ===
        # Residual add
        hidden = x + residual
        
        # RMSNorm
        variance = hidden.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden * torch.rsqrt(variance + self.eps) * self.weight
        
        # FP8 quantization
        absmax = normalized.abs().max()
        scale = self.fp8_max / absmax.clamp(min=1e-12)
        output = (normalized * scale).clamp(-self.fp8_max, self.fp8_max).to(torch.float16)
        # === FUSED KERNEL END ===
        
        # Also return hidden for next residual
        return output, 1.0 / scale, hidden


# Delayed scaling (for amax history)
class FusedRMSNormFP8DelayedScale(nn.Module):
    """
    Fused RMSNorm + FP8 with delayed scaling.
    
    Uses historical amax values for more stable quantization.
    """
    def __init__(self, hidden_dim, eps=1e-6, history_len=16):
        super(FusedRMSNormFP8DelayedScale, self).__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.fp8_max = 448.0
        self.history_len = history_len
        
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        
        # Amax history buffer
        self.register_buffer('amax_history', torch.zeros(history_len))
        self.register_buffer('history_idx', torch.tensor(0))
    
    def forward(self, x, update_scale=True):
        """Forward with delayed scaling."""
        # RMSNorm
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps) * self.weight
        
        # Current amax
        current_amax = normalized.abs().max()
        
        # Use historical max for scale
        if self.amax_history.sum() > 0:
            historical_max = self.amax_history.max()
            scale = self.fp8_max / historical_max.clamp(min=1e-12)
        else:
            scale = self.fp8_max / current_amax.clamp(min=1e-12)
        
        # Update history
        if update_scale:
            idx = self.history_idx.item() % self.history_len
            self.amax_history[idx] = current_amax
            self.history_idx += 1
        
        output = (normalized * scale).clamp(-self.fp8_max, self.fp8_max).to(torch.float16)
        
        return output, 1.0 / scale


# Test parameters
batch_size = 32
seq_len = 2048
hidden_dim = 4096

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    return [torch.randn(batch_size, seq_len, hidden_dim)]

def get_init_inputs():
    return [hidden_dim]

