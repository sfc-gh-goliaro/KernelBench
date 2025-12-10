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
    Fused SiLU + Quantization (FP8/INT8).
    
    Combines the SiLU/Swish activation with immediate quantization:
    1. Apply SiLU activation: x * sigmoid(x)
    2. Quantize to FP8 or INT8
    
    Used in FP8 inference pipelines for MLP blocks where activation
    output feeds into quantized GEMM.
    
    Reference: vLLM FP8, NVIDIA Transformer Engine
    """
    def __init__(self, quant_dtype='fp8_e4m3', per_token=False):
        """
        :param quant_dtype: 'fp8_e4m3', 'fp8_e5m2', or 'int8'
        :param per_token: Use per-token scaling (for better accuracy)
        """
        super(Model, self).__init__()
        self.quant_dtype = quant_dtype
        self.per_token = per_token
        
        if quant_dtype == 'fp8_e4m3':
            self.max_val = 448.0
        elif quant_dtype == 'fp8_e5m2':
            self.max_val = 57344.0
        else:  # int8
            self.max_val = 127.0
    
    def _quantize(self, x):
        """Quantize tensor."""
        if self.per_token:
            # Per-token (per-row) scaling
            absmax = x.abs().max(dim=-1, keepdim=True).values
            scale = self.max_val / absmax.clamp(min=1e-12)
        else:
            # Per-tensor scaling
            absmax = x.abs().max()
            scale = self.max_val / absmax.clamp(min=1e-12)
        
        x_quant = (x * scale).clamp(-self.max_val, self.max_val)
        
        if 'int8' in self.quant_dtype:
            x_quant = x_quant.round().to(torch.int8)
        else:
            x_quant = x_quant.to(torch.float16)
        
        return x_quant, 1.0 / scale if not self.per_token else (1.0 / scale).squeeze(-1)
    
    def forward(self, x):
        """
        Fused SiLU + Quantization.
        
        :param x: Input tensor (*, hidden_dim)
        :return: Tuple of (quantized_output, scale)
        """
        # === FUSED KERNEL START ===
        # SiLU activation
        activated = F.silu(x)
        
        # Quantization (fused in single kernel pass)
        output, scale = self._quantize(activated)
        # === FUSED KERNEL END ===
        
        return output, scale


# SiLU + Multiply + Quant (for SwiGLU second half)
class FusedSiLUMulQuant(nn.Module):
    """
    Fused SiLU + Multiply + Quantization.
    
    For SwiGLU: output = silu(gate) * up, then quantize
    """
    def __init__(self, quant_dtype='fp8_e4m3'):
        super(FusedSiLUMulQuant, self).__init__()
        self.quant_dtype = quant_dtype
        self.max_val = 448.0 if 'e4m3' in quant_dtype else 57344.0
    
    def forward(self, gate, up):
        """
        Fused SiLU + Mul + Quant.
        
        :param gate: Gate tensor
        :param up: Up-projection tensor
        :return: Tuple of (quantized_output, scale)
        """
        # === FUSED KERNEL START ===
        # SwiGLU activation
        activated = F.silu(gate) * up
        
        # Quantization
        absmax = activated.abs().max()
        scale = self.max_val / absmax.clamp(min=1e-12)
        output = (activated * scale).clamp(-self.max_val, self.max_val).to(torch.float16)
        # === FUSED KERNEL END ===
        
        return output, 1.0 / scale


# GELU + Quant variant
class FusedGELUQuant(nn.Module):
    """
    Fused GELU + Quantization.
    
    Alternative to SiLU for models using GELU (e.g., GPT-style).
    """
    def __init__(self, quant_dtype='fp8_e4m3', approximate='tanh'):
        super(FusedGELUQuant, self).__init__()
        self.quant_dtype = quant_dtype
        self.approximate = approximate
        self.max_val = 448.0 if 'e4m3' in quant_dtype else 57344.0
    
    def forward(self, x):
        """Fused GELU + Quant."""
        # GELU
        if self.approximate == 'tanh':
            activated = F.gelu(x, approximate='tanh')
        else:
            activated = F.gelu(x)
        
        # Quantization
        absmax = activated.abs().max()
        scale = self.max_val / absmax.clamp(min=1e-12)
        output = (activated * scale).clamp(-self.max_val, self.max_val).to(torch.float16)
        
        return output, 1.0 / scale


# Per-channel SiLU + Quant (for better accuracy)
class FusedSiLUQuantPerChannel(nn.Module):
    """
    Fused SiLU + Per-Channel Quantization.
    
    Uses per-channel scaling for better quantization accuracy.
    """
    def __init__(self, num_channels, quant_dtype='fp8_e4m3'):
        super(FusedSiLUQuantPerChannel, self).__init__()
        self.num_channels = num_channels
        self.quant_dtype = quant_dtype
        self.max_val = 448.0 if 'e4m3' in quant_dtype else 57344.0
        
        # Learned per-channel scales (calibrated)
        self.register_buffer('channel_scales', torch.ones(num_channels))
    
    def calibrate(self, sample_data):
        """Calibrate per-channel scales from sample data."""
        with torch.no_grad():
            activated = F.silu(sample_data)
            absmax = activated.abs().max(dim=0).values  # Per-channel
            self.channel_scales = self.max_val / absmax.clamp(min=1e-12)
    
    def forward(self, x):
        """Fused SiLU + Per-channel Quant."""
        activated = F.silu(x)
        output = (activated * self.channel_scales).clamp(-self.max_val, self.max_val)
        return output.to(torch.float16), 1.0 / self.channel_scales


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
    return ['fp8_e4m3']

