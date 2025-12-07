import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused AllReduce + RMSNorm + FP8 Quantization.
    
    The complete fusion for tensor-parallel inference with FP8:
    1. All-reduce partial sums from TP ranks
    2. Apply RMSNorm
    3. Quantize output to FP8 for next layer
    
    This eliminates storing FP16/BF16 normalized values and directly
    produces FP8 for subsequent quantized GEMM.
    
    Reference: vLLM FP8 quantization, SGLang, TensorRT-LLM
    """
    def __init__(self, hidden_dim, world_size, eps=1e-6, 
                 quant_dtype='fp8_e4m3', dynamic_scale=True):
        """
        :param hidden_dim: Hidden dimension
        :param world_size: Tensor parallel world size
        :param eps: RMSNorm epsilon
        :param quant_dtype: 'fp8_e4m3' or 'fp8_e5m2' or 'int8'
        :param dynamic_scale: Use per-tensor dynamic scaling
        """
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.world_size = world_size
        self.eps = eps
        self.quant_dtype = quant_dtype
        self.dynamic_scale = dynamic_scale
        
        # RMSNorm weight
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        
        # Static scale (if not dynamic)
        if not dynamic_scale:
            self.register_buffer('scale', torch.tensor(1.0))
    
    def _quantize_fp8(self, x, dtype='fp8_e4m3'):
        """Quantize to FP8 format."""
        if dtype == 'fp8_e4m3':
            # E4M3: range [-448, 448], good for forward pass
            max_val = 448.0
        else:  # fp8_e5m2
            # E5M2: range [-57344, 57344], good for gradients
            max_val = 57344.0
        
        # Compute scale
        if self.dynamic_scale:
            absmax = x.abs().max()
            scale = max_val / absmax.clamp(min=1e-12)
        else:
            scale = self.scale
        
        # Quantize (simulate FP8 with clamp + round)
        x_scaled = x * scale
        x_quant = x_scaled.clamp(-max_val, max_val)
        
        # In actual FP8, this would be stored in 8-bit format
        # Here we keep float but with FP8 precision
        x_quant = x_quant.to(torch.float16)  # Simulate reduced precision
        
        return x_quant, 1.0 / scale
    
    def _quantize_int8(self, x):
        """Quantize to INT8."""
        absmax = x.abs().max()
        scale = 127.0 / absmax.clamp(min=1e-12)
        
        x_quant = (x * scale).round().clamp(-128, 127).to(torch.int8)
        
        return x_quant, 1.0 / scale
    
    def forward(self, x, all_rank_partials=None):
        """
        Fused AllReduce + RMSNorm + Quantization.
        
        :param x: Local partial result
        :param all_rank_partials: Partials from all ranks
        :return: Tuple of (quantized_output, scale)
        """
        # === FUSED KERNEL START ===
        # Step 1: All-Reduce
        if all_rank_partials is not None:
            reduced = sum(all_rank_partials)
        else:
            reduced = x
        
        # Step 2: RMSNorm
        variance = reduced.pow(2).mean(dim=-1, keepdim=True)
        normalized = reduced * torch.rsqrt(variance + self.eps) * self.weight
        
        # Step 3: Quantization
        if 'fp8' in self.quant_dtype:
            output, scale = self._quantize_fp8(normalized, self.quant_dtype)
        else:
            output, scale = self._quantize_int8(normalized)
        # === FUSED KERNEL END ===
        
        return output, scale


# With residual variant
class FusedAllReduceRMSNormQuantResidual(nn.Module):
    """
    Fused AllReduce + Residual + RMSNorm + FP8 Quant.
    
    Complete post-sublayer fusion with quantization.
    """
    def __init__(self, hidden_dim, world_size, eps=1e-6):
        super(FusedAllReduceRMSNormQuantResidual, self).__init__()
        self.hidden_dim = hidden_dim
        self.world_size = world_size
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, partial, residual, all_rank_partials=None):
        """
        Full fusion with residual.
        """
        # All-Reduce
        if all_rank_partials is not None:
            reduced = sum(all_rank_partials)
        else:
            reduced = partial
        
        # Residual + RMSNorm
        hidden = reduced + residual
        variance = hidden.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden * torch.rsqrt(variance + self.eps) * self.weight
        
        # FP8 Quantization
        absmax = normalized.abs().max()
        scale = 448.0 / absmax.clamp(min=1e-12)
        output = (normalized * scale).clamp(-448, 448).to(torch.float16)
        
        return output, 1.0 / scale


# Test parameters
batch_size = 32
seq_len = 2048
hidden_dim = 4096
world_size = 8

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_dim)
    return [x]

def get_init_inputs():
    return [hidden_dim, world_size]

