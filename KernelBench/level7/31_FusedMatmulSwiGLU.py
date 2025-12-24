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
    Fused Matmul + SwiGLU (fused_swiglu).
    
    Fuses the matrix multiplication with SiLU-and-mul activation in a single operation:
    1. Compute gate = x @ W_gate
    2. Compute up = x @ W_up  
    3. output = SiLU(gate) * up
    
    Unlike separate Linear + activation, this:
    - Uses a single fused weight matrix [W_gate; W_up]
    - Computes both projections in one matmul
    - Applies SiLU * mul inline
    - Single memory read/write pass
    
    Reference: SGLang fused_swiglu, Triton SwiGLU kernels
    """
    def __init__(self, in_features, hidden_features, bias=False):
        """
        :param in_features: Input dimension
        :param hidden_features: SwiGLU intermediate dimension
        :param bias: Use bias in projection
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        
        # Fused weight: [W_gate; W_up] for single matmul
        self.weight = nn.Parameter(
            torch.randn(2 * hidden_features, in_features) * 0.02
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(2 * hidden_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        """
        Fused matmul + SwiGLU forward.
        
        :param x: Input (batch, seq, in_features)
        :return: Output (batch, seq, hidden_features)
        """
        # === FUSED KERNEL START ===
        # Single matmul for both gate and up projections
        gate_up = F.linear(x, self.weight, self.bias)
        
        # Split and apply SwiGLU
        gate, up = gate_up.chunk(2, dim=-1)
        output = F.silu(gate) * up
        # === FUSED KERNEL END ===
        
        return output


# Full MLP with fused matmul + SwiGLU + down projection
class FusedSwiGLUMLP(nn.Module):
    """
    Complete SwiGLU MLP with fused operations.
    
    Fuses:
    1. Gate/Up projection (single matmul)
    2. SiLU * mul activation
    3. Down projection
    """
    def __init__(self, hidden_dim, intermediate_dim, bias=False):
        super(FusedSwiGLUMLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        
        # Fused gate+up
        self.gate_up_proj = nn.Linear(hidden_dim, 2 * intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)
    
    def forward(self, x):
        """Full SwiGLU MLP forward."""
        # === FUSED KERNEL ===
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        # === END FUSED ===
        
        return self.down_proj(hidden)


# Fused with output quantization
class FusedMatmulSwiGLUQuant(nn.Module):
    """
    Fused Matmul + SwiGLU + FP8 Quantization.
    
    For FP8 inference pipelines.
    """
    def __init__(self, in_features, hidden_features, quant_dtype='fp8_e4m3'):
        super(FusedMatmulSwiGLUQuant, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.fp8_max = 448.0 if 'e4m3' in quant_dtype else 57344.0
        
        self.weight = nn.Parameter(
            torch.randn(2 * hidden_features, in_features) * 0.02
        )
    
    def forward(self, x):
        """Fused matmul + SwiGLU + quant."""
        # Fused projection
        gate_up = F.linear(x, self.weight)
        gate, up = gate_up.chunk(2, dim=-1)
        output = F.silu(gate) * up
        
        # FP8 quantization
        absmax = output.abs().max()
        scale = self.fp8_max / absmax.clamp(min=1e-12)
        output_quant = (output * scale).clamp(-self.fp8_max, self.fp8_max).to(torch.float16)
        
        return output_quant, 1.0 / scale


# Triton-style split-K variant
class FusedMatmulSwiGLUSplitK(nn.Module):
    """
    Fused Matmul + SwiGLU with split-K for large matrices.
    
    Splits the K dimension for better GPU utilization on large matrices.
    """
    def __init__(self, in_features, hidden_features, num_splits=4):
        super(FusedMatmulSwiGLUSplitK, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_splits = num_splits
        
        assert in_features % num_splits == 0
        self.split_size = in_features // num_splits
        
        # Weight split along K dimension
        self.weight_splits = nn.ParameterList([
            nn.Parameter(torch.randn(2 * hidden_features, self.split_size) * 0.02)
            for _ in range(num_splits)
        ])
    
    def forward(self, x):
        """Split-K fused matmul + SwiGLU."""
        # Split input along last dimension
        x_splits = x.split(self.split_size, dim=-1)
        
        # Partial matmuls (can be parallelized)
        partial_sums = []
        for x_part, w_part in zip(x_splits, self.weight_splits):
            partial_sums.append(F.linear(x_part, w_part))
        
        # Sum partials
        gate_up = sum(partial_sums)
        
        # SwiGLU
        gate, up = gate_up.chunk(2, dim=-1)
        return F.silu(gate) * up


# Test parameters
batch_size = 32
seq_len = 2048
in_features = 4096
hidden_features = 11008

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
    return [in_features, hidden_features]

