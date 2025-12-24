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
    Fused RMSNorm + SwiGLU MLP.
    
    Combines RMS normalization with the gated MLP block in a single kernel.
    This is a critical hot path in modern LLMs like LLaMA, Mistral, Qwen.
    
    Fusing eliminates intermediate memory traffic between:
    1. RMSNorm output write/read
    2. Gate/Up projection intermediate
    
    Reference: vLLM, TensorRT-LLM fused kernels
    """
    def __init__(self, hidden_dim, intermediate_dim, eps=1e-6):
        """
        :param hidden_dim: Model hidden dimension
        :param intermediate_dim: MLP intermediate dimension
        :param eps: RMSNorm epsilon
        """
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.eps = eps
        
        # RMSNorm parameters
        self.rms_weight = nn.Parameter(torch.ones(hidden_dim))
        
        # SwiGLU projections
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
    
    def forward(self, x):
        """
        Fused RMSNorm + SwiGLU forward.
        
        :param x: Input tensor (batch, seq, hidden_dim)
        :return: Output tensor (batch, seq, hidden_dim)
        """
        # === FUSED KERNEL START ===
        # Step 1: RMSNorm
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        hidden = x_norm * self.rms_weight
        
        # Step 2: SwiGLU - gate and up projections can be computed together
        gate = self.gate_proj(hidden)
        up = self.up_proj(hidden)
        
        # Step 3: SiLU activation on gate, element-wise multiply
        activated = F.silu(gate) * up
        # === FUSED KERNEL END ===
        
        # Down projection
        output = self.down_proj(activated)
        
        return output


# Variant with fused gate/up projection (single matmul)
class FusedRMSNormSwiGLUPacked(nn.Module):
    """
    Fused RMSNorm + SwiGLU with packed gate/up weights.
    
    Uses a single fused weight matrix for gate and up projections.
    """
    def __init__(self, hidden_dim, intermediate_dim, eps=1e-6):
        super(FusedRMSNormSwiGLUPacked, self).__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.eps = eps
        
        self.rms_weight = nn.Parameter(torch.ones(hidden_dim))
        
        # Packed gate+up projection (2 * intermediate_dim)
        self.gate_up_proj = nn.Linear(hidden_dim, 2 * intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
    
    def forward(self, x):
        """Fused forward with packed weights."""
        # RMSNorm
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        hidden = x_norm * self.rms_weight
        
        # Single matmul for gate and up
        gate_up = self.gate_up_proj(hidden)
        gate, up = gate_up.chunk(2, dim=-1)
        
        # SwiGLU activation
        activated = F.silu(gate) * up
        
        return self.down_proj(activated)


# Test parameters
batch_size = 32
seq_len = 2048
hidden_dim = 4096
intermediate_dim = 11008  # LLaMA-7B style

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
    return [hidden_dim, intermediate_dim]

