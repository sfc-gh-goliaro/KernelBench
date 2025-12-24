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
    SwiGLU (Swish-Gated Linear Unit) activation function.
    
    Combines Swish activation with gating mechanism. Used in
    LLaMA, Mistral, Gemma, and many modern LLMs.
    
    Based on: "GLU Variants Improve Transformer" (Noam Shazeer)
    """
    def __init__(self, dim, hidden_dim, bias=False):
        """
        :param dim: Input/output dimension
        :param hidden_dim: Hidden dimension (typically ~2.7x dim for 8/3 ratio)
        :param bias: Whether to use bias in linear layers
        """
        super(Model, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # Gate and up projections (split or separate)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
    
    def forward(self, x):
        """
        Forward pass for SwiGLU.
        
        SwiGLU(x) = (Swish(xW_gate) ⊙ xW_up) W_down
        
        :param x: Input tensor (..., dim)
        :return: Output tensor (..., dim)
        """
        # Gate with Swish activation
        gate = F.silu(self.gate_proj(x))
        
        # Up projection
        up = self.up_proj(x)
        
        # Element-wise product
        hidden = gate * up
        
        # Down projection
        return self.down_proj(hidden)


# Variant: GeGLU (GELU-gated)
class GeGLU(nn.Module):
    """
    GeGLU (GELU-Gated Linear Unit) variant.
    """
    def __init__(self, dim, hidden_dim, bias=False):
        super(GeGLU, self).__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
    
    def forward(self, x):
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# Variant: ReGLU (ReLU-gated)
class ReGLU(nn.Module):
    """
    ReGLU (ReLU-Gated Linear Unit) variant.
    """
    def __init__(self, dim, hidden_dim, bias=False):
        super(ReGLU, self).__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
    
    def forward(self, x):
        gate = F.relu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# Test parameters
batch_size = 32
seq_len = 512
dim = 4096
hidden_dim = 11008  # LLaMA-7B FFN dimension

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    return [torch.randn(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, hidden_dim]

