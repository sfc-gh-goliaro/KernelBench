import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused GeGLU/SwiGLU/ReGLU Activation.
    
    Gated Linear Units with various activation functions:
    - GeGLU: GELU(xW_g) * (xW_u)
    - SwiGLU: SiLU(xW_g) * (xW_u)
    - ReGLU: ReLU(xW_g) * (xW_u)
    
    This fuses the gate/up projection, activation, and element-wise multiply.
    
    Reference: GLU Variants paper, LLaMA/Mistral/Gemma architectures
    """
    def __init__(self, hidden_dim, intermediate_dim, activation='silu', bias=False):
        """
        :param hidden_dim: Input hidden dimension
        :param intermediate_dim: Intermediate (gate/up) dimension
        :param activation: Gate activation ('gelu', 'silu', 'relu')
        :param bias: Use bias in projections
        """
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.activation = activation
        
        # Separate gate and up projections
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
    
    def forward(self, x):
        """
        Fused GLU forward.
        
        :param x: Input (batch, seq, hidden_dim)
        :return: GLU output (batch, seq, intermediate_dim)
        """
        # === FUSED KERNEL START ===
        # Gate projection
        gate = self.gate_proj(x)
        
        # Up projection
        up = self.up_proj(x)
        
        # Apply activation to gate
        if self.activation == 'gelu':
            gate = F.gelu(gate)
        elif self.activation == 'silu' or self.activation == 'swish':
            gate = F.silu(gate)
        elif self.activation == 'relu':
            gate = F.relu(gate)
        elif self.activation == 'sigmoid':
            gate = torch.sigmoid(gate)
        
        # Element-wise multiply
        output = gate * up
        # === FUSED KERNEL END ===
        
        return output


# Packed weight variant (single matmul)
class FusedPackedGLU(nn.Module):
    """
    Fused GLU with packed gate/up weights.
    
    Uses a single matrix multiplication for both gate and up projections.
    """
    def __init__(self, hidden_dim, intermediate_dim, activation='silu', bias=False):
        super(FusedPackedGLU, self).__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.activation = activation
        
        # Packed weights: [gate_weights; up_weights]
        self.gate_up_proj = nn.Linear(hidden_dim, 2 * intermediate_dim, bias=bias)
    
    def forward(self, x):
        """
        Fused GLU with packed projection.
        """
        # === FUSED KERNEL START ===
        # Single matmul for both gate and up
        gate_up = self.gate_up_proj(x)
        
        # Split
        gate, up = gate_up.chunk(2, dim=-1)
        
        # Activation and multiply
        if self.activation == 'gelu':
            output = F.gelu(gate) * up
        elif self.activation == 'silu':
            output = F.silu(gate) * up
        elif self.activation == 'relu':
            output = F.relu(gate) * up
        else:
            output = gate * up
        # === FUSED KERNEL END ===
        
        return output


# Full MLP with fused GLU
class FusedGLUMLP(nn.Module):
    """
    Full GLU MLP block with fused operations.
    
    Combines:
    1. Gate/Up projections (optionally packed)
    2. Activation
    3. Down projection
    """
    def __init__(self, hidden_dim, intermediate_dim, activation='silu', 
                 bias=False, packed=True):
        super(FusedGLUMLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.activation = activation
        self.packed = packed
        
        if packed:
            self.gate_up_proj = nn.Linear(hidden_dim, 2 * intermediate_dim, bias=bias)
        else:
            self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
            self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)
    
    def forward(self, x):
        """
        Full GLU MLP forward.
        """
        if self.packed:
            gate_up = self.gate_up_proj(x)
            gate, up = gate_up.chunk(2, dim=-1)
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
        
        # Fused activation + multiply
        if self.activation == 'silu':
            hidden = F.silu(gate) * up
        elif self.activation == 'gelu':
            hidden = F.gelu(gate) * up
        else:
            hidden = gate * up
        
        return self.down_proj(hidden)


# Variant with different activations for experimentation
class FusedFlexibleGLU(nn.Module):
    """
    Flexible GLU with various activation combinations.
    
    Supports different activations and optionally applies activation to value too.
    """
    def __init__(self, hidden_dim, intermediate_dim, 
                 gate_activation='silu', value_activation=None):
        super(FusedFlexibleGLU, self).__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.gate_activation = gate_activation
        self.value_activation = value_activation
        
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
    
    def _get_activation(self, name):
        if name == 'silu':
            return F.silu
        elif name == 'gelu':
            return F.gelu
        elif name == 'relu':
            return F.relu
        elif name == 'tanh':
            return torch.tanh
        elif name == 'sigmoid':
            return torch.sigmoid
        else:
            return lambda x: x
    
    def forward(self, x):
        gate = self.gate_proj(x)
        value = self.value_proj(x)
        
        # Apply activations
        gate_act = self._get_activation(self.gate_activation)
        gate = gate_act(gate)
        
        if self.value_activation:
            value_act = self._get_activation(self.value_activation)
            value = value_act(value)
        
        return gate * value


# Test parameters
batch_size = 32
seq_len = 2048
hidden_dim = 4096
intermediate_dim = 11008

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_dim)]

def get_init_inputs():
    return [hidden_dim, intermediate_dim, 'silu']

