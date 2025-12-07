import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused Causal Conv1d + Activation (Mamba Pre-processing).
    
    The conv + activation step before selective scan in Mamba.
    Fuses:
    1. Causal 1D convolution (depthwise)
    2. SiLU/Swish activation
    
    This pattern appears in Mamba, RWKV, and other SSM-based models.
    Fusion eliminates intermediate memory traffic.
    
    Reference: Mamba causal_conv1d_fn
    """
    def __init__(self, d_inner, d_conv=4, activation='silu'):
        """
        :param d_inner: Number of channels
        :param d_conv: Convolution kernel size
        :param activation: Activation function ('silu', 'gelu', 'relu')
        """
        super(Model, self).__init__()
        self.d_inner = d_inner
        self.d_conv = d_conv
        self.activation = activation
        
        # Depthwise causal convolution
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # Causal padding
            groups=d_inner,
            bias=True
        )
        
        # For fused kernel, weight is stored as (d_inner, 1, d_conv)
        # and we process in (batch, d_inner, length) format
        
    def forward(self, x, conv_state=None):
        """
        Fused causal conv1d + activation.
        
        :param x: Input (batch, length, d_inner) or (batch, d_inner, length)
        :param conv_state: Previous convolution state for incremental decoding
        :return: Activated output (batch, length, d_inner)
        """
        # Handle input format
        if x.dim() == 3 and x.shape[-1] == self.d_inner:
            # (batch, length, d_inner) -> (batch, d_inner, length)
            x = x.transpose(1, 2)
            transposed = True
        else:
            transposed = False
        
        batch, channels, length = x.shape
        
        # === FUSED KERNEL START ===
        if conv_state is not None:
            # Incremental decoding: use cached state
            # Prepend state to input
            x = torch.cat([conv_state, x], dim=-1)
        
        # Causal convolution
        y = self.conv1d(x)
        
        # Truncate to maintain causality
        if conv_state is not None:
            y = y[..., -1:]  # Only last position for incremental
            new_state = x[..., -(self.d_conv - 1):]
        else:
            y = y[..., :length]  # Remove future padding
            new_state = x[..., -(self.d_conv - 1):] if length >= self.d_conv - 1 else x
        
        # Activation
        if self.activation == 'silu':
            y = F.silu(y)
        elif self.activation == 'gelu':
            y = F.gelu(y)
        elif self.activation == 'relu':
            y = F.relu(y)
        # === FUSED KERNEL END ===
        
        # Restore format if needed
        if transposed:
            y = y.transpose(1, 2)
        
        if conv_state is not None:
            return y, new_state
        return y


# Variant with separate update for incremental decoding
class FusedCausalConv1dUpdate(nn.Module):
    """
    Fused Causal Conv1d with explicit state update.
    
    Optimized for token-by-token generation.
    """
    def __init__(self, d_inner, d_conv=4, activation='silu'):
        super(FusedCausalConv1dUpdate, self).__init__()
        self.d_inner = d_inner
        self.d_conv = d_conv
        self.activation = activation
        
        # Store weight as (d_inner, d_conv) for efficient access
        self.weight = nn.Parameter(torch.randn(d_inner, d_conv) * 0.02)
        self.bias = nn.Parameter(torch.zeros(d_inner))
    
    def forward(self, x, conv_state):
        """
        Single-token update with state.
        
        :param x: New input (batch, d_inner) - single token
        :param conv_state: State (batch, d_inner, d_conv-1)
        :return: Output (batch, d_inner), new_state (batch, d_inner, d_conv-1)
        """
        batch = x.shape[0]
        
        # === FUSED KERNEL START ===
        # Shift state and add new input
        new_state = torch.cat([conv_state[..., 1:], x.unsqueeze(-1)], dim=-1)
        
        # Convolution as dot product
        y = (new_state * self.weight).sum(dim=-1) + self.bias
        
        # Activation
        if self.activation == 'silu':
            y = F.silu(y)
        elif self.activation == 'gelu':
            y = F.gelu(y)
        # === FUSED KERNEL END ===
        
        return y, new_state
    
    def init_state(self, batch_size, device):
        """Initialize convolution state."""
        return torch.zeros(batch_size, self.d_inner, self.d_conv - 1, device=device)


# Grouped Causal Conv1d (for multi-head variants)
class FusedGroupedCausalConv1d(nn.Module):
    """
    Grouped Causal Conv1d + Activation.
    
    Supports different groups for different SSM heads.
    """
    def __init__(self, d_inner, d_conv=4, ngroups=1, activation='silu'):
        super(FusedGroupedCausalConv1d, self).__init__()
        self.d_inner = d_inner
        self.d_conv = d_conv
        self.ngroups = ngroups
        self.activation = activation
        
        assert d_inner % ngroups == 0
        
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner
        )
    
    def forward(self, x):
        """Grouped causal conv forward."""
        x = x.transpose(1, 2)  # (batch, d_inner, length)
        y = self.conv1d(x)[..., :x.shape[-1]]
        
        if self.activation == 'silu':
            y = F.silu(y)
        
        return y.transpose(1, 2)


# Test parameters
batch_size = 32
seq_len = 2048
d_inner = 4096
d_conv = 4

def get_inputs():
    return [torch.randn(batch_size, seq_len, d_inner)]

def get_init_inputs():
    return [d_inner, d_conv]

