import torch
import torch.nn as nn
import torch.nn.functional as F
import math


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Hyena Operator: Subquadratic replacement for attention.
    
    Uses long convolutions and element-wise gating to achieve
    O(N log N) complexity while maintaining expressiveness.
    
    Based on: "Hyena Hierarchy: Towards Larger Convolutional Language Models"
    """
    def __init__(self, dim, seq_len, order=2, filter_order=64):
        """
        :param dim: Model dimension
        :param seq_len: Sequence length
        :param order: Order of the Hyena operator (number of gating layers)
        :param filter_order: Order of the implicit filter
        """
        super(Model, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.order = order
        self.filter_order = filter_order
        
        # Input projections for gating
        self.in_proj = nn.Linear(dim, (order + 1) * dim)
        
        # Implicit filter parameters (learnable positional filter)
        self.filter_fn = nn.Sequential(
            nn.Linear(1, filter_order),
            nn.SiLU(),
            nn.Linear(filter_order, filter_order),
            nn.SiLU(),
            nn.Linear(filter_order, dim, bias=False)
        )
        
        # Short convolution (local processing)
        self.short_filter = nn.Conv1d(
            (order + 1) * dim,
            (order + 1) * dim,
            kernel_size=3,
            padding=1,
            groups=(order + 1) * dim
        )
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Decay for exponential windowing
        self.decay = nn.Parameter(torch.linspace(0.01, 0.1, dim))
        
        # Positional encoding for filter
        self.register_buffer('positions', 
            torch.linspace(0, 1, seq_len).unsqueeze(-1))
    
    def generate_filter(self, seq_len):
        """Generate the implicit long convolution filter."""
        positions = self.positions[:seq_len]
        
        # Generate filter through MLP
        h = self.filter_fn(positions)  # (seq_len, dim)
        
        # Apply exponential decay
        decay = torch.exp(-self.decay.unsqueeze(0) * 
                         torch.arange(seq_len, device=h.device).unsqueeze(-1))
        h = h * decay
        
        return h.transpose(0, 1)  # (dim, seq_len)
    
    def fft_conv(self, u, h):
        """
        FFT-based long convolution.
        
        :param u: Input (batch, dim, seq)
        :param h: Filter (dim, seq)
        :return: Convolved output
        """
        seq_len = u.shape[-1]
        
        # Pad for circular convolution
        fft_size = 2 * seq_len
        
        # FFT of input and filter
        u_f = torch.fft.rfft(u, n=fft_size, dim=-1)
        h_f = torch.fft.rfft(h, n=fft_size, dim=-1)
        
        # Multiply in frequency domain
        y_f = u_f * h_f.unsqueeze(0)
        
        # Inverse FFT and truncate
        y = torch.fft.irfft(y_f, n=fft_size, dim=-1)
        y = y[..., :seq_len]
        
        return y
    
    def forward(self, x):
        """
        Forward pass for Hyena operator.
        
        :param x: Input tensor of shape (batch_size, seq_len, dim)
        :return: Output tensor of shape (batch_size, seq_len, dim)
        """
        batch_size, seq_len, dim = x.shape
        
        # Project input to multiple components
        z = self.in_proj(x)  # (batch, seq, (order+1)*dim)
        z = z.transpose(1, 2)  # (batch, (order+1)*dim, seq)
        
        # Apply short convolution
        z = self.short_filter(z)
        
        # Split into v (value) and gates x_1, x_2, ... x_order
        z = z.view(batch_size, self.order + 1, dim, seq_len)
        v = z[:, 0]  # (batch, dim, seq)
        gates = z[:, 1:]  # (batch, order, dim, seq)
        
        # Generate long convolution filter
        h = self.generate_filter(seq_len)  # (dim, seq)
        
        # Iteratively apply gating and convolution
        y = v
        for i in range(self.order):
            # Element-wise gating
            y = y * gates[:, i]
            
            # Long convolution via FFT
            y = self.fft_conv(y, h)
        
        # Transpose back
        y = y.transpose(1, 2)  # (batch, seq, dim)
        
        return self.out_proj(y)


# Test parameters
batch_size = 16
seq_len = 1024
dim = 512
order = 2

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
    return [dim, seq_len, order]

