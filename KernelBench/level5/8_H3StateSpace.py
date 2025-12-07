import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    H3 (Hungry Hungry Hippos) layer combining SSM with multiplicative gating.
    
    H3 extends S4 with input-dependent gating to improve expressiveness
    while maintaining efficient long convolution computation.
    
    Based on: "Hungry Hungry Hippos: Towards Language Modeling with State Space Models"
    """
    def __init__(self, dim, state_dim, seq_len, num_heads=1):
        """
        :param dim: Model dimension
        :param state_dim: State space dimension per head
        :param seq_len: Sequence length
        :param num_heads: Number of SSM heads
        """
        super(Model, self).__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Input projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # SSM parameters (diagonal S4D)
        # Use diagonal approximation for efficiency
        self.log_A_real = nn.Parameter(torch.randn(num_heads, state_dim) * 0.1)
        self.A_imag = nn.Parameter(torch.randn(num_heads, state_dim) * 0.1)
        
        # B and C per head
        self.B = nn.Parameter(torch.randn(num_heads, state_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(num_heads, state_dim) * 0.01)
        
        # Discretization step
        self.log_dt = nn.Parameter(torch.rand(num_heads) * -2)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Short convolution for local patterns
        self.short_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        
    def _compute_kernel(self, seq_len):
        """Compute SSM kernel using diagonal state space."""
        dt = torch.exp(self.log_dt).unsqueeze(-1)  # (heads, 1)
        
        # Diagonal A (complex)
        A_real = -torch.exp(self.log_A_real)  # Ensure stability
        A = A_real + 1j * self.A_imag  # (heads, state_dim)
        
        # Discretize: A_bar = exp(A * dt)
        A_bar = torch.exp(A * dt)  # (heads, state_dim)
        
        # Compute powers of A_bar
        positions = torch.arange(seq_len, device=A.device).float()
        A_powers = A_bar.unsqueeze(-1) ** positions.unsqueeze(0).unsqueeze(0)  # (heads, state, seq)
        
        # Kernel: K = C * A^n * B (element-wise for diagonal)
        kernel = self.C.unsqueeze(-1) * A_powers * self.B.unsqueeze(-1)
        kernel = kernel.sum(dim=1).real  # (heads, seq)
        
        return kernel
    
    def _ssm_conv(self, x, kernel):
        """Apply SSM convolution via FFT."""
        seq_len = x.shape[-1]
        fft_size = 2 * seq_len
        
        x_f = torch.fft.rfft(x, n=fft_size, dim=-1)
        k_f = torch.fft.rfft(kernel, n=fft_size, dim=-1)
        
        y_f = x_f * k_f.unsqueeze(0).unsqueeze(2)  # (batch, heads, head_dim, freq)
        y = torch.fft.irfft(y_f, n=fft_size, dim=-1)[..., :seq_len]
        
        return y
    
    def forward(self, x):
        """
        Forward pass for H3 layer.
        
        :param x: Input tensor of shape (batch_size, seq_len, dim)
        :return: Output tensor of shape (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Short convolution on Q
        q = self.short_conv(q.transpose(1, 2)).transpose(1, 2)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for SSM: (batch, heads, head_dim, seq)
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)
        
        # Compute SSM kernel
        kernel = self._compute_kernel(seq_len)  # (heads, seq)
        
        # Apply SSM to K*V
        kv = k * v
        ssm_out = self._ssm_conv(kv, kernel)  # (batch, heads, head_dim, seq)
        
        # Multiplicative gating with Q
        out = q * ssm_out
        
        # Reshape back
        out = out.permute(0, 3, 1, 2).reshape(batch_size, seq_len, self.dim)
        
        return self.out_proj(out)


# Test parameters
batch_size = 16
seq_len = 1024
dim = 512
state_dim = 64
num_heads = 8

def get_inputs():
    return [torch.randn(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, state_dim, seq_len, num_heads]

