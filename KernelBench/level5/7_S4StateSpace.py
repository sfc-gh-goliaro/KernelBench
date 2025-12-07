import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    S4 (Structured State Space) model layer.
    
    Efficiently models long-range dependencies using structured state space
    models with HiPPO initialization and diagonal approximation.
    
    Based on: "Efficiently Modeling Long Sequences with Structured State Spaces"
    """
    def __init__(self, dim, state_dim, seq_len, dt_min=0.001, dt_max=0.1):
        """
        :param dim: Model dimension (number of channels)
        :param state_dim: State space dimension (N in the paper)
        :param seq_len: Sequence length
        :param dt_min: Minimum discretization step
        :param dt_max: Maximum discretization step
        """
        super(Model, self).__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.seq_len = seq_len
        
        # Initialize A matrix using HiPPO-LegS
        A = self._make_hippo(state_dim)
        self.register_buffer('A', A)
        
        # Learnable parameters
        # Log of discretization step (dt)
        log_dt = torch.rand(dim) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        
        # B and C matrices
        self.B = nn.Parameter(torch.randn(dim, state_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(dim, state_dim) * 0.01)
        
        # D matrix (skip connection)
        self.D = nn.Parameter(torch.ones(dim))
        
        # Precompute powers of A for efficient convolution
        self.register_buffer('A_powers', None)
        
    def _make_hippo(self, N):
        """Create HiPPO-LegS matrix."""
        # HiPPO matrix for Legendre polynomials
        P = torch.sqrt(1 + 2 * torch.arange(N))
        A = P.unsqueeze(1) * P.unsqueeze(0)
        A = torch.tril(A) - torch.diag(torch.arange(N) + 0.5)
        return -A
    
    def _discretize(self, dt):
        """
        Discretize continuous-time SSM using bilinear transform.
        
        Converts (A, B, C) to discrete (A_bar, B_bar, C)
        """
        I = torch.eye(self.state_dim, device=dt.device)
        dt = dt.unsqueeze(-1).unsqueeze(-1)  # (dim, 1, 1)
        A = self.A.unsqueeze(0)  # (1, N, N)
        
        # Bilinear transform
        A_bar = torch.linalg.solve(I - dt/2 * A, I + dt/2 * A)
        B_bar = torch.linalg.solve(I - dt/2 * A, dt * self.B.unsqueeze(-1)).squeeze(-1)
        
        return A_bar, B_bar
    
    def _compute_kernel(self, seq_len):
        """
        Compute convolution kernel from SSM parameters.
        
        K[i] = C @ A_bar^i @ B_bar
        """
        dt = torch.exp(self.log_dt)
        A_bar, B_bar = self._discretize(dt)  # (dim, N, N), (dim, N)
        
        # Compute powers of A_bar and apply to B_bar
        kernel = []
        x = B_bar.unsqueeze(-1)  # (dim, N, 1)
        
        for i in range(seq_len):
            # C @ A^i @ B
            k_i = torch.bmm(self.C.unsqueeze(1), x).squeeze(-1).squeeze(-1)  # (dim,)
            kernel.append(k_i)
            x = torch.bmm(A_bar, x)
        
        return torch.stack(kernel, dim=1)  # (dim, seq_len)
    
    def forward(self, x):
        """
        Forward pass for S4 layer.
        
        :param x: Input tensor of shape (batch_size, seq_len, dim)
        :return: Output tensor of shape (batch_size, seq_len, dim)
        """
        batch_size, seq_len, dim = x.shape
        
        # Compute SSM kernel
        kernel = self._compute_kernel(seq_len)  # (dim, seq_len)
        
        # Transpose for convolution
        x = x.transpose(1, 2)  # (batch, dim, seq)
        
        # FFT convolution
        fft_size = 2 * seq_len
        x_f = torch.fft.rfft(x, n=fft_size, dim=-1)
        k_f = torch.fft.rfft(kernel, n=fft_size, dim=-1)
        
        y_f = x_f * k_f.unsqueeze(0)
        y = torch.fft.irfft(y_f, n=fft_size, dim=-1)[..., :seq_len]
        
        # Add skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(-1) * x
        
        return y.transpose(1, 2)  # (batch, seq, dim)


# Test parameters
batch_size = 16
seq_len = 512
dim = 256
state_dim = 64

def get_inputs():
    return [torch.randn(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, state_dim, seq_len]

