import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Mamba Selective Scan (S6) Fused Kernel.
    
    The core SSM computation for Mamba models, implementing the selective
    state space model with input-dependent dynamics.
    
    Fuses:
    1. Discretization (A, B, C parameters)
    2. Parallel associative scan
    3. Output computation
    
    This is the computational bottleneck of Mamba and requires specialized
    CUDA kernels for efficiency.
    
    Reference: Mamba, Mamba-2
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank='auto'):
        """
        :param d_model: Model dimension
        :param d_state: SSM state dimension
        :param d_conv: Convolution kernel size
        :param expand: Expansion factor
        :param dt_rank: Rank of delta (dt) projection
        """
        super(Model, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == 'auto' else dt_rank
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # A is structured (diagonal)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    
    def _selective_scan(self, u, delta, A, B, C, D):
        """
        Selective scan implementation.
        
        :param u: Input (batch, length, d_inner)
        :param delta: Time delta (batch, length, d_inner)
        :param A: State matrix log (d_inner, d_state)
        :param B: Input matrix (batch, length, d_state)
        :param C: Output matrix (batch, length, d_state)
        :param D: Skip connection (d_inner,)
        :return: Output (batch, length, d_inner)
        """
        batch, length, d_inner = u.shape
        d_state = A.shape[1]
        
        # Discretization
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (batch, length, d_inner, d_state)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        
        # Parallel scan
        # In a fused kernel, this would be an associative scan
        x = torch.zeros(batch, d_inner, d_state, device=u.device)
        ys = []
        
        for i in range(length):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = (x * C[:, i].unsqueeze(1)).sum(-1)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (batch, length, d_inner)
        
        # Add skip connection
        y = y + D * u
        
        return y
    
    def forward(self, x):
        """
        Mamba block forward pass.
        
        :param x: Input (batch, length, d_model)
        :return: Output (batch, length, d_model)
        """
        batch, length, _ = x.shape
        
        # Input projection and split
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        # Causal conv (transpose for Conv1d)
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :length].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # SSM parameter projection
        x_proj = self.x_proj(x_conv)
        delta, B, C = x_proj.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Delta projection
        delta = F.softplus(self.dt_proj(delta))
        
        # Selective scan (fused kernel in practice)
        A = -torch.exp(self.A_log)
        y = self._selective_scan(x_conv, delta, A, B, C, self.D)
        
        # Gate and output
        y = y * F.silu(z)
        output = self.out_proj(y)
        
        return output


# Mamba-2 SSD (State Space Dual) variant
class Mamba2SSD(nn.Module):
    """
    Mamba-2 State Space Dual.
    
    Uses structured matrices for improved efficiency and
    attention-like duality.
    """
    def __init__(self, d_model, d_state=64, nheads=8, d_conv=4, expand=2):
        super(Mamba2SSD, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.nheads = nheads
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.head_dim = self.d_inner // nheads
        
        # Projections
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + nheads, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner + d_state, self.d_inner + d_state,
                               kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner + d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Structured A matrix
        self.A_log = nn.Parameter(torch.randn(nheads))
        self.D = nn.Parameter(torch.ones(nheads))
    
    def forward(self, x):
        batch, length, _ = x.shape
        
        proj = self.in_proj(x)
        x_inner, z, B, C, dt = proj.split(
            [self.d_inner, self.d_inner, self.d_state, self.d_state, self.nheads], dim=-1
        )
        
        # Conv
        xBC = torch.cat([x_inner, B], dim=-1).transpose(1, 2)
        xBC = self.conv1d(xBC)[:, :, :length].transpose(1, 2)
        x_inner = F.silu(xBC[..., :self.d_inner])
        
        # SSD computation (simplified)
        y = x_inner * F.silu(z)
        
        return self.out_proj(y)


# Test parameters
batch_size = 8
seq_len = 2048
d_model = 2048

def get_inputs():
    return [torch.randn(batch_size, seq_len, d_model)]

def get_init_inputs():
    return [d_model]

