import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class MambaMixer(nn.Module):
    """
    Mamba-1 Selective State Space Model (S6).
    
    Implements selective scan with input-dependent state transitions.
    The key innovation is making A, B, C parameters input-dependent
    rather than fixed, enabling content-aware sequence modeling.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        
        # Input projection (x -> expanded)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Conv for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # SSM parameters projection
        # Projects to dt, B, C
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        
        # dt projection
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # A parameter (log-space for stability)
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D "skip connection" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def selective_scan(self, u, delta, A, B, C, D):
        """
        Selective scan algorithm.
        
        Args:
            u: Input (batch, seq_len, d_inner)
            delta: Time step (batch, seq_len, d_inner)
            A: State transition (d_inner, d_state)
            B: Input-to-state (batch, seq_len, d_state)
            C: State-to-output (batch, seq_len, d_state)
            D: Skip connection (d_inner,)
        """
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[1]
        
        # Discretize A and B
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D, N)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (B, L, D, N)
        
        # Scan
        x = torch.zeros(batch_size, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        
        for i in range(seq_len):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = (x * C[:, i].unsqueeze(1)).sum(-1)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)
        y = y + u * D
        
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        # Conv1d for local context
        x_conv = x_inner.transpose(1, 2)  # (B, D, L)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Trim padding
        x_conv = x_conv.transpose(1, 2)  # (B, L, D)
        x_inner = F.silu(x_conv)
        
        # SSM parameters
        x_dbl = self.x_proj(x_inner)
        dt = x_dbl[..., :1]
        B = x_dbl[..., 1:self.d_state + 1]
        C = x_dbl[..., self.d_state + 1:]
        
        # dt projection with softplus
        delta = F.softplus(self.dt_proj(dt)).squeeze(-1)
        
        # Get A (negative for stability)
        A = -torch.exp(self.A_log)
        
        # Selective scan
        y = self.selective_scan(x_inner, delta, A, B, C, self.D)
        
        # Gate and output
        y = y * F.silu(z)
        return self.out_proj(y)


class Model(nn.Module):
    """
    Mamba-1 Block
    
    The core repeated block in Mamba-1 architecture.
    Used by: Mamba-130M to Mamba-2.8B
    
    Architecture:
        x -> RMSNorm -> MambaMixer (SSM) -> + residual
    
    Key components:
    - No attention mechanism (attention-free)
    - Selective state space model (S6)
    - Linear time complexity O(n)
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = MambaMixer(d_model, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


# Benchmark configuration (Mamba-2.8B dimensions)
batch_size = 8
seq_len = 2048
d_model = 2560
d_state = 16
d_conv = 4
expand = 2

def get_inputs():
    return [torch.randn(batch_size, seq_len, d_model)]

def get_init_inputs():
    return [d_model, d_state, d_conv, expand]

