"""
Mamba-2 State Space Model

Implements Mamba-2 with State-Space Duality (SSD):
- Selective scan mechanism
- Multi-head structure for state-space computation
- Linear-time complexity O(n)

Variants from Table 5:
- Mamba-2-1.3B: d_model=2048, d_state=128, num_layers=48
- Mamba-2-2.7B: d_model=2560, d_state=128, num_layers=64

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

# Import level1 operators
# Operators that need wrapping
from ..level1.normalization._4_RMSNorm import Model as RMSNormL1
# Operators used directly
from ..level1.activations._7_Swish import Model as Swish
from ..level1.activations._11_Softplus import Model as Softplus


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "1.3B": "state-spaces/mamba2-1.3b",
    "2.7B": "state-spaces/mamba2-2.7b",
}


# ============================================================================
# Wrapper classes for level1 operators that need adaptation
# ============================================================================

class RMSNorm(nn.Module):
    """RMS Normalization with learnable weight, using level1 operator."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self._rms_norm = RMSNormL1(hidden_size, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self._rms_norm(x.transpose(1, -1)).transpose(1, -1)
        return normalized * self.weight


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class Mamba2Mixer(nn.Module):
    """
    Mamba-2 mixer with State-Space Duality (SSD) using level1 operators.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.headdim = headdim
        self.ngroups = ngroups
        self.nheads = self.d_inner // headdim
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Conv for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # SSM parameters projection
        self.x_proj = nn.Linear(self.d_inner, (self.nheads + 2 * ngroups) * d_state, bias=False)
        
        # dt projection
        self.dt_proj = nn.Linear(self.nheads * d_state // ngroups, self.nheads, bias=True)
        
        # A parameter (log-space for stability)
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(self.nheads, -1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D "skip connection" parameter
        self.D = nn.Parameter(torch.ones(self.nheads))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Use level1 operators directly
        self.swish = Swish()
        self.softplus = Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        # Conv1d for local context
        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_inner = self.swish(x_conv)
        
        # SSM parameters
        x_ssm = self.x_proj(x_inner)
        
        # Split into components
        dt_size = self.nheads * self.d_state // self.ngroups
        bc_size = 2 * self.ngroups * self.d_state
        
        dt_bc = x_ssm[..., :dt_size + bc_size]
        dt_input = dt_bc[..., :dt_size]
        bc = dt_bc[..., dt_size:]
        
        B = bc[..., :self.ngroups * self.d_state].view(batch_size, seq_len, self.ngroups, self.d_state)
        C = bc[..., self.ngroups * self.d_state:].view(batch_size, seq_len, self.ngroups, self.d_state)
        
        # dt projection with softplus using level1 operator
        delta = self.softplus(self.dt_proj(dt_input))
        
        # Get A (negative for stability)
        A = -torch.exp(self.A_log)
        
        # Selective scan (simplified for clarity)
        y = self._selective_scan(x_inner, delta, A, B, C)
        
        # Gate and output using level1 Swish
        y = y * self.swish(z)
        return self.out_proj(y)

    def _selective_scan(self, u, delta, A, B, C):
        """Simplified selective scan implementation."""
        batch_size, seq_len, d_inner = u.shape
        
        u = u.view(batch_size, seq_len, self.nheads, self.headdim)
        delta = delta.unsqueeze(-1)
        
        deltaA = torch.exp(delta * A)
        
        B_expanded = B.unsqueeze(2).expand(-1, -1, self.nheads // self.ngroups, -1, -1)
        B_expanded = B_expanded.reshape(batch_size, seq_len, self.nheads, self.d_state)
        
        x = torch.zeros(batch_size, self.nheads, self.headdim, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        
        for t in range(seq_len):
            x = deltaA[:, t].unsqueeze(2) * x + delta[:, t].unsqueeze(2) * B_expanded[:, t].unsqueeze(2) * u[:, t].unsqueeze(-1)
            C_t = C[:, t].unsqueeze(1).expand(-1, self.nheads // self.ngroups, -1, -1)
            C_t = C_t.reshape(batch_size, self.nheads, self.d_state)
            y = (x * C_t.unsqueeze(2)).sum(-1)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)
        y = y + u * self.D.view(1, 1, self.nheads, 1)
        
        return y.view(batch_size, seq_len, self.d_inner)


class Mamba2Block(nn.Module):
    """Mamba-2 block using level1 operators."""
    def __init__(self, d_model: int, d_state: int = 128, d_conv: int = 4, 
                 expand: int = 2, headdim: int = 64, ngroups: int = 8):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = Mamba2Mixer(d_model, d_state, d_conv, expand, headdim, ngroups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Mamba-2 state space model.
    
    Uses level1 operators from KernelBench:
    - RMSNorm from level1/normalization/4_RMSNorm
    - Swish/SiLU from level1/activations/7_Swish
    - Softplus from level1/activations/11_Softplus
    """
    
    def __init__(
        self,
        d_model: int = 2048,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 8,
        num_layers: int = 48,
        vocab_size: int = 50280,
        **kwargs  # Accept and ignore extra kwargs for flexibility
    ):
        super().__init__()
        
        # Store config values
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            Mamba2Block(d_model, d_state, d_conv, expand, headdim, ngroups)
            for _ in range(num_layers)
        ])
        
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm_f(x)
        return self.lm_head(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 4
sequence_length = 1024
d_model = 2048
d_state = 128
d_conv = 4
expand = 2
headdim = 64
ngroups = 8
num_layers = 8
vocab_size = 50280


def get_inputs():
    return [torch.randint(0, vocab_size, (batch_size, sequence_length))]


def get_init_inputs():
    return [{
        'd_model': d_model,
        'd_state': d_state,
        'd_conv': d_conv,
        'expand': expand,
        'headdim': headdim,
        'ngroups': ngroups,
        'num_layers': num_layers,
        'vocab_size': vocab_size,
    }]
