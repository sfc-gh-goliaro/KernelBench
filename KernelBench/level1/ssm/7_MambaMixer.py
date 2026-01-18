import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Mamba Mixer (Full S6 Block)
    
    Used by: Mamba-1, Jamba (hybrid layers)
    
    Complete Mamba mixer block with input projection, 1D convolution,
    selective scan (S6), and output projection.
    
    Shapes:
        Input: (batch_size, seq_length, hidden_size)
        Output: (batch_size, seq_length, hidden_size)
    """
    
    def __init__(self, hidden_size: int = 2560, state_size: int = 16,
                 conv_kernel_size: int = 4, expand_factor: int = 2):
        """
        Initialize Mamba Mixer.
        
        Args:
            hidden_size: Input/output hidden dimension
            state_size: SSM state dimension (N in Mamba paper)
            conv_kernel_size: 1D convolution kernel size
            expand_factor: Expansion factor for intermediate dimension
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.conv_kernel_size = conv_kernel_size
        self.intermediate_size = hidden_size * expand_factor
        
        # Input projection: x -> (z, x_proj)
        self.in_proj = nn.Linear(hidden_size, self.intermediate_size * 2, bias=False)
        
        # 1D convolution
        self.conv1d = nn.Conv1d(
            self.intermediate_size, self.intermediate_size,
            kernel_size=conv_kernel_size, padding=conv_kernel_size - 1,
            groups=self.intermediate_size
        )
        
        # SSM parameters projection
        # Projects to: dt, B, C (delta, B, C in selective scan)
        self.x_proj = nn.Linear(
            self.intermediate_size, 
            state_size * 2 + 1,  # dt_rank=1, B, C
            bias=False
        )
        
        # dt projection
        self.dt_proj = nn.Linear(1, self.intermediate_size, bias=True)
        
        # SSM parameters (A is fixed, D is learnable)
        A = torch.arange(1, state_size + 1, dtype=torch.float32).repeat(self.intermediate_size, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        
        # Output projection
        self.out_proj = nn.Linear(self.intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Mamba mixer block.
        
        Args:
            x: Input tensor (batch_size, seq_length, hidden_size)
            
        Returns:
            Output tensor (batch_size, seq_length, hidden_size)
        """
        batch_size, seq_length, _ = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (B, L, 2*intermediate)
        x_proj, z = xz.chunk(2, dim=-1)  # Each: (B, L, intermediate)
        
        # 1D convolution (requires channels-first)
        x_conv = x_proj.transpose(1, 2)  # (B, intermediate, L)
        x_conv = self.conv1d(x_conv)[:, :, :seq_length]  # Trim to original length
        x_conv = x_conv.transpose(1, 2)  # (B, L, intermediate)
        
        # SiLU activation
        x_conv = F.silu(x_conv)
        
        # Project to SSM parameters
        ssm_params = self.x_proj(x_conv)  # (B, L, state_size*2 + 1)
        dt = ssm_params[:, :, :1]
        B = ssm_params[:, :, 1:1+self.state_size]
        C = ssm_params[:, :, 1+self.state_size:]
        
        # dt projection
        dt = F.softplus(self.dt_proj(dt))  # (B, L, intermediate)
        
        # Discretize A
        A = -torch.exp(self.A_log)  # (intermediate, state_size)
        
        # Simplified selective scan (sequential for clarity)
        # In practice, this would use a parallel scan algorithm
        y = self._selective_scan(x_conv, dt, A, B, C)
        
        # Apply D (skip connection)
        y = y + self.D * x_conv
        
        # Gate with z
        y = y * F.silu(z)
        
        # Output projection
        return self.out_proj(y)
    
    def _selective_scan(self, x, dt, A, B, C):
        """Simplified sequential selective scan."""
        batch_size, seq_length, intermediate = x.shape
        
        # Initialize state
        h = torch.zeros(batch_size, intermediate, self.state_size, device=x.device)
        outputs = []
        
        for t in range(seq_length):
            # Get current inputs
            x_t = x[:, t, :]  # (B, intermediate)
            dt_t = dt[:, t, :]  # (B, intermediate)
            B_t = B[:, t, :]  # (B, state_size)
            C_t = C[:, t, :]  # (B, state_size)
            
            # Discretize: dA = exp(dt * A)
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, intermediate, state_size)
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, intermediate, state_size)
            
            # Update state: h = dA * h + dB * x
            h = dA * h + dB * x_t.unsqueeze(-1)
            
            # Output: y = C @ h
            y_t = (h * C_t.unsqueeze(1)).sum(-1)  # (B, intermediate)
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "seq_length": 512, "hidden_size": 2560, "state_size": 16, "conv_kernel_size": 4, "expand_factor": 2},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("ssm", "7_MambaMixer")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["hidden_size"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["state_size"], p["conv_kernel_size"], p["expand_factor"]]
