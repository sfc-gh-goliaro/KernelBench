import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Selective Scan (Mamba S6)
    
    Used by: Mamba, Jamba (SSM layers)
    
    Mamba S6 selective scan: input-dependent state transitions.
    The core operation that makes SSM competitive with attention.
    
    Shapes:
        x: (batch, seq_len, d_inner)
        delta: (batch, seq_len, d_inner) - discretization step
        A: (d_inner, d_state) - state matrix
        B: (batch, seq_len, d_state) - input-dependent B
        C: (batch, seq_len, d_state) - input-dependent C
        Output: (batch, seq_len, d_inner)
    """
    
    def __init__(self, d_inner: int, d_state: int = 16):
        """
        Initialize selective scan.
        
        Args:
            d_inner: Inner dimension
            d_state: State dimension
        """
        super(Model, self).__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        
        # A is learned (log space for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.register_buffer('A_log', torch.log(A))
        
        # D is a skip connection parameter
        self.D = nn.Parameter(torch.ones(d_inner))
    
    def forward(self, x: torch.Tensor, delta: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Run selective scan.
        
        Args:
            x: Input (batch, seq_len, d_inner)
            delta: Discretization step (batch, seq_len, d_inner)
            B: Input-dependent B (batch, seq_len, d_state)
            C: Input-dependent C (batch, seq_len, d_state)
            
        Returns:
            Output (batch, seq_len, d_inner)
        """
        batch_size, seq_len, _ = x.shape
        
        # Get A from log space
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Discretize A and B using delta
        # deltaA = exp(delta * A)
        # deltaB = delta * B
        delta_A = torch.exp(delta.unsqueeze(-1) * A)  # (batch, seq, d_inner, d_state)
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)  # (batch, seq, d_inner, d_state)
        
        # x contribution to B
        delta_B_x = delta_B * x.unsqueeze(-1)  # (batch, seq, d_inner, d_state)
        
        # Sequential scan (this is the slow path; real impl uses parallel scan)
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(seq_len):
            # h = deltaA * h + deltaB * x
            h = delta_A[:, t] * h + delta_B_x[:, t]
            # y = h @ C^T
            y = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (batch, d_inner)
            outputs.append(y)
        
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)
        
        # Add skip connection
        y = y + x * self.D
        
        return y


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "d_inner": 4096, "d_state": 16},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("ssm", "1_SelectiveScan")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["d_inner"]), dtype=dtype, device=device)
    B = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["d_state"]), dtype=dtype, device=device)
    C = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["d_state"]), dtype=dtype, device=device)
    return [x, delta, B, C]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["d_inner"], p["d_state"]]
