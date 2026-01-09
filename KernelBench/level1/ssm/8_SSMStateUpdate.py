import torch
import torch.nn as nn

class Model(nn.Module):
    """
    SSM State Update
    
    Used by: Mamba, Mamba-2, RetNet, RWKV
    
    Core state update operation for state-space models:
    h_t = A * h_{t-1} + B * x_t
    y_t = C * h_t + D * x_t
    
    This is the inner loop computation of selective scan.
    
    Shapes:
        Input x: (batch_size, intermediate_size)
        Input h: (batch_size, intermediate_size, state_size)
        Output h: (batch_size, intermediate_size, state_size)
        Output y: (batch_size, intermediate_size)
    """
    
    def __init__(self, intermediate_size: int = 5120, state_size: int = 16):
        """
        Initialize SSM State Update.
        
        Args:
            intermediate_size: SSM intermediate dimension
            state_size: State dimension (N in Mamba)
        """
        super(Model, self).__init__()
        self.intermediate_size = intermediate_size
        self.state_size = state_size
        
        # Fixed A matrix (log-parameterized for stability)
        A = torch.arange(1, state_size + 1, dtype=torch.float32).repeat(intermediate_size, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D: skip connection parameter
        self.D = nn.Parameter(torch.ones(intermediate_size))
    
    def forward(self, x: torch.Tensor, h: torch.Tensor, 
                dt: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> tuple:
        """
        Compute one step of SSM state update.
        
        Args:
            x: Input at current timestep (batch_size, intermediate_size)
            h: Previous state (batch_size, intermediate_size, state_size)
            dt: Delta/timestep (batch_size, intermediate_size)
            B: Input-to-state matrix (batch_size, state_size)
            C: State-to-output matrix (batch_size, state_size)
            
        Returns:
            Tuple of (new_h, y):
                new_h: Updated state (batch_size, intermediate_size, state_size)
                y: Output (batch_size, intermediate_size)
        """
        # Get A (negative for stability)
        A = -torch.exp(self.A_log)  # (intermediate_size, state_size)
        
        # Discretize A: dA = exp(dt * A)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # (B, intermediate, state_size)
        
        # Discretize B: dB = dt * B
        dB = dt.unsqueeze(-1) * B.unsqueeze(1)  # (B, intermediate, state_size)
        
        # State update: h_new = dA * h + dB * x
        h_new = dA * h + dB * x.unsqueeze(-1)
        
        # Output: y = (C @ h) + D * x
        y = (h_new * C.unsqueeze(1)).sum(-1) + self.D * x
        
        return h_new, y


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 32
intermediate_size = 5120  # Mamba-2.8B expanded
state_size = 16

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, intermediate_size, device='cuda')
    h = torch.randn(batch_size, intermediate_size, state_size, device='cuda')
    dt = torch.rand(batch_size, intermediate_size, device='cuda') * 0.1
    B = torch.randn(batch_size, state_size, device='cuda')
    C = torch.randn(batch_size, state_size, device='cuda')
    return [x, h, dt, B, C]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [intermediate_size, state_size]

