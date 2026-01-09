import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Mamba-2 State Space Duality (SSD)
    
    Used by: Mamba-2, Codestral Mamba
    
    Mamba-2 structured state space duality kernel with improved
    parallelization via chunked computation.
    
    Shapes:
        x: (batch, seq_len, d_model)
        Output: (batch, seq_len, d_model)
    """
    
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, 
                 expand: int = 2, chunk_size: int = 256):
        """
        Initialize Mamba-2 SSD.
        
        Args:
            d_model: Model dimension
            d_state: State dimension (larger than Mamba-1)
            d_conv: Convolution kernel size
            expand: Expansion factor for inner dimension
            chunk_size: Chunk size for parallel scan
        """
        super(Model, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.chunk_size = chunk_size
        
        # Projections
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + 1, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv, 
            padding=d_conv - 1, groups=self.d_inner
        )
        
        # State matrix (simplified)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.register_buffer('A_log', torch.log(A))
        
        self.D = nn.Parameter(torch.ones(self.d_inner))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mamba-2 SSD forward pass.
        
        Args:
            x: Input (batch, seq_len, d_model)
            
        Returns:
            Output (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        xz = self.in_proj(x)
        
        # Split into components
        x_proj, z, B, C, dt = xz.split(
            [self.d_inner, self.d_inner, self.d_state, self.d_state, 1], dim=-1
        )
        
        # Convolution (causal)
        x_conv = x_proj.transpose(1, 2)  # (batch, d_inner, seq)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)  # (batch, seq, d_inner)
        x_conv = F.silu(x_conv)
        
        # Discretization
        dt = F.softplus(dt)  # (batch, seq, 1)
        A = -torch.exp(self.A_log)  # (d_state,)
        
        # Chunked parallel scan (simplified sequential version)
        # In practice, this uses a more sophisticated parallel algorithm
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(seq_len):
            # Discretize
            dt_t = dt[:, t]  # (batch, 1)
            dA = torch.exp(dt_t.unsqueeze(-1) * A)  # (batch, 1, d_state)
            dB = dt_t.unsqueeze(-1) * B[:, t].unsqueeze(1)  # (batch, 1, d_state)
            
            # State update
            h = dA * h + dB * x_conv[:, t].unsqueeze(-1)
            
            # Output
            y = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (batch, d_inner)
            outputs.append(y)
        
        y = torch.stack(outputs, dim=1)  # (batch, seq, d_inner)
        
        # Skip connection and gating
        y = y + x_conv * self.D
        y = y * F.silu(z)
        
        return self.out_proj(y)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
d_model = 4096
d_state = 64

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, d_model, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [d_model, d_state]

