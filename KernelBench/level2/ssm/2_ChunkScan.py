import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused Chunk-Wise Scan for Mamba/SSM
    
    Used by: Mamba, Mamba-2, State Space Models
    
    Implements fused chunk-wise selective scan computation.
    Processes sequence in chunks with inter-chunk state passing,
    fusing the scan operations within each chunk.
    
    Found in: Mamba (mamba_ssm), TensorRT-LLM (selectiveScan.cu)
    
    Shapes:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, 
                 expand: int = 2, chunk_size: int = 256):
        """
        Initialize fused chunk scan.
        
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Convolution width
            expand: Expansion factor
            chunk_size: Size of each chunk for processing
        """
        super(Model, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.chunk_size = chunk_size
        
        # SSM parameters
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Conv for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 
            kernel_size=d_conv, 
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # Delta, B, C projections
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
    
    def _selective_scan_chunk(self, u: torch.Tensor, delta: torch.Tensor,
                               A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                               D: torch.Tensor, initial_state: torch.Tensor = None):
        """
        Selective scan over a single chunk.
        
        This is where the fusion happens - computing the scan
        over multiple timesteps in a single kernel.
        """
        batch, chunk_len, d_inner = u.shape
        
        # Discretize A
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D, N)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        
        # Initialize state
        if initial_state is None:
            state = torch.zeros(batch, d_inner, self.d_state, device=u.device, dtype=u.dtype)
        else:
            state = initial_state
        
        # Scan (in real impl, this is a parallel scan)
        outputs = []
        for t in range(chunk_len):
            state = deltaA[:, t] * state + deltaB_u[:, t]
            y_t = (state * C[:, t].unsqueeze(1)).sum(-1)
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)
        
        # Add skip connection
        y = y + u * D
        
        return y, state
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fused chunk-wise scan.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        # Conv
        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_inner = F.silu(x_conv)
        
        # SSM parameters
        x_dbc = self.x_proj(x_inner)
        delta = F.softplus(self.dt_proj(x_dbc[..., :1]))
        B = x_dbc[..., 1:self.d_state+1]
        C = x_dbc[..., self.d_state+1:]
        
        A = -torch.exp(self.A_log)
        
        # Process in chunks
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        outputs = []
        state = None
        
        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, seq_len)
            
            chunk_out, state = self._selective_scan_chunk(
                x_inner[:, start:end],
                delta[:, start:end].squeeze(-1),
                A, B[:, start:end], C[:, start:end],
                self.D, state
            )
            outputs.append(chunk_out)
        
        y = torch.cat(outputs, dim=1)
        
        # Gate and output
        y = y * F.silu(z)
        return self.out_proj(y)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
d_model = 2048
d_state = 16

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, d_model, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [d_model, d_state]

