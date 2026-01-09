import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Adapter Layer
    
    Used by: Adapter-based fine-tuning (Houlsby adapters)
    
    Lightweight bottleneck layer inserted between transformer layers.
    Projects down, applies nonlinearity, projects up, adds residual.
    
    Architecture: x + W_up(activation(W_down(x)))
    
    Shapes:
        Input: (batch_size, seq_length, hidden_size)
        Output: (batch_size, seq_length, hidden_size)
    """
    
    def __init__(self, hidden_size: int = 4096, adapter_size: int = 64,
                 activation: str = 'relu'):
        """
        Initialize Adapter Layer.
        
        Args:
            hidden_size: Input/output hidden dimension
            adapter_size: Bottleneck dimension
            activation: Activation function ('relu', 'gelu', 'swish')
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        
        # Down projection
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        # Up projection
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        
        # Initialize up_proj to near-zero for identity initialization
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adapter transformation with residual.
        
        Args:
            x: Input tensor (batch_size, seq_length, hidden_size)
            
        Returns:
            Output tensor (batch_size, seq_length, hidden_size)
        """
        # Bottleneck transformation
        h = self.down_proj(x)
        h = self.activation(h)
        h = self.up_proj(h)
        
        # Residual connection
        return x + h


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
hidden_size = 4096
adapter_size = 64
activation = 'relu'

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, adapter_size, activation]

