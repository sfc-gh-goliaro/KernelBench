import torch
import torch.nn as nn

class Model(nn.Module):
    """
    LoRA Linear Layer
    
    Used by: LoRA finetuning and serving
    
    LoRA: output = base_linear(x) + scale * B(A(x)) where A, B are
    low-rank adapter matrices.
    
    Shapes:
        Input: (batch, seq_len, in_features)
        Output: (batch, seq_len, out_features)
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, 
                 alpha: float = 16.0, dropout: float = 0.0):
        """
        Initialize LoRA linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: LoRA rank (r)
            alpha: LoRA alpha for scaling
            dropout: Dropout probability
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Base linear layer (frozen during LoRA training)
        self.base_linear = nn.Linear(in_features, out_features, bias=False)
        
        # LoRA adapter matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA.
        
        Args:
            x: Input tensor (batch, seq_len, in_features)
            
        Returns:
            Output tensor (batch, seq_len, out_features)
        """
        # Base model output
        base_output = self.base_linear(x)
        
        # LoRA adapter output
        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        
        # Combine with scaling
        return base_output + self.scaling * lora_output


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
in_features = 4096
out_features = 4096
rank = 16

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, in_features, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [in_features, out_features, rank]

