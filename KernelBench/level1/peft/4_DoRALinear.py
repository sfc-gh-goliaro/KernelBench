import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    DoRA (Weight-Decomposed Low-Rank Adaptation) Linear Layer
    
    Used by: DoRA fine-tuning, parameter-efficient training
    
    Decomposes weight updates into magnitude and direction components.
    W = m * (W0 + BA) / ||W0 + BA||
    More expressive than LoRA while maintaining similar parameter count.
    
    Shapes:
        Input: (batch_size, seq_length, in_features)
        Output: (batch_size, seq_length, out_features)
    """
    
    def __init__(self, in_features: int = 4096, out_features: int = 4096,
                 rank: int = 16, alpha: float = 32.0):
        """
        Initialize DoRA Linear.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            rank: Rank of low-rank matrices
            alpha: Scaling factor
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        
        # Frozen base weight
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02,
            requires_grad=False
        )
        
        # Low-rank adaptation matrices (trainable)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Magnitude vector (trainable) - DoRA specific
        # Initialized to match base weight column norms
        with torch.no_grad():
            base_norm = self.weight.norm(dim=1, keepdim=True)
        self.magnitude = nn.Parameter(base_norm.squeeze())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply DoRA linear transformation.
        
        Args:
            x: Input tensor (batch_size, seq_length, in_features)
            
        Returns:
            Output tensor (batch_size, seq_length, out_features)
        """
        # Compute adapted weight: W0 + scaling * B @ A
        adapted_weight = self.weight + self.scaling * (self.lora_B @ self.lora_A)
        
        # Normalize adapted weight (direction)
        weight_norm = adapted_weight.norm(dim=1, keepdim=True)
        direction = adapted_weight / (weight_norm + 1e-8)
        
        # Apply magnitude scaling
        final_weight = self.magnitude.unsqueeze(1) * direction
        
        return F.linear(x, final_weight)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
in_features = 4096
out_features = 4096
rank = 16
alpha = 32.0

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, in_features, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [in_features, out_features, rank, alpha]

