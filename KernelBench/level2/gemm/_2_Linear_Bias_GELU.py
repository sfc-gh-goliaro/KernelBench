import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused Linear + Bias + GELU
    
    Used by: BERT, GPT-2, T5, many encoder models
    
    Fuses linear projection with bias addition and GELU activation.
    Common in FFN blocks of older transformer architectures.
    
    Found in: cuBLAS epilogue fusions, TensorRT
    
    Shapes:
        Input: (batch_size, seq_len, in_features)
        Output: (batch_size, seq_len, out_features)
    """
    
    def __init__(self, in_features: int, out_features: int):
        """
        Initialize fused Linear + Bias + GELU.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
        """
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fused Linear + Bias + GELU.
        
        Args:
            x: Input tensor (batch_size, seq_len, in_features)
            
        Returns:
            Output tensor (batch_size, seq_len, out_features)
        """
        # Fused: matmul + bias + GELU
        return F.gelu(self.linear(x))


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 512
in_features = 768
out_features = 3072  # BERT intermediate size

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, in_features, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [in_features, out_features]

