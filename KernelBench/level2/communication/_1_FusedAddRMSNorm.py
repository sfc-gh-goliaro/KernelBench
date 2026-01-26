import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused Add + RMSNorm
    
    Used by: All transformer architectures (Llama, Mistral, etc.)
    
    Fuses residual addition with RMSNorm into a single kernel pass.
    This is a critical fusion for transformer blocks that avoids
    reading/writing the residual tensor twice.
    
    Found in: sglang (sgl_fused_add_rmsnorm), vLLM, TensorRT-LLM
    
    Shapes:
        Input x: (batch_size, seq_len, hidden_size)
        Input residual: (batch_size, seq_len, hidden_size)
        Output: (batch_size, seq_len, hidden_size)
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Initialize fused add + RMSNorm.
        
        Args:
            hidden_size: Hidden dimension
            eps: Epsilon for numerical stability
        """
        super(Model, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fused add + RMSNorm.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            residual: Residual tensor (batch_size, seq_len, hidden_size)
            
        Returns:
            Tuple of (normalized output, updated residual)
        """
        # Fused: residual = x + residual, then RMSNorm(residual)
        residual = x + residual
        
        # RMSNorm
        rms = torch.sqrt(residual.pow(2).mean(-1, keepdim=True) + self.eps)
        normed = residual / rms * self.weight
        
        return normed, residual


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
hidden_size = 4096

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    residual = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    return [x, residual]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size]

