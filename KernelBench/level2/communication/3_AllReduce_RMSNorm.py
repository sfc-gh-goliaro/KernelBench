import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused AllReduce + RMSNorm
    
    Used by: Tensor-parallel LLM inference (vLLM, TensorRT-LLM, Megatron)
    
    Fuses the all-reduce collective with RMSNorm computation.
    The pattern AR -> Residual -> RMSNorm is common after attention
    and FFN in tensor-parallel transformers.
    
    Found in: TensorRT-LLM (allReduceFusionKernels - kARResidualRMSNorm)
    
    This benchmark simulates the local computation; actual distributed
    execution would overlap communication with compute.
    
    Shapes:
        Input: (batch_size, seq_len, hidden_size)
        Residual: (batch_size, seq_len, hidden_size)
        Output: (batch_size, seq_len, hidden_size)
    """
    
    def __init__(self, hidden_size: int, tp_size: int = 1, eps: float = 1e-6):
        """
        Initialize fused AllReduce + RMSNorm.
        
        Args:
            hidden_size: Hidden dimension
            tp_size: Tensor parallel size (for simulation)
            eps: Epsilon for numerical stability
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fused AllReduce + Residual Add + RMSNorm.
        
        Args:
            x: Input tensor (local partition result)
            residual: Residual tensor
            
        Returns:
            Tuple of (normalized output, updated residual)
        """
        # Simulate all-reduce (in real TP, this would be NCCL all-reduce)
        # For single GPU, this is identity. For TP>1, would sum across ranks.
        reduced = x * self.tp_size  # Simulate sum reduction
        
        # Fused residual add
        residual = reduced + residual
        
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
tp_size = 1

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    residual = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    return [x, residual]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, tp_size]

