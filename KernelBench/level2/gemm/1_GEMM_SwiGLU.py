import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused GEMM + SwiGLU Activation
    
    Used by: All modern LLMs (Llama, Mistral, Qwen, DeepSeek)
    
    Fuses the gate/up projection matmuls with SwiGLU activation.
    This is different from level1/SiluAndMul which takes pre-computed
    projections as input. This operator fuses the actual GEMMs.
    
    Found in: TensorRT-LLM (gemmSwigluPlugin), vLLM
    
    Shapes:
        Input: (batch_size, seq_len, hidden_size)
        Output: (batch_size, seq_len, intermediate_size)
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        """
        Initialize fused GEMM + SwiGLU.
        
        Args:
            hidden_size: Input hidden dimension
            intermediate_size: FFN intermediate dimension
        """
        super(Model, self).__init__()
        # Fused weight: combines gate and up projections
        # Shape: (hidden_size, 2 * intermediate_size)
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fused GEMM + SwiGLU.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            
        Returns:
            Activated tensor (batch_size, seq_len, intermediate_size)
        """
        # Single fused matmul for both gate and up
        gate_up = self.gate_up_proj(x)
        
        # Split and apply SwiGLU
        gate, up = gate_up.chunk(2, dim=-1)
        return F.silu(gate) * up


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
hidden_size = 4096
intermediate_size = 14336

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, intermediate_size]

