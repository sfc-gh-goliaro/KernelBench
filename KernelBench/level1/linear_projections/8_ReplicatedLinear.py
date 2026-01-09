import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Replicated Linear (Router Linear)
    
    Used by: vLLM, TensorRT-LLM (MoE routing)
    
    Linear layer that is replicated across all tensor parallel ranks.
    Used for MoE routers where each rank needs the full routing decision.
    Unlike parallel linears, weights are not partitioned.
    
    Shapes:
        Input: (num_tokens, hidden_size)
        Output: (num_tokens, num_experts)
    """
    
    def __init__(self, hidden_size: int = 4096, num_experts: int = 8):
        """
        Initialize Replicated Linear.
        
        Args:
            hidden_size: Input hidden dimension
            num_experts: Number of experts (output dimension)
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        
        # Full (non-partitioned) weight matrix
        self.weight = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute router logits.
        
        Args:
            x: Input tensor (num_tokens, hidden_size)
            
        Returns:
            Router logits (num_tokens, num_experts)
        """
        return self.weight(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================

num_tokens = 16384  # batch_size * seq_length
hidden_size = 4096
num_experts = 8  # Mixtral

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(num_tokens, hidden_size, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, num_experts]

