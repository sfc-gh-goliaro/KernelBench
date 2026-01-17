import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Expert Combine
    
    Used by: All MoE architectures
    
    Unpermute expert outputs and compute weighted sum based on
    router weights. Reverses the dispatch operation.
    
    Shapes:
        Input: (total_tokens, hidden_size) expert outputs
        Output: (batch*seq, hidden_size) combined output
    """
    
    def __init__(self, top_k: int = 2):
        """
        Initialize expert combine.
        
        Args:
            top_k: Number of experts per token
        """
        super(Model, self).__init__()
        self.top_k = top_k
    
    def forward(self, expert_outputs: torch.Tensor, dispatch_indices: torch.Tensor,
                routing_weights: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """
        Combine expert outputs.
        
        Args:
            expert_outputs: Expert outputs (total_dispatched, hidden_size)
            dispatch_indices: Original token positions (total_dispatched,)
            routing_weights: Weights per expert (batch*seq, top_k)
            num_tokens: Original number of tokens (batch*seq)
            
        Returns:
            Combined output (batch*seq, hidden_size)
        """
        hidden_size = expert_outputs.shape[-1]
        
        # Initialize output
        output = torch.zeros(num_tokens, hidden_size, device=expert_outputs.device, 
                            dtype=expert_outputs.dtype)
        
        # Flatten routing weights
        flat_weights = routing_weights.view(-1)  # (batch*seq*top_k,)
        
        # Weight expert outputs
        weighted_outputs = expert_outputs * flat_weights.unsqueeze(-1)
        
        # Scatter-add back to original positions
        output.index_add_(0, dispatch_indices, weighted_outputs)
        
        return output


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
hidden_size = 4096
num_experts = 8
top_k = 2

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    num_tokens = batch_size * seq_length
    total_dispatched = num_tokens * top_k
    
    expert_outputs = torch.randn(total_dispatched, hidden_size, device='cuda')
    dispatch_indices = torch.randint(0, num_tokens, (total_dispatched,), device='cuda')
    routing_weights = torch.softmax(torch.randn(num_tokens, top_k, device='cuda'), dim=-1)
    
    return [expert_outputs, dispatch_indices, routing_weights, num_tokens]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [top_k]

