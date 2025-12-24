import torch
import torch.nn as nn
import torch.nn.functional as F


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Expert Choice Routing for Mixture of Experts.
    
    Instead of tokens choosing experts, experts choose their top-k tokens.
    This guarantees perfect load balancing across experts.
    
    Based on: "Mixture-of-Experts with Expert Choice Routing"
    """
    def __init__(self, dim, hidden_dim, num_experts, tokens_per_expert):
        """
        :param dim: Input/output dimension
        :param hidden_dim: Expert hidden dimension
        :param num_experts: Number of experts
        :param tokens_per_expert: Fixed number of tokens each expert processes
        """
        super(Model, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.tokens_per_expert = tokens_per_expert
        
        # Router (computes affinity scores)
        self.router = nn.Linear(dim, num_experts, bias=False)
        
        # Experts
        self.experts_w1 = nn.Parameter(torch.randn(num_experts, dim, hidden_dim) * 0.02)
        self.experts_w2 = nn.Parameter(torch.randn(num_experts, hidden_dim, dim) * 0.02)
        
    def forward(self, x):
        """
        Forward pass for Expert Choice routing.
        
        :param x: Input tensor of shape (batch_size, seq_len, dim)
        :return: Output tensor of shape (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        num_tokens = batch_size * seq_len
        
        # Flatten
        x_flat = x.view(-1, self.dim)  # (num_tokens, dim)
        
        # Compute router scores
        router_logits = self.router(x_flat)  # (num_tokens, num_experts)
        
        # Transpose: let experts choose tokens
        # Each expert picks top-k tokens based on affinity
        expert_scores = router_logits.t()  # (num_experts, num_tokens)
        
        # Apply softmax over tokens for each expert
        expert_probs = F.softmax(expert_scores, dim=-1)  # (num_experts, num_tokens)
        
        # Each expert selects top-k tokens
        k = min(self.tokens_per_expert, num_tokens)
        top_k_probs, top_k_indices = torch.topk(expert_probs, k, dim=-1)
        # top_k_indices: (num_experts, k) - which tokens each expert processes
        # top_k_probs: (num_experts, k) - the probability weights
        
        # Normalize weights per token (a token may be selected by multiple experts)
        # First, accumulate weights per token
        token_weight_sum = torch.zeros(num_tokens, device=x.device)
        for e in range(self.num_experts):
            token_weight_sum.scatter_add_(0, top_k_indices[e], top_k_probs[e])
        
        # Initialize output accumulator
        output = torch.zeros_like(x_flat)
        
        # Process each expert
        for e in range(self.num_experts):
            # Get indices and weights for this expert's selected tokens
            selected_indices = top_k_indices[e]  # (k,)
            selected_weights = top_k_probs[e]  # (k,)
            
            # Get the tokens
            expert_input = x_flat[selected_indices]  # (k, dim)
            
            # Apply expert FFN
            hidden = torch.matmul(expert_input, self.experts_w1[e])  # (k, hidden)
            hidden = F.gelu(hidden)
            expert_output = torch.matmul(hidden, self.experts_w2[e])  # (k, dim)
            
            # Weight the output
            weighted_output = selected_weights.unsqueeze(-1) * expert_output  # (k, dim)
            
            # Accumulate to output
            output.scatter_add_(0, 
                               selected_indices.unsqueeze(-1).expand(-1, self.dim),
                               weighted_output)
        
        # Normalize by total weight per token
        token_weight_sum = token_weight_sum.clamp(min=1e-9)
        output = output / token_weight_sum.unsqueeze(-1)
        
        return output.view(batch_size, seq_len, self.dim)


# Test parameters
batch_size = 16
seq_len = 256
dim = 512
hidden_dim = 2048
num_experts = 8
tokens_per_expert = 64  # Each expert processes 64 tokens

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    return [torch.randn(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, hidden_dim, num_experts, tokens_per_expert]

