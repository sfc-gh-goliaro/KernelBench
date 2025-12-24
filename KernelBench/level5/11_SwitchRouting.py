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
    Switch Transformer Routing (Top-1 Routing).
    
    Each token is routed to exactly one expert, achieving simpler
    and more efficient routing than top-k approaches.
    
    Based on: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
    """
    def __init__(self, dim, hidden_dim, num_experts, capacity_factor=1.0, jitter_noise=0.1):
        """
        :param dim: Input/output dimension
        :param hidden_dim: Expert hidden dimension
        :param num_experts: Number of experts
        :param capacity_factor: Factor to determine expert capacity
        :param jitter_noise: Noise added for load balancing during training
        """
        super(Model, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.jitter_noise = jitter_noise
        
        # Router
        self.router = nn.Linear(dim, num_experts, bias=False)
        
        # Experts (using batched linear layers)
        self.experts_w1 = nn.Parameter(torch.randn(num_experts, dim, hidden_dim) * 0.02)
        self.experts_w2 = nn.Parameter(torch.randn(num_experts, hidden_dim, dim) * 0.02)
        
    def _add_jitter(self, x):
        """Add multiplicative jitter noise for load balancing."""
        if self.training and self.jitter_noise > 0:
            noise = 1.0 + torch.rand_like(x) * self.jitter_noise
            return x * noise
        return x
    
    def forward(self, x):
        """
        Forward pass for Switch routing.
        
        :param x: Input tensor of shape (batch_size, seq_len, dim)
        :return: Output tensor of shape (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        num_tokens = batch_size * seq_len
        
        # Add jitter noise
        x_jittered = self._add_jitter(x)
        
        # Compute router logits
        router_logits = self.router(x_jittered)  # (batch, seq, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-1 expert
        expert_weights, expert_indices = router_probs.max(dim=-1)  # (batch, seq)
        
        # Flatten for processing
        x_flat = x.view(-1, self.dim)  # (num_tokens, dim)
        indices_flat = expert_indices.view(-1)  # (num_tokens,)
        weights_flat = expert_weights.view(-1, 1)  # (num_tokens, 1)
        
        # Compute capacity
        capacity = int(num_tokens / self.num_experts * self.capacity_factor)
        capacity = max(capacity, 4)  # Minimum capacity
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Compute load balancing auxiliary loss
        # Fraction of tokens routed to each expert
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        for e in range(self.num_experts):
            expert_counts[e] = (indices_flat == e).float().sum()
        router_fraction = expert_counts / num_tokens
        
        # Mean router probability per expert
        router_prob_mean = router_probs.view(-1, self.num_experts).mean(dim=0)
        
        # Auxiliary loss
        aux_loss = (router_fraction * router_prob_mean).sum() * self.num_experts
        
        # Process each expert
        for e in range(self.num_experts):
            # Find tokens for this expert
            expert_mask = (indices_flat == e)
            token_indices = expert_mask.nonzero(as_tuple=True)[0]
            
            if len(token_indices) == 0:
                continue
            
            # Apply capacity constraint
            if len(token_indices) > capacity:
                # Randomly select which tokens to keep
                perm = torch.randperm(len(token_indices), device=x.device)[:capacity]
                token_indices = token_indices[perm]
            
            # Get tokens for this expert
            expert_input = x_flat[token_indices]  # (selected, dim)
            
            # Apply expert FFN
            hidden = torch.matmul(expert_input, self.experts_w1[e])  # (selected, hidden)
            hidden = F.gelu(hidden)
            expert_output = torch.matmul(hidden, self.experts_w2[e])  # (selected, dim)
            
            # Weight by router probability
            token_weights = weights_flat[token_indices]
            output[token_indices] = token_weights * expert_output
        
        output = output.view(batch_size, seq_len, self.dim)
        
        return output, aux_loss


# Test parameters
batch_size = 32
seq_len = 256
dim = 512
hidden_dim = 2048
num_experts = 8

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
    return [dim, hidden_dim, num_experts]

