import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Expert Parallel Dispatch for Mixture of Experts.
    
    Handles the dispatching of tokens to experts and combining outputs,
    with capacity constraints and efficient batching.
    
    Based on: "GShard" and "Switch Transformers"
    """
    def __init__(self, dim, hidden_dim, num_experts, capacity_factor=1.25, drop_tokens=True):
        """
        :param dim: Input/output dimension
        :param hidden_dim: Expert hidden dimension
        :param num_experts: Number of experts
        :param capacity_factor: Factor to determine expert capacity
        :param drop_tokens: Whether to drop tokens that exceed capacity
        """
        super(Model, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        
        # Create expert FFN layers
        # Each expert is a simple 2-layer FFN
        self.experts_fc1 = nn.Parameter(torch.randn(num_experts, dim, hidden_dim) * 0.02)
        self.experts_fc2 = nn.Parameter(torch.randn(num_experts, hidden_dim, dim) * 0.02)
        self.experts_bias1 = nn.Parameter(torch.zeros(num_experts, hidden_dim))
        self.experts_bias2 = nn.Parameter(torch.zeros(num_experts, dim))
        
    def _compute_capacity(self, num_tokens):
        """Compute expert capacity based on tokens and capacity factor."""
        tokens_per_expert = num_tokens / self.num_experts
        capacity = int(tokens_per_expert * self.capacity_factor)
        return max(capacity, 1)
    
    def forward(self, x, expert_indices, expert_weights):
        """
        Forward pass: dispatch tokens to experts and combine outputs.
        
        :param x: Input tensor of shape (batch_size, seq_len, dim)
        :param expert_indices: Expert assignments (batch_size, seq_len, top_k)
        :param expert_weights: Expert weights (batch_size, seq_len, top_k)
        :return: Output tensor of shape (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        top_k = expert_indices.shape[-1]
        num_tokens = batch_size * seq_len
        
        # Compute capacity
        capacity = self._compute_capacity(num_tokens)
        
        # Flatten inputs
        x_flat = x.view(-1, self.dim)  # (num_tokens, dim)
        indices_flat = expert_indices.view(-1, top_k)  # (num_tokens, top_k)
        weights_flat = expert_weights.view(-1, top_k)  # (num_tokens, top_k)
        
        # Create dispatch mask: (num_tokens, num_experts, top_k)
        # This tracks which expert slot each token goes to
        expert_mask = torch.zeros(num_tokens, self.num_experts, device=x.device)
        
        # Count tokens per expert
        for k in range(top_k):
            for e in range(self.num_experts):
                mask_e = (indices_flat[:, k] == e)
                # Count how many tokens want expert e
                count = mask_e.sum()
                # Track assignment (simplified - in practice use cumsum for position)
                expert_mask[:, e] += mask_e.float() * weights_flat[:, k]
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each expert
        for e in range(self.num_experts):
            # Find tokens assigned to this expert
            # Get all tokens that have this expert in their top-k
            expert_weight = torch.zeros(num_tokens, device=x.device)
            expert_tokens_mask = torch.zeros(num_tokens, dtype=torch.bool, device=x.device)
            
            for k in range(top_k):
                mask_k = (indices_flat[:, k] == e)
                expert_tokens_mask |= mask_k
                expert_weight += mask_k.float() * weights_flat[:, k]
            
            # Get indices of tokens for this expert
            token_indices = expert_tokens_mask.nonzero(as_tuple=True)[0]
            
            if len(token_indices) == 0:
                continue
            
            # Apply capacity constraint
            if self.drop_tokens and len(token_indices) > capacity:
                token_indices = token_indices[:capacity]
            
            # Get tokens for this expert
            expert_input = x_flat[token_indices]  # (num_selected, dim)
            
            # Apply expert FFN
            # First layer
            hidden = F.linear(expert_input, self.experts_fc1[e].t(), self.experts_bias1[e])
            hidden = F.gelu(hidden)
            
            # Second layer
            expert_output = F.linear(hidden, self.experts_fc2[e].t(), self.experts_bias2[e])
            
            # Weight by gating score and accumulate
            weights = expert_weight[token_indices].unsqueeze(-1)
            output[token_indices] += weights * expert_output
        
        return output.view(batch_size, seq_len, self.dim)


# Test parameters
batch_size = 16
seq_len = 256
dim = 512
hidden_dim = 2048
num_experts = 8
top_k = 2

def get_inputs():
    x = torch.randn(batch_size, seq_len, dim)
    # Generate random expert assignments
    expert_indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k))
    expert_weights = F.softmax(torch.randn(batch_size, seq_len, top_k), dim=-1)
    return [x, expert_indices, expert_weights]

def get_init_inputs():
    return [dim, hidden_dim, num_experts]

