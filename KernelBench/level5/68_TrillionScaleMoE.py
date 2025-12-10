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
    Trillion-Scale Mixture of Experts for Large Models.
    
    Efficient expert routing and computation for models with
    100+ experts. Used in Ring-1T and similar large-scale models.
    
    Based on: GShard, Switch Transformer, and large-scale MoE optimizations
    """
    def __init__(self, dim, hidden_dim, num_experts=256, top_k=4,
                 expert_capacity_factor=1.25, aux_loss_weight=0.01,
                 use_expert_parallelism=True):
        """
        :param dim: Input/output dimension
        :param hidden_dim: Expert hidden dimension
        :param num_experts: Total number of experts
        :param top_k: Number of experts per token
        :param expert_capacity_factor: Capacity factor for load balancing
        :param aux_loss_weight: Weight for auxiliary loss
        :param use_expert_parallelism: Whether to use expert parallelism patterns
        """
        super(Model, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_factor = expert_capacity_factor
        self.aux_loss_weight = aux_loss_weight
        self.use_expert_parallelism = use_expert_parallelism
        
        # Router with noise for exploration
        self.router = nn.Linear(dim, num_experts, bias=False)
        
        # Expert parameters (chunked for memory efficiency)
        # Each expert: gate_proj, up_proj, down_proj
        self.expert_gate = nn.Parameter(
            torch.randn(num_experts, dim, hidden_dim) * 0.02
        )
        self.expert_up = nn.Parameter(
            torch.randn(num_experts, dim, hidden_dim) * 0.02
        )
        self.expert_down = nn.Parameter(
            torch.randn(num_experts, hidden_dim, dim) * 0.02
        )
        
        # Load balancing statistics
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.zeros(1))
        
    def _compute_routing(self, x):
        """Compute top-k routing with load balancing."""
        batch_size, seq_len, _ = x.shape
        num_tokens = batch_size * seq_len
        
        # Router logits with optional noise
        router_logits = self.router(x)  # (batch, seq, num_experts)
        
        if self.training:
            # Add noise for exploration
            noise = torch.randn_like(router_logits) * 0.1
            router_logits = router_logits + noise
        
        # Softmax routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize selected probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute auxiliary load balancing loss
        # Fraction of tokens routed to each expert
        flat_indices = top_k_indices.view(-1, self.top_k)
        expert_mask = torch.zeros(num_tokens, self.num_experts, device=x.device)
        for k in range(self.top_k):
            expert_mask.scatter_add_(
                1, flat_indices[:, k:k+1],
                torch.ones(num_tokens, 1, device=x.device)
            )
        
        # Load balancing loss
        tokens_per_expert = expert_mask.sum(dim=0) / num_tokens
        router_prob_per_expert = router_probs.view(-1, self.num_experts).mean(dim=0)
        aux_loss = (tokens_per_expert * router_prob_per_expert).sum() * self.num_experts
        
        return top_k_probs, top_k_indices, aux_loss * self.aux_loss_weight
    
    def _expert_forward(self, x, expert_idx):
        """Forward through a single expert."""
        gate = F.silu(F.linear(x, self.expert_gate[expert_idx].t()))
        up = F.linear(x, self.expert_up[expert_idx].t())
        hidden = gate * up
        return F.linear(hidden, self.expert_down[expert_idx].t())
    
    def forward(self, x):
        """
        Forward pass with trillion-scale MoE routing.
        
        :param x: Input tensor (batch, seq_len, dim)
        :return: Tuple of (output, aux_loss)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute routing
        top_k_probs, top_k_indices, aux_loss = self._compute_routing(x)
        
        # Flatten for processing
        x_flat = x.view(-1, self.dim)
        indices_flat = top_k_indices.view(-1, self.top_k)
        probs_flat = top_k_probs.view(-1, self.top_k)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Compute capacity
        num_tokens = batch_size * seq_len
        capacity = int(num_tokens * self.top_k / self.num_experts * self.expert_capacity_factor)
        capacity = max(capacity, 1)
        
        # Process experts in chunks for memory efficiency
        chunk_size = 32  # Process 32 experts at a time
        
        for chunk_start in range(0, self.num_experts, chunk_size):
            chunk_end = min(chunk_start + chunk_size, self.num_experts)
            
            for k in range(self.top_k):
                for e in range(chunk_start, chunk_end):
                    # Find tokens for this expert
                    mask = (indices_flat[:, k] == e)
                    if not mask.any():
                        continue
                    
                    # Get token indices (with capacity limit)
                    token_indices = mask.nonzero(as_tuple=True)[0]
                    if len(token_indices) > capacity:
                        # Randomly sample to capacity
                        perm = torch.randperm(len(token_indices), device=x.device)[:capacity]
                        token_indices = token_indices[perm]
                    
                    # Expert forward
                    expert_input = x_flat[token_indices]
                    expert_output = self._expert_forward(expert_input, e)
                    
                    # Weight by routing probability
                    weights = probs_flat[token_indices, k:k+1]
                    output[token_indices] += weights * expert_output
        
        output = output.view(batch_size, seq_len, self.dim)
        
        return output, aux_loss


# Test parameters
batch_size = 4
seq_len = 256
dim = 4096
hidden_dim = 2048
num_experts = 256  # Large number of experts
top_k = 4

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
    return [dim, hidden_dim, num_experts, top_k]

