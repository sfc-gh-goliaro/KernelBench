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
    Fused Top-K Gating + Token Permutation for MoE.
    
    Combines the routing decision with token permutation in a single pass:
    1. Compute gating scores
    2. Select top-k experts per token
    3. Permute tokens by expert assignment
    4. Compute routing weights
    
    This fusion eliminates intermediate memory traffic and is critical
    for efficient MoE inference.
    
    Reference: Megablocks, ScatterMoE, DeepSpeed-MoE
    """
    def __init__(self, hidden_dim, num_experts, top_k=2, normalize=True):
        """
        :param hidden_dim: Input hidden dimension
        :param num_experts: Number of experts
        :param top_k: Number of experts per token
        :param normalize: Whether to normalize routing weights
        """
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize = normalize
        
        # Gating layer
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
    
    def forward(self, x):
        """
        Fused gating + permutation.
        
        :param x: Input tokens (batch * seq, hidden_dim)
        :return: Tuple of:
            - permuted_tokens: Tokens reordered by expert (total_tokens * top_k, hidden_dim)
            - expert_indices: Expert assignment per permuted token
            - token_indices: Original token index per permuted token
            - routing_weights: Weight for each token-expert assignment
        """
        num_tokens = x.shape[0]
        device = x.device
        
        # === FUSED KERNEL START ===
        # Step 1: Compute gating logits
        gating_logits = self.gate(x)  # (num_tokens, num_experts)
        
        # Step 2: Top-k selection
        routing_weights, expert_indices = torch.topk(gating_logits, self.top_k, dim=-1)
        
        # Step 3: Apply softmax normalization
        if self.normalize:
            routing_weights = F.softmax(routing_weights, dim=-1)
        else:
            routing_weights = torch.sigmoid(routing_weights)
        
        # Step 4: Compute permutation indices
        # Flatten expert assignments: each token contributes top_k entries
        flat_expert_indices = expert_indices.view(-1)  # (num_tokens * top_k,)
        
        # Create token indices for each top-k assignment
        token_indices = torch.arange(num_tokens, device=device)
        token_indices = token_indices.unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        
        # Sort by expert for efficient batched processing
        sorted_expert_order = flat_expert_indices.argsort()
        
        # Permute everything according to expert order
        sorted_expert_indices = flat_expert_indices[sorted_expert_order]
        sorted_token_indices = token_indices[sorted_expert_order]
        sorted_weights = routing_weights.view(-1)[sorted_expert_order]
        
        # Gather permuted tokens
        permuted_tokens = x[sorted_token_indices]
        # === FUSED KERNEL END ===
        
        # Compute expert boundaries for grouped GEMM
        expert_counts = torch.bincount(sorted_expert_indices, minlength=self.num_experts)
        
        return permuted_tokens, sorted_expert_indices, sorted_token_indices, sorted_weights, expert_counts


# Variant with capacity limiting
class FusedTopKGatingWithCapacity(nn.Module):
    """
    Fused Top-K Gating + Permutation with capacity limiting.
    
    Enforces maximum tokens per expert to ensure load balance.
    """
    def __init__(self, hidden_dim, num_experts, top_k=2, capacity_factor=1.25):
        super(FusedTopKGatingWithCapacity, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
    
    def forward(self, x):
        """Gating with capacity limiting."""
        num_tokens = x.shape[0]
        device = x.device
        
        # Compute capacity per expert
        tokens_per_expert = (num_tokens * self.top_k) // self.num_experts
        capacity = int(tokens_per_expert * self.capacity_factor)
        
        # Gating
        gating_logits = self.gate(x)
        routing_weights, expert_indices = torch.topk(gating_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Flatten and count assignments
        flat_experts = expert_indices.view(-1)
        token_indices = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        
        # Apply capacity constraint
        expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        mask = torch.ones(num_tokens * self.top_k, dtype=torch.bool, device=device)
        
        for i in range(num_tokens * self.top_k):
            exp_idx = flat_experts[i].item()
            if expert_counts[exp_idx] >= capacity:
                mask[i] = False
            else:
                expert_counts[exp_idx] += 1
        
        # Apply mask
        kept_experts = flat_experts[mask]
        kept_tokens = token_indices[mask]
        kept_weights = routing_weights.view(-1)[mask]
        permuted_tokens = x[kept_tokens]
        
        return permuted_tokens, kept_experts, kept_tokens, kept_weights


# Grouped permutation for large batch sizes
class FusedGroupedGatingPermutation(nn.Module):
    """
    Fused Gating + Permutation optimized for grouped GEMM.
    
    Outputs directly usable by grouped matrix multiplication.
    """
    def __init__(self, hidden_dim, num_experts, top_k=2):
        super(FusedGroupedGatingPermutation, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
    
    def forward(self, x):
        """
        Output format optimized for grouped GEMM.
        
        Returns tensors ready for Megablocks-style grouped matmul.
        """
        num_tokens = x.shape[0]
        device = x.device
        
        # Gating
        logits = self.gate(x)
        weights, experts = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        # Create batched expert inputs
        expert_inputs = []
        expert_weights_list = []
        expert_token_ids = []
        
        for e in range(self.num_experts):
            mask = (experts == e).any(dim=-1)
            if mask.any():
                expert_inputs.append(x[mask])
                expert_token_ids.append(mask.nonzero(as_tuple=True)[0])
                
                # Get weights for this expert
                exp_weights = torch.where(experts[mask] == e, weights[mask], 
                                         torch.zeros_like(weights[mask]))
                expert_weights_list.append(exp_weights.sum(dim=-1))
        
        return expert_inputs, expert_weights_list, expert_token_ids


# Test parameters
batch_size = 32
seq_len = 512
hidden_dim = 4096
num_experts = 8
top_k = 2

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    return [torch.randn(batch_size * seq_len, hidden_dim)]

def get_init_inputs():
    return [hidden_dim, num_experts, top_k]

