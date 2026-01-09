import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Expert Parallel AllToAll
    
    Used by: DeepSeek-V2/V3, Mixtral with expert parallelism
    
    AllToAll operation for MoE expert parallelism.
    Redistributes tokens from tensor parallel layout to expert parallel layout.
    Tokens are sent to ranks that own their selected experts.
    
    Shapes:
        Input: (num_local_tokens, hidden_size) with routing info
        Output: (num_tokens_for_local_experts, hidden_size)
    """
    
    def __init__(self, hidden_size: int = 4096, num_experts: int = 64,
                 world_size: int = 8):
        """
        Initialize Expert Parallel AllToAll.
        
        Args:
            hidden_size: Token hidden dimension
            num_experts: Total number of experts
            world_size: Number of expert parallel ranks
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.world_size = world_size
        self.experts_per_rank = num_experts // world_size
    
    def forward(self, tokens: torch.Tensor, 
                expert_indices: torch.Tensor) -> tuple:
        """
        Simulate expert parallel all-to-all communication.
        
        In actual distributed setting:
        1. Partition tokens by destination expert rank
        2. AllToAll exchange tokens
        3. Each rank receives tokens for its local experts
        
        Args:
            tokens: Token hidden states (num_tokens, hidden_size)
            expert_indices: Selected expert for each token (num_tokens,)
            
        Returns:
            Tuple of (redistributed_tokens, token_counts_per_expert)
        """
        num_tokens = tokens.shape[0]
        device = tokens.device
        
        # Compute which rank owns each expert
        expert_to_rank = expert_indices // self.experts_per_rank
        
        # Count tokens going to each rank
        tokens_per_rank = torch.zeros(self.world_size, dtype=torch.long, device=device)
        for rank in range(self.world_size):
            tokens_per_rank[rank] = (expert_to_rank == rank).sum()
        
        # Simulate redistribution (actual impl uses NCCL all_to_all)
        # Sort tokens by destination rank for simulation
        sorted_indices = expert_to_rank.argsort()
        redistributed_tokens = tokens[sorted_indices]
        
        # Local expert indices after redistribution
        local_expert_indices = expert_indices[sorted_indices] % self.experts_per_rank
        
        return redistributed_tokens, local_expert_indices, tokens_per_rank


# ============================================================================
# Benchmark Configuration
# ============================================================================

num_tokens = 8192
hidden_size = 4096
num_experts = 64
world_size = 8

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    tokens = torch.randn(num_tokens, hidden_size, device='cuda')
    expert_indices = torch.randint(0, num_experts, (num_tokens,), device='cuda')
    return [tokens, expert_indices]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, num_experts, world_size]

