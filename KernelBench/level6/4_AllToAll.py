import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    All-to-All Collective Communication.
    
    Each rank sends different data to every other rank and receives
    different data from every rank. Critical for MoE expert parallelism
    where tokens need to be routed to different experts on different ranks.
    
    Simulates the communication pattern for benchmarking purposes.
    """
    def __init__(self, world_size, split_dim=0, concat_dim=0):
        """
        :param world_size: Number of parallel ranks
        :param split_dim: Dimension along which to split for sending
        :param concat_dim: Dimension along which to concatenate received data
        """
        super(Model, self).__init__()
        self.world_size = world_size
        self.split_dim = split_dim
        self.concat_dim = concat_dim
    
    def forward(self, tensors):
        """
        Simulate all-to-all across ranks.
        
        :param tensors: List of tensors from each rank, each split into world_size parts
        :return: List of tensors where each rank has gathered its portion from all ranks
        """
        assert len(tensors) == self.world_size, \
            f"Expected {self.world_size} tensors, got {len(tensors)}"
        
        # Split each tensor into world_size parts
        split_size = tensors[0].shape[self.split_dim] // self.world_size
        splits = [torch.split(t, split_size, dim=self.split_dim) for t in tensors]
        
        # Each rank receives one part from every other rank
        results = []
        for recv_rank in range(self.world_size):
            # Gather parts destined for recv_rank from all senders
            parts_for_rank = [splits[send_rank][recv_rank] for send_rank in range(self.world_size)]
            # Concatenate along concat_dim
            gathered = torch.cat(parts_for_rank, dim=self.concat_dim)
            results.append(gathered)
        
        return results


# All-to-All for MoE Expert Routing
class MoEAllToAll(nn.Module):
    """
    MoE-specific All-to-All for expert parallelism.
    
    Routes tokens to experts across ranks based on routing decisions.
    Handles variable number of tokens per expert.
    """
    def __init__(self, world_size, num_experts_per_rank):
        super(MoEAllToAll, self).__init__()
        self.world_size = world_size
        self.num_experts_per_rank = num_experts_per_rank
        self.total_experts = world_size * num_experts_per_rank
    
    def forward(self, tokens, expert_assignments):
        """
        Route tokens to appropriate expert ranks.
        
        :param tokens: List of token tensors from each rank (batch, dim)
        :param expert_assignments: List of expert indices for each token per rank
        :return: Tuple of (routed_tokens, token_counts) per rank
        """
        results = [[] for _ in range(self.world_size)]
        counts = [[0] * self.num_experts_per_rank for _ in range(self.world_size)]
        
        for send_rank in range(self.world_size):
            rank_tokens = tokens[send_rank]
            rank_assignments = expert_assignments[send_rank]
            
            for token_idx, expert_idx in enumerate(rank_assignments):
                # Determine which rank owns this expert
                recv_rank = expert_idx // self.num_experts_per_rank
                local_expert = expert_idx % self.num_experts_per_rank
                
                results[recv_rank].append(rank_tokens[token_idx])
                counts[recv_rank][local_expert] += 1
        
        # Stack tokens for each rank
        routed = []
        for rank in range(self.world_size):
            if results[rank]:
                routed.append(torch.stack(results[rank], dim=0))
            else:
                routed.append(torch.empty(0, tokens[0].shape[-1]))
        
        return routed, counts


# Test parameters
world_size = 8
tokens_per_rank = 1024
hidden_dim = 512

def get_inputs():
    # Each rank has tokens to route to different experts
    tensors = [torch.randn(tokens_per_rank, hidden_dim) for _ in range(world_size)]
    return [tensors]

def get_init_inputs():
    return [world_size, 0, 0]

