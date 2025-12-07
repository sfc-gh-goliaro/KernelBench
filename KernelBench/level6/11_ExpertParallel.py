import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class Model(nn.Module):
    """
    Expert Parallelism Communication for MoE.
    
    Distributes experts across ranks and routes tokens using
    all-to-all communication. Essential for large MoE models
    like Mixtral, DeepSeek-V3, and MiniMax-M2.
    
    Based on: GShard, Switch Transformer, and DeepSeek expert parallelism
    """
    def __init__(self, world_size, rank, num_experts_per_rank, hidden_dim, expert_dim):
        """
        :param world_size: Number of expert parallel ranks
        :param rank: Current rank
        :param num_experts_per_rank: Number of experts per rank
        :param hidden_dim: Input/output dimension
        :param expert_dim: Expert intermediate dimension
        """
        super(Model, self).__init__()
        self.world_size = world_size
        self.rank = rank
        self.num_experts_per_rank = num_experts_per_rank
        self.total_experts = world_size * num_experts_per_rank
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        
        # Local experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, expert_dim),
                nn.SiLU(),
                nn.Linear(expert_dim, hidden_dim)
            )
            for _ in range(num_experts_per_rank)
        ])
    
    def dispatch(self, tokens, expert_indices, expert_weights):
        """
        Dispatch tokens to experts across ranks (All-to-All).
        
        :param tokens: Token tensors from all ranks [(num_tokens, hidden_dim), ...]
        :param expert_indices: Expert assignments per token per rank
        :param expert_weights: Routing weights per token per rank
        :return: Tokens grouped by local expert
        """
        # Group tokens by destination rank
        send_buffers = [[] for _ in range(self.world_size)]
        send_weights = [[] for _ in range(self.world_size)]
        send_indices = [[] for _ in range(self.world_size)]
        token_origins = [[] for _ in range(self.world_size)]  # Track where tokens came from
        
        for src_rank in range(self.world_size):
            for token_idx, (expert_idx, weight) in enumerate(
                zip(expert_indices[src_rank], expert_weights[src_rank])
            ):
                dst_rank = expert_idx // self.num_experts_per_rank
                local_expert_idx = expert_idx % self.num_experts_per_rank
                
                send_buffers[dst_rank].append(tokens[src_rank][token_idx])
                send_weights[dst_rank].append(weight)
                send_indices[dst_rank].append(local_expert_idx)
                token_origins[dst_rank].append((src_rank, token_idx))
        
        # Convert to tensors
        local_tokens = {}
        local_weights = {}
        local_origins = {}
        
        for expert_idx in range(self.num_experts_per_rank):
            expert_tokens = []
            expert_weights_list = []
            expert_origins = []
            
            for dst_rank, idx_list in enumerate(send_indices):
                for i, local_idx in enumerate(idx_list):
                    if local_idx == expert_idx and dst_rank == self.rank:
                        expert_tokens.append(send_buffers[dst_rank][i])
                        expert_weights_list.append(send_weights[dst_rank][i])
                        expert_origins.append(token_origins[dst_rank][i])
            
            if expert_tokens:
                local_tokens[expert_idx] = torch.stack(expert_tokens)
                local_weights[expert_idx] = torch.tensor(expert_weights_list)
                local_origins[expert_idx] = expert_origins
        
        return local_tokens, local_weights, local_origins
    
    def combine(self, expert_outputs, expert_weights, token_origins, 
                original_shapes):
        """
        Combine expert outputs back to original token positions (All-to-All).
        
        :param expert_outputs: Dict of outputs per expert
        :param expert_weights: Dict of weights per expert
        :param token_origins: Dict mapping to original positions
        :param original_shapes: Original token counts per rank
        :return: Combined outputs per rank
        """
        results = [torch.zeros(count, self.hidden_dim) 
                   for count in original_shapes]
        
        for expert_idx, outputs in expert_outputs.items():
            origins = token_origins[expert_idx]
            weights = expert_weights[expert_idx]
            
            for i, (src_rank, token_idx) in enumerate(origins):
                results[src_rank][token_idx] += weights[i] * outputs[i]
        
        return results
    
    def forward(self, all_tokens, all_expert_indices, all_expert_weights):
        """
        Full expert-parallel forward pass.
        
        :param all_tokens: List of token tensors per rank
        :param all_expert_indices: List of expert indices per rank
        :param all_expert_weights: List of routing weights per rank
        :return: List of output tensors per rank
        """
        original_shapes = [t.shape[0] for t in all_tokens]
        
        # Dispatch (All-to-All)
        local_tokens, local_weights, local_origins = self.dispatch(
            all_tokens, all_expert_indices, all_expert_weights
        )
        
        # Process local experts
        expert_outputs = {}
        for expert_idx, tokens in local_tokens.items():
            expert_outputs[expert_idx] = self.experts[expert_idx](tokens)
        
        # Combine (All-to-All)
        combined = self.combine(
            expert_outputs, local_weights, local_origins, original_shapes
        )
        
        return combined


# Hierarchical Expert Parallelism (for large-scale MoE)
class HierarchicalExpertParallel(nn.Module):
    """
    Hierarchical Expert Parallelism for trillion-parameter MoE.
    
    Uses two-level hierarchy: intra-node and inter-node expert groups.
    """
    def __init__(self, num_nodes, ranks_per_node, experts_per_rank, hidden_dim):
        super(HierarchicalExpertParallel, self).__init__()
        self.num_nodes = num_nodes
        self.ranks_per_node = ranks_per_node
        self.experts_per_rank = experts_per_rank
        self.hidden_dim = hidden_dim
        
        self.total_ranks = num_nodes * ranks_per_node
        self.experts_per_node = ranks_per_node * experts_per_rank
        self.total_experts = num_nodes * self.experts_per_node
    
    def route_hierarchical(self, tokens, expert_indices):
        """
        Two-level routing: first to node, then to expert within node.
        """
        node_assignments = expert_indices // self.experts_per_node
        local_expert_idx = expert_indices % self.experts_per_node
        
        return node_assignments, local_expert_idx
    
    def forward(self, tokens, expert_indices):
        """Hierarchical routing forward."""
        node_assignments, local_indices = self.route_hierarchical(
            tokens, expert_indices
        )
        
        # First All-to-All: route to nodes
        # Second All-to-All: route within nodes
        # (Simulated here)
        
        return tokens  # Placeholder


# Test parameters
world_size = 8
rank = 0
num_experts_per_rank = 8
hidden_dim = 512
expert_dim = 2048
tokens_per_rank = 256

def get_inputs():
    # Tokens from each rank
    all_tokens = [torch.randn(tokens_per_rank, hidden_dim) for _ in range(world_size)]
    # Random expert assignments
    total_experts = world_size * num_experts_per_rank
    all_expert_indices = [
        torch.randint(0, total_experts, (tokens_per_rank,)).tolist()
        for _ in range(world_size)
    ]
    all_expert_weights = [
        torch.rand(tokens_per_rank).tolist() for _ in range(world_size)
    ]
    return [all_tokens, all_expert_indices, all_expert_weights]

def get_init_inputs():
    return [world_size, rank, num_experts_per_rank, hidden_dim, expert_dim]

