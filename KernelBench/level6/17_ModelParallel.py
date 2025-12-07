import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

class Model(nn.Module):
    """
    Model Parallelism Partitioning.
    
    Partitions model layers/parameters across devices for large models.
    Combines tensor and pipeline parallelism strategies.
    
    Based on: Megatron-LM and Alpa automatic parallelization
    """
    def __init__(self, world_size, rank, tp_size=1, pp_size=1, dp_size=1):
        """
        :param world_size: Total number of ranks
        :param rank: Current rank
        :param tp_size: Tensor parallel size
        :param pp_size: Pipeline parallel size
        :param dp_size: Data parallel size
        """
        super(Model, self).__init__()
        self.world_size = world_size
        self.rank = rank
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = dp_size
        
        assert tp_size * pp_size * dp_size == world_size, \
            "tp_size * pp_size * dp_size must equal world_size"
        
        # Compute local rank indices
        self.tp_rank = rank % tp_size
        self.pp_rank = (rank // tp_size) % pp_size
        self.dp_rank = rank // (tp_size * pp_size)
        
        # Communication groups (simulated)
        self.tp_group = list(range(self.dp_rank * tp_size * pp_size + 
                                   self.pp_rank * tp_size,
                                   self.dp_rank * tp_size * pp_size + 
                                   self.pp_rank * tp_size + tp_size))
        
        self.pp_group = [self.dp_rank * tp_size * pp_size + 
                         p * tp_size + self.tp_rank 
                         for p in range(pp_size)]
        
        self.dp_group = [d * tp_size * pp_size + 
                         self.pp_rank * tp_size + self.tp_rank 
                         for d in range(dp_size)]
    
    def get_parallel_config(self):
        """Get current rank's parallel configuration."""
        return {
            'tp_rank': self.tp_rank,
            'pp_rank': self.pp_rank,
            'dp_rank': self.dp_rank,
            'tp_group': self.tp_group,
            'pp_group': self.pp_group,
            'dp_group': self.dp_group
        }
    
    def partition_layers(self, num_layers):
        """
        Partition layers across pipeline stages.
        
        :param num_layers: Total number of layers
        :return: (start_layer, end_layer) for this rank
        """
        layers_per_stage = num_layers // self.pp_size
        start = self.pp_rank * layers_per_stage
        end = start + layers_per_stage if self.pp_rank < self.pp_size - 1 else num_layers
        return start, end
    
    def partition_hidden_dim(self, hidden_dim):
        """
        Partition hidden dimension for tensor parallelism.
        
        :param hidden_dim: Total hidden dimension
        :return: Local hidden dimension slice
        """
        assert hidden_dim % self.tp_size == 0
        local_dim = hidden_dim // self.tp_size
        start = self.tp_rank * local_dim
        return start, start + local_dim
    
    def forward(self, x, layer_fns, communication_fn=None):
        """
        Execute model parallel forward.
        
        :param x: Input tensor
        :param layer_fns: List of layer forward functions
        :param communication_fn: Optional communication callback
        :return: Output tensor
        """
        start_layer, end_layer = self.partition_layers(len(layer_fns))
        
        # Receive from previous pipeline stage
        if self.pp_rank > 0 and communication_fn:
            x = communication_fn('recv', self.pp_group[self.pp_rank - 1], x)
        
        # Execute local layers
        for layer_idx in range(start_layer, end_layer):
            x = layer_fns[layer_idx](x)
        
        # Send to next pipeline stage
        if self.pp_rank < self.pp_size - 1 and communication_fn:
            communication_fn('send', self.pp_group[self.pp_rank + 1], x)
        
        return x


# 3D Parallelism (TP + PP + DP)
class ThreeDParallelism(nn.Module):
    """
    3D Parallelism combining Tensor, Pipeline, and Data parallelism.
    
    Optimal for very large models across many GPUs.
    """
    def __init__(self, world_size, rank, tp_size, pp_size, hidden_dim, num_layers):
        super(ThreeDParallelism, self).__init__()
        self.world_size = world_size
        self.rank = rank
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = world_size // (tp_size * pp_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Compute ranks
        self.tp_rank = rank % tp_size
        self.pp_rank = (rank // tp_size) % pp_size
        self.dp_rank = rank // (tp_size * pp_size)
        
        # Local layer partition
        layers_per_stage = num_layers // pp_size
        self.start_layer = self.pp_rank * layers_per_stage
        self.end_layer = (self.start_layer + layers_per_stage 
                         if self.pp_rank < pp_size - 1 else num_layers)
        
        # Local hidden dimension
        assert hidden_dim % tp_size == 0
        self.local_hidden = hidden_dim // tp_size
        
        # Create local layers
        self.layers = nn.ModuleList([
            nn.Linear(self.local_hidden, self.local_hidden)
            for _ in range(self.end_layer - self.start_layer)
        ])
    
    def forward(self, x):
        """3D parallel forward pass."""
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


# Expert Parallel + Data Parallel
class ExpertDataParallel(nn.Module):
    """
    Combined Expert and Data Parallelism for MoE.
    
    Experts distributed across EP dimension, replicated across DP.
    """
    def __init__(self, num_experts, ep_size, dp_size, hidden_dim, expert_dim):
        super(ExpertDataParallel, self).__init__()
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.dp_size = dp_size
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        
        assert num_experts % ep_size == 0
        self.experts_per_rank = num_experts // ep_size
        
        # Local experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, expert_dim),
                nn.SiLU(),
                nn.Linear(expert_dim, hidden_dim)
            )
            for _ in range(self.experts_per_rank)
        ])
    
    def forward(self, x, expert_indices):
        """
        Expert + Data parallel forward.
        
        :param x: Input tokens
        :param expert_indices: Expert routing indices
        :return: Processed tokens
        """
        # Route tokens to local experts
        outputs = torch.zeros_like(x)
        
        for local_idx, expert in enumerate(self.experts):
            global_idx = local_idx  # Simplified
            mask = expert_indices == global_idx
            if mask.any():
                outputs[mask] = expert(x[mask])
        
        return outputs


# Test parameters
world_size = 8
rank = 0
tp_size = 2
pp_size = 2
dp_size = 2
hidden_dim = 4096
num_layers = 32

def get_inputs():
    x = torch.randn(32, 512, hidden_dim // tp_size)
    layer_fns = [lambda x: F.relu(x) for _ in range(num_layers)]
    return [x, layer_fns]

def get_init_inputs():
    return [world_size, rank, tp_size, pp_size, dp_size]

