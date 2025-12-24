import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Iterator


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    ZeRO (Zero Redundancy Optimizer) Sharding.
    
    Implements ZeRO stages 1-3 for memory-efficient distributed training.
    Shards optimizer states, gradients, and parameters across ranks.
    
    Based on: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
    """
    def __init__(self, world_size, rank, stage=2):
        """
        :param world_size: Number of ranks
        :param rank: Current rank
        :param stage: ZeRO stage (1, 2, or 3)
        """
        super(Model, self).__init__()
        self.world_size = world_size
        self.rank = rank
        self.stage = stage
        
        # Simulated sharded storage
        self.sharded_optimizer_states = {}  # Stage 1+
        self.sharded_gradients = {}          # Stage 2+
        self.sharded_parameters = {}         # Stage 3
    
    def shard_optimizer_state(self, param_name, state_dict):
        """
        Shard optimizer state across ranks (Stage 1).
        
        :param param_name: Parameter name
        :param state_dict: Full optimizer state
        :return: Local shard of optimizer state
        """
        full_size = sum(v.numel() for v in state_dict.values() if torch.is_tensor(v))
        shard_size = full_size // self.world_size
        start = self.rank * shard_size
        end = start + shard_size if self.rank < self.world_size - 1 else full_size
        
        # Flatten and shard
        sharded = {}
        for key, value in state_dict.items():
            if torch.is_tensor(value):
                flat = value.view(-1)
                param_start = max(0, start)
                param_end = min(flat.numel(), end)
                if param_start < flat.numel() and param_end > 0:
                    local_start = max(0, param_start - start)
                    local_end = local_start + (min(param_end, flat.numel()) - param_start)
                    sharded[key] = flat[param_start:param_end]
        
        self.sharded_optimizer_states[param_name] = sharded
        return sharded
    
    def shard_gradient(self, param_name, gradient):
        """
        Shard gradient across ranks (Stage 2).
        
        :param param_name: Parameter name
        :param gradient: Full gradient tensor
        :return: Local gradient shard
        """
        flat_grad = gradient.view(-1)
        shard_size = flat_grad.numel() // self.world_size
        start = self.rank * shard_size
        end = start + shard_size if self.rank < self.world_size - 1 else flat_grad.numel()
        
        shard = flat_grad[start:end].clone()
        self.sharded_gradients[param_name] = shard
        return shard
    
    def shard_parameter(self, param_name, parameter):
        """
        Shard parameter across ranks (Stage 3).
        
        :param param_name: Parameter name
        :param parameter: Full parameter tensor
        :return: Local parameter shard
        """
        flat_param = parameter.view(-1)
        shard_size = flat_param.numel() // self.world_size
        start = self.rank * shard_size
        end = start + shard_size if self.rank < self.world_size - 1 else flat_param.numel()
        
        shard = flat_param[start:end].clone()
        self.sharded_parameters[param_name] = shard
        return shard
    
    def all_gather_parameter(self, param_name, all_shards):
        """
        All-gather parameter shards (Stage 3).
        
        :param param_name: Parameter name
        :param all_shards: List of shards from all ranks
        :return: Full parameter
        """
        return torch.cat(all_shards, dim=0)
    
    def reduce_scatter_gradient(self, param_name, all_gradients):
        """
        Reduce-scatter gradients (Stage 2).
        
        :param param_name: Parameter name
        :param all_gradients: List of full gradients from all ranks
        :return: Reduced local gradient shard
        """
        # Stack and reduce
        stacked = torch.stack(all_gradients, dim=0)
        reduced = stacked.sum(dim=0)
        
        # Scatter
        shard_size = reduced.numel() // self.world_size
        start = self.rank * shard_size
        end = start + shard_size if self.rank < self.world_size - 1 else reduced.numel()
        
        return reduced[start:end]
    
    def forward(self, gradients_per_param):
        """
        Execute ZeRO sharding and communication.
        
        :param gradients_per_param: Dict mapping param names to gradients
        :return: Sharded gradients for this rank
        """
        if self.stage >= 2:
            # Reduce-scatter gradients
            sharded_grads = {}
            for name, grad in gradients_per_param.items():
                # Simulate all-ranks having the gradient
                all_grads = [grad.clone() for _ in range(self.world_size)]
                sharded_grads[name] = self.reduce_scatter_gradient(name, all_grads)
            return sharded_grads
        else:
            # Stage 1: only shard optimizer states
            return gradients_per_param


# ZeRO++ with Hierarchical Partitioning
class ZeROPlusPlus(nn.Module):
    """
    ZeRO++ with hierarchical partitioning.
    
    Uses intra-node and inter-node communication optimization
    for better scaling.
    """
    def __init__(self, world_size, rank, local_world_size, stage=3):
        super(ZeROPlusPlus, self).__init__()
        self.world_size = world_size
        self.rank = rank
        self.local_world_size = local_world_size  # Ranks per node
        self.num_nodes = world_size // local_world_size
        self.local_rank = rank % local_world_size
        self.node_id = rank // local_world_size
        self.stage = stage
    
    def hierarchical_reduce_scatter(self, gradients):
        """
        Two-level reduce-scatter: intra-node then inter-node.
        
        :param gradients: List of gradients from all ranks
        :return: Local gradient shard
        """
        # First: intra-node reduce
        node_gradients = []
        for node in range(self.num_nodes):
            start = node * self.local_world_size
            end = start + self.local_world_size
            node_sum = sum(gradients[start:end])
            node_gradients.append(node_sum)
        
        # Second: inter-node reduce-scatter
        inter_node_reduced = sum(node_gradients)
        
        # Scatter to ranks
        shard_size = inter_node_reduced.numel() // self.world_size
        start = self.rank * shard_size
        end = start + shard_size
        
        return inter_node_reduced.view(-1)[start:end]
    
    def quantized_all_gather(self, parameter_shard, bits=8):
        """
        Quantized all-gather for bandwidth reduction.
        
        :param parameter_shard: Local parameter shard
        :param bits: Quantization bits
        :return: Full parameter (dequantized)
        """
        # Quantize for communication
        scale = parameter_shard.abs().max() / (2 ** (bits - 1) - 1)
        quantized = (parameter_shard / scale).round().clamp(
            -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        )
        
        # Simulate all-gather of quantized values
        all_quantized = [quantized.clone() for _ in range(self.world_size)]
        
        # Dequantize
        all_dequantized = [q * scale for q in all_quantized]
        
        return torch.cat(all_dequantized, dim=0)
    
    def forward(self, gradients):
        """
        ZeRO++ forward with hierarchical communication.
        """
        return self.hierarchical_reduce_scatter(gradients)


# Test parameters
world_size = 8
rank = 0
hidden_dim = 4096

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    gradients = {'layer.weight': torch.randn(1024, hidden_dim)}
    return [gradients]

def get_init_inputs():
    return [world_size, rank, 2]  # stage 2
