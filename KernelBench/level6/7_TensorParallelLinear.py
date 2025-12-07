import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Tensor Parallel Linear Layer (Column Parallel).
    
    Splits the output dimension across ranks. Each rank computes
    a portion of the output, then results are gathered.
    
    Based on: Megatron-LM tensor parallelism
    """
    def __init__(self, in_features, out_features, world_size, rank, 
                 gather_output=True, bias=True):
        """
        :param in_features: Input dimension
        :param out_features: Total output dimension
        :param world_size: Number of tensor parallel ranks
        :param rank: Current rank
        :param gather_output: Whether to gather output across ranks
        :param bias: Whether to use bias
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.gather_output = gather_output
        
        # Each rank has a portion of the output dimension
        assert out_features % world_size == 0
        self.out_features_per_rank = out_features // world_size
        
        # Local weight shard
        self.weight = nn.Parameter(
            torch.randn(self.out_features_per_rank, in_features) * 0.02
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features_per_rank))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x, all_rank_outputs=None):
        """
        Column-parallel linear forward.
        
        :param x: Input tensor (batch, seq, in_features)
        :param all_rank_outputs: For simulation, list to collect outputs from all ranks
        :return: Local output (batch, seq, out_features_per_rank)
                 or gathered output if gather_output=True
        """
        # Local linear computation
        local_output = F.linear(x, self.weight, self.bias)
        
        if self.gather_output and all_rank_outputs is not None:
            # Simulate all-gather
            all_rank_outputs.append(local_output)
            if len(all_rank_outputs) == self.world_size:
                # Concatenate outputs from all ranks
                gathered = torch.cat(all_rank_outputs, dim=-1)
                return gathered
        
        return local_output


# Row Parallel Linear
class RowParallelLinear(nn.Module):
    """
    Tensor Parallel Linear Layer (Row Parallel).
    
    Splits the input dimension across ranks. Each rank has a portion
    of the weight matrix. Results are reduce-scattered or all-reduced.
    """
    def __init__(self, in_features, out_features, world_size, rank,
                 input_is_parallel=True, bias=True):
        """
        :param in_features: Total input dimension
        :param out_features: Output dimension
        :param world_size: Number of tensor parallel ranks
        :param rank: Current rank
        :param input_is_parallel: Whether input is already split across ranks
        :param bias: Whether to use bias (only on rank 0 typically)
        """
        super(RowParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.input_is_parallel = input_is_parallel
        
        # Each rank has a portion of the input dimension
        assert in_features % world_size == 0
        self.in_features_per_rank = in_features // world_size
        
        # Local weight shard
        self.weight = nn.Parameter(
            torch.randn(out_features, self.in_features_per_rank) * 0.02
        )
        
        # Bias only on one rank to avoid duplication
        if bias and rank == 0:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x, all_rank_outputs=None):
        """
        Row-parallel linear forward.
        
        :param x: Input tensor - either full (batch, seq, in_features) if not input_is_parallel
                  or partial (batch, seq, in_features_per_rank) if input_is_parallel
        :param all_rank_outputs: For simulation, list to collect for all-reduce
        :return: Output after reduction (batch, seq, out_features)
        """
        if not self.input_is_parallel:
            # Split input across ranks
            x = x[..., self.rank * self.in_features_per_rank:
                     (self.rank + 1) * self.in_features_per_rank]
        
        # Local linear computation
        local_output = F.linear(x, self.weight)
        
        if all_rank_outputs is not None:
            # Simulate all-reduce
            all_rank_outputs.append(local_output)
            if len(all_rank_outputs) == self.world_size:
                # Sum outputs from all ranks
                reduced = sum(all_rank_outputs)
                if self.bias is not None:
                    reduced = reduced + self.bias
                return reduced
        
        return local_output


# Test parameters
batch_size = 32
seq_len = 512
in_features = 4096
out_features = 4096
world_size = 8
rank = 0

def get_inputs():
    x = torch.randn(batch_size, seq_len, in_features)
    return [x]

def get_init_inputs():
    return [in_features, out_features, world_size, rank]

