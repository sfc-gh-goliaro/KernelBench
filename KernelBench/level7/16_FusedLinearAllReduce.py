import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused Linear + All-Reduce (Tensor Parallel Row Linear).
    
    Combines the row-parallel linear computation with the all-reduce
    communication required to aggregate partial results.
    
    In tensor parallelism, row-parallel linear requires:
    1. Local matmul with weight shard
    2. All-reduce to sum partial results
    
    Fusion enables compute-communication overlap.
    
    Reference: Megatron-LM, DeepSpeed tensor parallelism
    """
    def __init__(self, in_features, out_features, world_size, rank, bias=True):
        """
        :param in_features: Total input features
        :param out_features: Output features
        :param world_size: Tensor parallel world size
        :param rank: Current rank
        :param bias: Use bias (only on rank 0)
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        
        # Each rank has portion of input dimension
        assert in_features % world_size == 0
        self.in_features_per_rank = in_features // world_size
        
        # Local weight shard
        self.weight = nn.Parameter(
            torch.randn(out_features, self.in_features_per_rank) * 0.02
        )
        
        # Bias only on rank 0
        if bias and rank == 0:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def _simulate_all_reduce(self, partial_outputs):
        """Simulate all-reduce across ranks."""
        # Sum all partial outputs
        return sum(partial_outputs)
    
    def forward(self, x, all_rank_inputs=None):
        """
        Fused linear + all-reduce.
        
        :param x: Local input shard (batch, seq, in_features_per_rank)
        :param all_rank_inputs: For simulation, inputs from all ranks
        :return: Full output after all-reduce
        """
        # === FUSED KERNEL START ===
        # In a true fused implementation:
        # 1. Compute local matmul
        # 2. Overlap communication with computation
        # 3. Accumulate results
        
        # Local computation
        local_output = F.linear(x, self.weight)
        
        # Simulate all-reduce (in practice, this overlaps with computation)
        if all_rank_inputs is not None:
            # Compute all partial outputs
            partial_outputs = []
            for r in range(self.world_size):
                # Each rank computes with its shard
                partial = F.linear(all_rank_inputs[r], self.weight)
                partial_outputs.append(partial)
            
            output = self._simulate_all_reduce(partial_outputs)
        else:
            output = local_output
        
        # Add bias (only on rank 0)
        if self.bias is not None:
            output = output + self.bias
        # === FUSED KERNEL END ===
        
        return output


# Chunked All-Reduce Linear (for overlap)
class FusedChunkedLinearAllReduce(nn.Module):
    """
    Chunked Linear + All-Reduce for compute-communication overlap.
    
    Splits the matmul into chunks and pipelines all-reduce.
    """
    def __init__(self, in_features, out_features, world_size, rank, 
                 num_chunks=4, bias=True):
        super(FusedChunkedLinearAllReduce, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.num_chunks = num_chunks
        
        self.in_features_per_rank = in_features // world_size
        assert out_features % num_chunks == 0
        self.chunk_size = out_features // num_chunks
        
        # Weight shards per chunk
        self.weight_chunks = nn.ParameterList([
            nn.Parameter(torch.randn(self.chunk_size, self.in_features_per_rank) * 0.02)
            for _ in range(num_chunks)
        ])
        
        if bias and rank == 0:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x, all_rank_inputs=None):
        """
        Chunked fused linear + all-reduce.
        
        Pipeline: compute chunk i while all-reducing chunk i-1
        """
        # === FUSED KERNEL WITH OVERLAP ===
        outputs = []
        
        for chunk_idx in range(self.num_chunks):
            # Compute chunk
            chunk_out = F.linear(x, self.weight_chunks[chunk_idx])
            
            # In fused kernel, all-reduce for previous chunk overlaps
            # with computation of current chunk
            outputs.append(chunk_out)
        
        # Concatenate chunks
        output = torch.cat(outputs, dim=-1)
        
        # Simulate all-reduce
        if all_rank_inputs is not None:
            # Would sum partial results from all ranks
            pass
        
        if self.bias is not None:
            output = output + self.bias
        # === END FUSED KERNEL ===
        
        return output


# AG-GEMM (All-Gather + GEMM fusion)
class FusedAllGatherGEMM(nn.Module):
    """
    Fused All-Gather + GEMM (Tensor Parallel Column Linear).
    
    For column-parallel linear where input needs to be gathered first.
    """
    def __init__(self, in_features, out_features, world_size, rank):
        super(FusedAllGatherGEMM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        
        # Each rank has portion of output dimension
        assert out_features % world_size == 0
        self.out_features_per_rank = out_features // world_size
        
        self.weight = nn.Parameter(
            torch.randn(self.out_features_per_rank, in_features) * 0.02
        )
    
    def forward(self, x_local, all_rank_x=None):
        """
        Fused all-gather + GEMM.
        
        :param x_local: Local input shard (after previous row-parallel)
        :param all_rank_x: Inputs from all ranks (for simulation)
        :return: Local output (column-parallel)
        """
        # === FUSED KERNEL ===
        if all_rank_x is not None:
            # All-gather input
            x_full = torch.cat(all_rank_x, dim=-1)
        else:
            x_full = x_local
        
        # Local matmul
        output = F.linear(x_full, self.weight)
        # === END FUSED KERNEL ===
        
        return output


# RS-GEMM (Reduce-Scatter + GEMM fusion)
class FusedReduceScatterGEMM(nn.Module):
    """
    Fused Reduce-Scatter + GEMM.
    
    Combines gradient reduce-scatter with weight gradient computation.
    Useful for ZeRO-style gradient sharding.
    """
    def __init__(self, in_features, out_features, world_size, rank):
        super(FusedReduceScatterGEMM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02
        )
    
    def forward(self, x, grad_output=None, all_rank_grads=None):
        """
        Fused forward or backward with reduce-scatter.
        """
        # Forward pass
        output = F.linear(x, self.weight)
        
        # In backward, would fuse reduce-scatter with gradient computation
        # grad_weight = reduce_scatter(grad_output.T @ x)
        
        return output


# Test parameters
batch_size = 32
seq_len = 512
in_features = 4096
out_features = 4096
world_size = 8
rank = 0

def get_inputs():
    # Local input shard
    in_per_rank = in_features // world_size
    x = torch.randn(batch_size, seq_len, in_per_rank)
    return [x]

def get_init_inputs():
    return [in_features, out_features, world_size, rank]

