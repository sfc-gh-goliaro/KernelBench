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
    Fused GEMM + Reduce-Scatter.
    
    Combines matrix multiplication with reduce-scatter for tensor parallelism:
    1. Compute local GEMM
    2. Reduce partial results across ranks
    3. Scatter result portions to respective ranks
    
    Used in row-parallel linear where output needs to be distributed.
    Enables overlapping computation with communication.
    
    Reference: Megatron-LM, DeepSpeed tensor parallelism
    """
    def __init__(self, in_features, out_features, world_size, rank, bias=True):
        """
        :param in_features: Input dimension (full)
        :param out_features: Output dimension (full)
        :param world_size: Tensor parallel world size
        :param rank: Current rank
        :param bias: Use bias
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        
        # Each rank has a shard of input dimension
        assert in_features % world_size == 0
        self.in_per_rank = in_features // world_size
        
        # Output is split across ranks after reduce-scatter
        assert out_features % world_size == 0
        self.out_per_rank = out_features // world_size
        
        # Local weight: (out_features, in_per_rank) -> after reduce-scatter: (out_per_rank,)
        self.weight = nn.Parameter(
            torch.randn(out_features, self.in_per_rank) * 0.02
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_per_rank))
        else:
            self.register_parameter('bias', None)
    
    def _simulate_reduce_scatter(self, partial_outputs, all_rank_partials=None):
        """
        Simulate reduce-scatter operation.
        
        :param partial_outputs: Local partial output
        :param all_rank_partials: Partials from all ranks
        :return: This rank's portion of reduced output
        """
        if all_rank_partials is None:
            # Just return local portion
            return partial_outputs[..., self.rank * self.out_per_rank:
                                      (self.rank + 1) * self.out_per_rank]
        
        # Reduce (sum all partials)
        reduced = sum(all_rank_partials)
        
        # Scatter (each rank gets its portion)
        return reduced[..., self.rank * self.out_per_rank:
                          (self.rank + 1) * self.out_per_rank]
    
    def forward(self, x, all_rank_inputs=None):
        """
        Fused GEMM + Reduce-Scatter.
        
        :param x: Local input shard (batch, seq, in_per_rank)
        :param all_rank_inputs: Inputs from all ranks for simulation
        :return: Scattered output (batch, seq, out_per_rank)
        """
        # === FUSED KERNEL START ===
        # Step 1: Local GEMM
        local_output = F.linear(x, self.weight)  # (batch, seq, out_features)
        
        # Step 2: Reduce-Scatter
        # In fused kernel, this overlaps with GEMM computation
        if all_rank_inputs is not None:
            # Compute all partials
            all_partials = [
                F.linear(inp, self.weight) 
                for inp in all_rank_inputs
            ]
            output = self._simulate_reduce_scatter(local_output, all_partials)
        else:
            output = self._simulate_reduce_scatter(local_output)
        
        # Step 3: Add bias (only on scattered portion)
        if self.bias is not None:
            output = output + self.bias
        # === FUSED KERNEL END ===
        
        return output


# Chunked GEMM + Reduce-Scatter for overlap
class FusedChunkedGEMMReduceScatter(nn.Module):
    """
    Chunked GEMM + Reduce-Scatter for compute-communication overlap.
    
    Splits GEMM into chunks and pipelines reduce-scatter.
    """
    def __init__(self, in_features, out_features, world_size, rank, num_chunks=4):
        super(FusedChunkedGEMMReduceScatter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.num_chunks = num_chunks
        
        self.in_per_rank = in_features // world_size
        self.out_per_rank = out_features // world_size
        
        # Split weight into chunks
        assert out_features % num_chunks == 0
        self.chunk_size = out_features // num_chunks
        
        self.weight_chunks = nn.ParameterList([
            nn.Parameter(torch.randn(self.chunk_size, self.in_per_rank) * 0.02)
            for _ in range(num_chunks)
        ])
    
    def forward(self, x, all_rank_inputs=None):
        """
        Chunked fused GEMM + reduce-scatter.
        
        Pipeline: Compute chunk i while reduce-scatter chunk i-1
        """
        # === FUSED KERNEL WITH OVERLAP ===
        output_chunks = []
        
        for chunk_idx in range(self.num_chunks):
            # Compute this chunk's GEMM
            chunk_output = F.linear(x, self.weight_chunks[chunk_idx])
            
            # Simulate reduce-scatter for this chunk
            # (In real impl, this overlaps with next chunk's compute)
            if all_rank_inputs is not None:
                all_chunk_partials = [
                    F.linear(inp, self.weight_chunks[chunk_idx])
                    for inp in all_rank_inputs
                ]
                reduced_chunk = sum(all_chunk_partials)
            else:
                reduced_chunk = chunk_output
            
            # This rank's portion of this chunk
            chunk_start = (chunk_idx * self.chunk_size * self.rank) // self.world_size
            chunk_portion = reduced_chunk[..., :self.chunk_size // self.world_size]
            output_chunks.append(chunk_portion)
        
        return torch.cat(output_chunks, dim=-1)


# For ZeRO-style gradient sharding
class FusedGradientGEMMReduceScatter(nn.Module):
    """
    Fused Gradient GEMM + Reduce-Scatter for ZeRO training.
    
    Combines weight gradient computation with gradient sharding.
    """
    def __init__(self, in_features, out_features, world_size, rank):
        super(FusedGradientGEMMReduceScatter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02
        )
    
    def forward(self, x):
        """Forward pass."""
        return F.linear(x, self.weight)
    
    def compute_weight_grad_sharded(self, grad_output, input_activations, 
                                    all_rank_grads=None):
        """
        Compute weight gradient with reduce-scatter.
        
        :param grad_output: Gradient of output
        :param input_activations: Saved input
        :param all_rank_grads: Gradients from all ranks
        :return: This rank's shard of weight gradient
        """
        # grad_weight = grad_output.T @ input_activations
        batch_seq = grad_output.shape[0] * grad_output.shape[1]
        grad_out_flat = grad_output.view(batch_seq, -1)
        input_flat = input_activations.view(batch_seq, -1)
        
        # Local gradient
        local_grad = grad_out_flat.t() @ input_flat
        
        # Reduce-scatter
        if all_rank_grads is not None:
            reduced = sum(all_rank_grads)
        else:
            reduced = local_grad
        
        # Return this rank's shard
        shard_size = reduced.numel() // self.world_size
        start = self.rank * shard_size
        end = start + shard_size
        
        return reduced.view(-1)[start:end]


# Test parameters
batch_size = 32
seq_len = 512
in_features = 4096
out_features = 4096
world_size = 8
rank = 0

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    in_per_rank = in_features // world_size
    x = torch.randn(batch_size, seq_len, in_per_rank)
    return [x]

def get_init_inputs():
    return [in_features, out_features, world_size, rank]

