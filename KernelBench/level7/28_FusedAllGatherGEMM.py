import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused All-Gather + GEMM.
    
    Combines all-gather communication with subsequent matrix multiplication:
    1. All-gather input shards from all ranks
    2. Perform GEMM with gathered input
    
    Used in column-parallel linear where input needs to be gathered
    before projection to local output shard.
    
    Enables overlapping all-gather with GEMM computation.
    
    Reference: Megatron-LM, DeepSpeed tensor parallelism
    """
    def __init__(self, in_features, out_features, world_size, rank, bias=True):
        """
        :param in_features: Total input dimension
        :param out_features: Output dimension (each rank produces full output)
        :param world_size: Tensor parallel world size
        :param rank: Current rank
        :param bias: Use bias
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        
        # Each rank has a portion of output dimension
        assert out_features % world_size == 0
        self.out_per_rank = out_features // world_size
        
        # Local weight: (out_per_rank, in_features)
        self.weight = nn.Parameter(
            torch.randn(self.out_per_rank, in_features) * 0.02
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_per_rank))
        else:
            self.register_parameter('bias', None)
    
    def _simulate_all_gather(self, x_local, all_rank_shards=None):
        """
        Simulate all-gather operation.
        
        :param x_local: Local input shard
        :param all_rank_shards: Shards from all ranks
        :return: Gathered full tensor
        """
        if all_rank_shards is not None:
            return torch.cat(all_rank_shards, dim=-1)
        return x_local
    
    def forward(self, x_local, all_rank_shards=None):
        """
        Fused All-Gather + GEMM.
        
        :param x_local: Local input shard (batch, seq, in_per_rank)
        :param all_rank_shards: Input shards from all ranks
        :return: Local output (batch, seq, out_per_rank)
        """
        # === FUSED KERNEL START ===
        # Step 1: All-Gather
        # In fused kernel, we can overlap gathering with GEMM:
        # - Start GEMM with local shard
        # - As remote shards arrive, continue GEMM
        x_full = self._simulate_all_gather(x_local, all_rank_shards)
        
        # Step 2: GEMM
        output = F.linear(x_full, self.weight)
        
        # Step 3: Add bias
        if self.bias is not None:
            output = output + self.bias
        # === FUSED KERNEL END ===
        
        return output


# Chunked All-Gather + GEMM for overlap
class FusedChunkedAllGatherGEMM(nn.Module):
    """
    Chunked All-Gather + GEMM for compute-communication overlap.
    
    Splits input into chunks and pipelines all-gather with GEMM.
    """
    def __init__(self, in_features, out_features, world_size, rank, num_chunks=4):
        super(FusedChunkedAllGatherGEMM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.num_chunks = num_chunks
        
        self.out_per_rank = out_features // world_size
        
        # Split weight along input dimension for chunked computation
        assert in_features % num_chunks == 0
        self.chunk_size = in_features // num_chunks
        
        self.weight_chunks = nn.ParameterList([
            nn.Parameter(torch.randn(self.out_per_rank, self.chunk_size) * 0.02)
            for _ in range(num_chunks)
        ])
        self.bias = nn.Parameter(torch.zeros(self.out_per_rank))
    
    def forward(self, x_chunks, all_rank_chunks=None):
        """
        Chunked all-gather + GEMM with overlap.
        
        :param x_chunks: List of local input chunks
        :param all_rank_chunks: List of chunks from all ranks
        :return: Output
        """
        # === FUSED KERNEL WITH OVERLAP ===
        batch_shape = x_chunks[0].shape[:-1]
        output = torch.zeros(*batch_shape, self.out_per_rank, device=x_chunks[0].device)
        
        for chunk_idx in range(self.num_chunks):
            # Gather this chunk from all ranks
            if all_rank_chunks is not None:
                gathered_chunk = torch.cat(
                    [all_rank_chunks[r][chunk_idx] for r in range(self.world_size)],
                    dim=-1
                )
            else:
                gathered_chunk = x_chunks[chunk_idx]
            
            # GEMM for this chunk (accumulate into output)
            # In real impl, this overlaps with gathering next chunk
            chunk_out = F.linear(gathered_chunk, self.weight_chunks[chunk_idx])
            output = output + chunk_out
        
        return output + self.bias


# Ring All-Gather + GEMM
class FusedRingAllGatherGEMM(nn.Module):
    """
    Ring All-Gather + GEMM using ring topology.
    
    Uses ring communication pattern for bandwidth-optimal all-gather
    fused with GEMM computation.
    """
    def __init__(self, in_features, out_features, world_size, rank):
        super(FusedRingAllGatherGEMM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        
        self.out_per_rank = out_features // world_size
        self.in_per_rank = in_features // world_size
        
        # Weight partitioned by input dimension for ring overlap
        self.weight_parts = nn.ParameterList([
            nn.Parameter(torch.randn(self.out_per_rank, self.in_per_rank) * 0.02)
            for _ in range(world_size)
        ])
    
    def forward(self, x_local, ring_buffer=None):
        """
        Ring all-gather + GEMM.
        
        :param x_local: Local input shard
        :param ring_buffer: For simulating ring communication
        """
        batch_shape = x_local.shape[:-1]
        output = torch.zeros(*batch_shape, self.out_per_rank, device=x_local.device)
        
        # Ring steps
        current_shard = x_local
        current_rank = self.rank
        
        for step in range(self.world_size):
            # GEMM with current shard
            shard_output = F.linear(current_shard, self.weight_parts[current_rank])
            output = output + shard_output
            
            # Send to next, receive from previous (simulated)
            if ring_buffer is not None:
                prev_rank = (self.rank - step - 1) % self.world_size
                current_shard = ring_buffer[prev_rank]
                current_rank = prev_rank
        
        return output


# Test parameters
batch_size = 32
seq_len = 512
in_features = 4096
out_features = 4096
world_size = 8
rank = 0
in_per_rank = in_features // world_size

def get_inputs():
    x_local = torch.randn(batch_size, seq_len, in_per_rank)
    return [x_local]

def get_init_inputs():
    return [in_features, out_features, world_size, rank]

