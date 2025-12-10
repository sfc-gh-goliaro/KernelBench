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
    Sequence Parallelism for Transformer Layers.
    
    Distributes sequence dimension across ranks for LayerNorm and
    Dropout operations, reducing memory per rank.
    
    Based on: Megatron-LM sequence parallelism
    """
    def __init__(self, dim, world_size, rank):
        """
        :param dim: Model dimension
        :param world_size: Number of sequence parallel ranks
        :param rank: Current rank
        """
        super(Model, self).__init__()
        self.dim = dim
        self.world_size = world_size
        self.rank = rank
        
        # LayerNorm operates on local sequence shard
        self.layer_norm = nn.LayerNorm(dim)
        
        # Dropout also operates on local shard
        self.dropout = nn.Dropout(0.1)
    
    def scatter_to_sequence_parallel(self, x):
        """
        Scatter tensor along sequence dimension to all ranks.
        
        :param x: Full tensor (batch, seq, dim)
        :return: Local shard (batch, seq/world_size, dim)
        """
        batch, seq, dim = x.shape
        assert seq % self.world_size == 0
        seq_per_rank = seq // self.world_size
        
        return x[:, self.rank * seq_per_rank:(self.rank + 1) * seq_per_rank]
    
    def gather_from_sequence_parallel(self, x_shards):
        """
        Gather tensor shards from all ranks.
        
        :param x_shards: List of local shards from each rank
        :return: Gathered tensor (batch, seq, dim)
        """
        return torch.cat(x_shards, dim=1)
    
    def forward(self, x, is_scattered=True, all_shards=None):
        """
        Apply sequence-parallel LayerNorm and Dropout.
        
        :param x: Input tensor (local shard if is_scattered)
        :param is_scattered: Whether input is already scattered
        :param all_shards: For simulation, list of all shards
        :return: Normalized and dropout-applied tensor
        """
        if not is_scattered:
            x = self.scatter_to_sequence_parallel(x)
        
        # Apply LayerNorm on local shard
        x = self.layer_norm(x)
        
        # Apply Dropout on local shard
        x = self.dropout(x)
        
        if all_shards is not None:
            all_shards.append(x)
            if len(all_shards) == self.world_size:
                return self.gather_from_sequence_parallel(all_shards)
        
        return x


# Sequence Parallel with Ring Communication
class SequenceParallelRing(nn.Module):
    """
    Sequence Parallelism with Ring All-Gather.
    
    Uses ring topology for efficient all-gather of sequence shards.
    """
    def __init__(self, dim, world_size, rank):
        super(SequenceParallelRing, self).__init__()
        self.dim = dim
        self.world_size = world_size
        self.rank = rank
        self.layer_norm = nn.LayerNorm(dim)
    
    def ring_all_gather(self, local_shard, all_shards):
        """
        Ring-based all-gather for sequence shards.
        
        :param local_shard: This rank's shard
        :param all_shards: List for collecting all shards (simulation)
        :return: Gathered tensor when all shards collected
        """
        # In practice, this would be ring-based point-to-point
        all_shards[self.rank] = local_shard
        
        # Simulate ring steps
        for step in range(1, self.world_size):
            recv_from = (self.rank - step) % self.world_size
            # In real impl: receive from recv_from rank
        
        if all(s is not None for s in all_shards):
            return torch.cat(all_shards, dim=1)
        return None
    
    def forward(self, x):
        """
        Sequence-parallel forward with ring communication.
        
        :param x: Local sequence shard
        :return: LayerNorm applied shard
        """
        return self.layer_norm(x)


# Ulysses-style Sequence Parallelism (for attention)
class UlyssesSequenceParallel(nn.Module):
    """
    Ulysses Sequence Parallelism for Attention.
    
    Distributes the sequence across ranks for attention computation
    using all-to-all communication for Q, K, V.
    """
    def __init__(self, dim, num_heads, world_size, rank):
        super(UlyssesSequenceParallel, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.world_size = world_size
        self.rank = rank
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x_shards):
        """
        Ulysses attention with sequence parallelism.
        
        :param x_shards: List of sequence shards from each rank
        :return: Output shards for each rank
        """
        # Each rank has a portion of the sequence
        # Project Q, K, V locally
        q_shards = [self.q_proj(shard) for shard in x_shards]
        k_shards = [self.k_proj(shard) for shard in x_shards]
        v_shards = [self.v_proj(shard) for shard in x_shards]
        
        # All-to-all: redistribute by heads instead of sequence
        # After all-to-all, each rank has all sequence positions for subset of heads
        batch = x_shards[0].shape[0]
        seq_per_rank = x_shards[0].shape[1]
        full_seq = seq_per_rank * self.world_size
        
        # Gather full Q, K, V (simulation of all-to-all)
        q_full = torch.cat(q_shards, dim=1)
        k_full = torch.cat(k_shards, dim=1)
        v_full = torch.cat(v_shards, dim=1)
        
        # Compute attention on full sequence
        q = q_full.view(batch, full_seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_full.view(batch, full_seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_full.view(batch, full_seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch, full_seq, self.dim)
        out = self.out_proj(out)
        
        # Split back to shards
        return torch.split(out, seq_per_rank, dim=1)


# Test parameters
batch_size = 8
seq_len = 2048
dim = 4096
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
    # Local sequence shard
    seq_per_rank = seq_len // world_size
    x = torch.randn(batch_size, seq_per_rank, dim)
    return [x]

def get_init_inputs():
    return [dim, world_size, rank]

