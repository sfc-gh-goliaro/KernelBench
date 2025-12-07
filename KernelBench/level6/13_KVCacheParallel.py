import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Distributed KV Cache for Parallel Inference.
    
    Distributes KV cache across ranks for memory efficiency
    during long-context inference. Supports sharded and 
    replicated cache strategies.
    
    Based on: vLLM, TensorRT-LLM, and DeepSpeed inference
    """
    def __init__(self, num_layers, num_heads, head_dim, world_size, rank,
                 max_seq_len, cache_strategy='sharded'):
        """
        :param num_layers: Number of transformer layers
        :param num_heads: Number of attention heads
        :param head_dim: Dimension per head
        :param world_size: Number of ranks
        :param rank: Current rank
        :param max_seq_len: Maximum sequence length
        :param cache_strategy: 'sharded' or 'replicated'
        """
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.world_size = world_size
        self.rank = rank
        self.max_seq_len = max_seq_len
        self.cache_strategy = cache_strategy
        
        if cache_strategy == 'sharded':
            # Each rank stores a portion of heads
            assert num_heads % world_size == 0
            self.heads_per_rank = num_heads // world_size
            cache_heads = self.heads_per_rank
        else:
            # Full replication
            cache_heads = num_heads
            self.heads_per_rank = num_heads
        
        # Allocate KV cache per layer
        self.k_caches = nn.ParameterList([
            nn.Parameter(torch.zeros(1, cache_heads, max_seq_len, head_dim), 
                        requires_grad=False)
            for _ in range(num_layers)
        ])
        self.v_caches = nn.ParameterList([
            nn.Parameter(torch.zeros(1, cache_heads, max_seq_len, head_dim),
                        requires_grad=False)
            for _ in range(num_layers)
        ])
        
        # Track sequence position per batch item
        self.register_buffer('seq_positions', torch.zeros(1, dtype=torch.long))
    
    def update(self, layer_idx, new_k, new_v, positions=None):
        """
        Update KV cache for a layer.
        
        :param layer_idx: Layer index
        :param new_k: New keys (batch, heads_per_rank, seq, head_dim)
        :param new_v: New values (batch, heads_per_rank, seq, head_dim)
        :param positions: Sequence positions to update
        :return: Updated KV tensors
        """
        batch_size = new_k.shape[0]
        seq_len = new_k.shape[2]
        
        if positions is None:
            # Append to end
            start = self.seq_positions[0].item()
            positions = torch.arange(start, start + seq_len, device=new_k.device)
            self.seq_positions[0] = start + seq_len
        
        # Expand cache if needed
        if batch_size > self.k_caches[layer_idx].shape[0]:
            self.k_caches[layer_idx] = nn.Parameter(
                self.k_caches[layer_idx].expand(batch_size, -1, -1, -1).clone(),
                requires_grad=False
            )
            self.v_caches[layer_idx] = nn.Parameter(
                self.v_caches[layer_idx].expand(batch_size, -1, -1, -1).clone(),
                requires_grad=False
            )
        
        # Update cache at specified positions
        for i, pos in enumerate(positions):
            self.k_caches[layer_idx].data[:batch_size, :, pos] = new_k[:, :, i]
            self.v_caches[layer_idx].data[:batch_size, :, pos] = new_v[:, :, i]
        
        return self.k_caches[layer_idx][:batch_size], self.v_caches[layer_idx][:batch_size]
    
    def gather_kv(self, layer_idx, seq_len, all_rank_caches=None):
        """
        Gather KV cache from all ranks (for sharded strategy).
        
        :param layer_idx: Layer index
        :param seq_len: Sequence length to gather
        :param all_rank_caches: Caches from all ranks (simulation)
        :return: Full KV cache
        """
        if self.cache_strategy == 'replicated':
            return (self.k_caches[layer_idx][:, :, :seq_len],
                    self.v_caches[layer_idx][:, :, :seq_len])
        
        if all_rank_caches is not None:
            # All-gather across head dimension
            all_k = torch.cat([c[0][:, :, :seq_len] for c in all_rank_caches], dim=1)
            all_v = torch.cat([c[1][:, :, :seq_len] for c in all_rank_caches], dim=1)
            return all_k, all_v
        
        return (self.k_caches[layer_idx][:, :, :seq_len],
                self.v_caches[layer_idx][:, :, :seq_len])
    
    def forward(self, layer_idx, q, seq_len, all_rank_caches=None):
        """
        Attend to cached KV.
        
        :param layer_idx: Layer index
        :param q: Query tensor (batch, heads, 1, head_dim)
        :param seq_len: Current sequence length
        :param all_rank_caches: For simulation, caches from all ranks
        :return: Attention output
        """
        # Gather full KV
        k, v = self.gather_kv(layer_idx, seq_len, all_rank_caches)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v)
        
        return out


# Paged KV Cache (vLLM-style)
class PagedKVCacheParallel(nn.Module):
    """
    Paged KV Cache with distributed block management.
    
    Organizes cache into fixed-size blocks for efficient memory
    management across ranks.
    """
    def __init__(self, num_layers, num_heads, head_dim, block_size, 
                 num_blocks_per_rank, world_size, rank):
        super(PagedKVCacheParallel, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks_per_rank = num_blocks_per_rank
        self.world_size = world_size
        self.rank = rank
        
        # Block pool per layer
        self.k_blocks = nn.ParameterList([
            nn.Parameter(
                torch.zeros(num_blocks_per_rank, num_heads, block_size, head_dim),
                requires_grad=False
            )
            for _ in range(num_layers)
        ])
        self.v_blocks = nn.ParameterList([
            nn.Parameter(
                torch.zeros(num_blocks_per_rank, num_heads, block_size, head_dim),
                requires_grad=False
            )
            for _ in range(num_layers)
        ])
        
        # Block allocation tracking
        self.register_buffer('free_blocks', 
            torch.ones(num_blocks_per_rank, dtype=torch.bool))
    
    def allocate_block(self):
        """Allocate a free block."""
        free_idx = self.free_blocks.nonzero(as_tuple=True)[0]
        if len(free_idx) == 0:
            return None
        
        block_idx = free_idx[0].item()
        self.free_blocks[block_idx] = False
        return block_idx
    
    def free_block(self, block_idx):
        """Free a block."""
        self.free_blocks[block_idx] = True
    
    def forward(self, layer_idx, block_indices, slot_indices, new_k, new_v):
        """
        Update paged cache.
        
        :param layer_idx: Layer index
        :param block_indices: Block indices for each token
        :param slot_indices: Slot within block for each token
        :param new_k: New keys
        :param new_v: New values
        """
        for i, (block_idx, slot_idx) in enumerate(zip(block_indices, slot_indices)):
            self.k_blocks[layer_idx].data[block_idx, :, slot_idx] = new_k[i]
            self.v_blocks[layer_idx].data[block_idx, :, slot_idx] = new_v[i]


# Test parameters
num_layers = 32
num_heads = 32
head_dim = 128
world_size = 8
rank = 0
max_seq_len = 8192

def get_inputs():
    # Query for attention
    batch_size = 4
    q = torch.randn(batch_size, num_heads // world_size, 1, head_dim)
    seq_len = 1024
    layer_idx = 0
    return [layer_idx, q, seq_len]

def get_init_inputs():
    return [num_layers, num_heads, head_dim, world_size, rank, max_seq_len]

