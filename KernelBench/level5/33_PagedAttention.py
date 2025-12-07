import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Paged Attention for efficient KV cache management.
    
    Organizes KV cache into fixed-size pages (blocks) for efficient
    memory management in batched inference scenarios.
    
    Based on: "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention"
    """
    def __init__(self, num_heads, head_dim, block_size=16, num_blocks=1024):
        """
        :param num_heads: Number of attention heads
        :param head_dim: Dimension per head
        :param block_size: Number of tokens per block
        :param num_blocks: Total number of blocks in the pool
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.scale = head_dim ** -0.5
        
        # Pre-allocated block pool for K and V
        # Shape: (num_blocks, num_heads, block_size, head_dim)
        self.register_buffer('k_blocks', 
            torch.zeros(num_blocks, num_heads, block_size, head_dim))
        self.register_buffer('v_blocks', 
            torch.zeros(num_blocks, num_heads, block_size, head_dim))
        
        # Block allocation tracking
        self.register_buffer('block_free_list', 
            torch.ones(num_blocks, dtype=torch.bool))
        
    def allocate_blocks(self, num_needed):
        """
        Allocate blocks from the pool.
        
        :param num_needed: Number of blocks to allocate
        :return: List of allocated block indices
        """
        free_indices = self.block_free_list.nonzero(as_tuple=True)[0]
        if len(free_indices) < num_needed:
            raise RuntimeError("Not enough free blocks")
        
        allocated = free_indices[:num_needed].tolist()
        self.block_free_list[allocated] = False
        return allocated
    
    def free_blocks(self, block_indices):
        """
        Free blocks back to the pool.
        
        :param block_indices: List of block indices to free
        """
        self.block_free_list[block_indices] = True
        # Zero out freed blocks
        for idx in block_indices:
            self.k_blocks[idx].zero_()
            self.v_blocks[idx].zero_()
    
    def write_kv(self, k, v, block_table, slot_offset):
        """
        Write K, V values to paged cache.
        
        :param k: Key tensor (batch, num_heads, 1, head_dim)
        :param v: Value tensor (batch, num_heads, 1, head_dim)
        :param block_table: Block indices for each sequence (batch, max_blocks)
        :param slot_offset: Current slot offset within sequence
        """
        batch_size = k.shape[0]
        block_idx = slot_offset // self.block_size
        slot_idx = slot_offset % self.block_size
        
        for b in range(batch_size):
            if block_idx < block_table.shape[1]:
                block = block_table[b, block_idx].item()
                self.k_blocks[block, :, slot_idx] = k[b, :, 0]
                self.v_blocks[block, :, slot_idx] = v[b, :, 0]
    
    def forward(self, q, block_table, context_lens):
        """
        Compute paged attention.
        
        :param q: Query tensor (batch, num_heads, 1, head_dim)
        :param block_table: Block table mapping (batch, max_num_blocks)
        :param context_lens: Context length for each sequence (batch,)
        :return: Attention output (batch, num_heads, 1, head_dim)
        """
        batch_size = q.shape[0]
        device = q.device
        
        outputs = []
        
        for b in range(batch_size):
            ctx_len = context_lens[b].item()
            num_full_blocks = ctx_len // self.block_size
            last_block_size = ctx_len % self.block_size
            
            if ctx_len == 0:
                outputs.append(torch.zeros(1, self.num_heads, 1, self.head_dim, device=device))
                continue
            
            # Gather K, V from blocks
            k_gathered = []
            v_gathered = []
            
            for block_idx in range(num_full_blocks):
                block = block_table[b, block_idx].item()
                k_gathered.append(self.k_blocks[block])
                v_gathered.append(self.v_blocks[block])
            
            # Handle partial last block
            if last_block_size > 0 and num_full_blocks < block_table.shape[1]:
                block = block_table[b, num_full_blocks].item()
                k_gathered.append(self.k_blocks[block, :, :last_block_size])
                v_gathered.append(self.v_blocks[block, :, :last_block_size])
            
            if not k_gathered:
                outputs.append(torch.zeros(1, self.num_heads, 1, self.head_dim, device=device))
                continue
            
            # Concatenate gathered KV
            k_seq = torch.cat(k_gathered, dim=1).unsqueeze(0)  # (1, heads, ctx_len, head_dim)
            v_seq = torch.cat(v_gathered, dim=1).unsqueeze(0)
            
            # Compute attention for this sequence
            q_b = q[b:b+1]  # (1, heads, 1, head_dim)
            
            attn_scores = torch.matmul(q_b, k_seq.transpose(-2, -1)) * self.scale
            attn_probs = F.softmax(attn_scores, dim=-1)
            out_b = torch.matmul(attn_probs, v_seq)
            
            outputs.append(out_b)
        
        return torch.cat(outputs, dim=0)


# Test parameters
batch_size = 8
num_heads = 32
head_dim = 128
block_size = 16
num_blocks = 256
max_context = 512

def get_inputs():
    q = torch.randn(batch_size, num_heads, 1, head_dim)
    # Block table: each sequence uses some blocks
    block_table = torch.randint(0, num_blocks // 2, (batch_size, max_context // block_size))
    context_lens = torch.randint(16, max_context, (batch_size,))
    return [q, block_table, context_lens]

def get_init_inputs():
    return [num_heads, head_dim, block_size, num_blocks]

