import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused KV Cache Update + Attention.
    
    Combines KV cache management with attention computation:
    1. Update KV cache with new K, V
    2. Compute attention with updated cache
    3. Handle paged memory management
    
    Critical for efficient LLM inference serving.
    
    Reference: vLLM, TensorRT-LLM, FlashInfer
    """
    def __init__(self, num_heads, head_dim, max_seq_len, num_kv_heads=None):
        """
        :param num_heads: Number of query heads
        :param head_dim: Head dimension
        :param max_seq_len: Maximum sequence length
        :param num_kv_heads: Number of KV heads (for GQA)
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        self.max_seq_len = max_seq_len
        self.scale = head_dim ** -0.5
        
        # Pre-allocated KV cache
        self.register_buffer('k_cache', 
            torch.zeros(1, self.num_kv_heads, max_seq_len, head_dim))
        self.register_buffer('v_cache', 
            torch.zeros(1, self.num_kv_heads, max_seq_len, head_dim))
        self.register_buffer('cache_len', torch.tensor(0))
    
    def forward(self, q, k_new, v_new, position=None):
        """
        Fused KV cache update + attention.
        
        :param q: Query (batch, num_heads, seq_q, head_dim)
        :param k_new: New keys (batch, num_kv_heads, seq_new, head_dim)
        :param v_new: New values (batch, num_kv_heads, seq_new, head_dim)
        :param position: Position index for cache update
        :return: Attention output
        """
        batch_size = q.shape[0]
        seq_new = k_new.shape[2]
        
        # Expand cache for batch
        if self.k_cache.shape[0] < batch_size:
            self.k_cache = self.k_cache.expand(batch_size, -1, -1, -1).clone()
            self.v_cache = self.v_cache.expand(batch_size, -1, -1, -1).clone()
        
        # === FUSED KERNEL START ===
        # Step 1: Cache update
        if position is None:
            position = self.cache_len.item()
        
        # Update cache
        end_pos = position + seq_new
        self.k_cache[:batch_size, :, position:end_pos] = k_new
        self.v_cache[:batch_size, :, position:end_pos] = v_new
        self.cache_len = torch.tensor(end_pos)
        
        # Step 2: Get relevant KV from cache
        k = self.k_cache[:batch_size, :, :end_pos]
        v = self.v_cache[:batch_size, :, :end_pos]
        
        # Step 3: GQA expansion if needed
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
                batch_size, self.num_heads, -1, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
                batch_size, self.num_heads, -1, self.head_dim)
        
        # Step 4: Attention computation
        seq_q = q.shape[2]
        seq_k = k.shape[2]
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask for the new query positions
        if seq_q > 1:
            causal_mask = torch.triu(
                torch.ones(seq_q, seq_k, device=q.device, dtype=torch.bool),
                diagonal=seq_k - seq_q + 1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)
        # === FUSED KERNEL END ===
        
        return output
    
    def reset_cache(self):
        """Reset KV cache."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_len = torch.tensor(0)


# Paged KV Cache Update
class FusedPagedKVCacheUpdate(nn.Module):
    """
    Fused Paged KV Cache Update + Attention.
    
    Uses paged memory management for efficient memory utilization.
    """
    def __init__(self, num_heads, head_dim, block_size=16, num_blocks=1024,
                 num_kv_heads=None):
        super(FusedPagedKVCacheUpdate, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.scale = head_dim ** -0.5
        
        # Block pool
        self.register_buffer('k_blocks',
            torch.zeros(num_blocks, self.num_kv_heads, block_size, head_dim))
        self.register_buffer('v_blocks',
            torch.zeros(num_blocks, self.num_kv_heads, block_size, head_dim))
        
        # Block allocation (simulated)
        self.register_buffer('free_blocks', torch.ones(num_blocks, dtype=torch.bool))
    
    def allocate_block(self):
        """Allocate a free block."""
        free_idx = self.free_blocks.nonzero(as_tuple=True)[0]
        if len(free_idx) == 0:
            return None
        block_idx = free_idx[0].item()
        self.free_blocks[block_idx] = False
        return block_idx
    
    def forward(self, q, k_new, v_new, block_table, slot_mapping):
        """
        Fused paged attention with cache update.
        
        :param q: Query (batch, num_heads, seq_q, head_dim)
        :param k_new: New keys
        :param v_new: New values
        :param block_table: Block indices per sequence (batch, max_blocks)
        :param slot_mapping: Slot indices for new tokens
        :return: Attention output
        """
        batch_size = q.shape[0]
        
        # === FUSED KERNEL START ===
        # Update paged cache
        for i, slot in enumerate(slot_mapping):
            if slot >= 0:
                block_idx = slot // self.block_size
                block_offset = slot % self.block_size
                actual_block = block_table[i // q.shape[2], block_idx]
                
                token_idx = i % k_new.shape[2]
                batch_idx = i // k_new.shape[2]
                
                self.k_blocks[actual_block, :, block_offset] = k_new[batch_idx, :, token_idx]
                self.v_blocks[actual_block, :, block_offset] = v_new[batch_idx, :, token_idx]
        
        # Gather KV from blocks
        # (Simplified - full impl would gather based on block_table)
        max_seq = block_table.shape[1] * self.block_size
        k_gathered = torch.zeros(batch_size, self.num_kv_heads, max_seq, self.head_dim,
                                device=q.device)
        v_gathered = torch.zeros_like(k_gathered)
        
        for b in range(batch_size):
            for block_pos in range(block_table.shape[1]):
                block_idx = block_table[b, block_pos]
                if block_idx >= 0:
                    start = block_pos * self.block_size
                    end = start + self.block_size
                    k_gathered[b, :, start:end] = self.k_blocks[block_idx]
                    v_gathered[b, :, start:end] = self.v_blocks[block_idx]
        
        # Attention
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k_gathered = k_gathered.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
                batch_size, self.num_heads, -1, self.head_dim)
            v_gathered = v_gathered.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
                batch_size, self.num_heads, -1, self.head_dim)
        
        attn = F.softmax(torch.matmul(q, k_gathered.transpose(-2, -1)) * self.scale, dim=-1)
        output = torch.matmul(attn, v_gathered)
        # === FUSED KERNEL END ===
        
        return output


# Test parameters
batch_size = 8
num_heads = 32
num_kv_heads = 8
head_dim = 128
max_seq_len = 4096
seq_new = 1  # Decode one token

def get_inputs():
    q = torch.randn(batch_size, num_heads, seq_new, head_dim)
    k_new = torch.randn(batch_size, num_kv_heads, seq_new, head_dim)
    v_new = torch.randn(batch_size, num_kv_heads, seq_new, head_dim)
    return [q, k_new, v_new]

def get_init_inputs():
    return [num_heads, head_dim, max_seq_len, num_kv_heads]

