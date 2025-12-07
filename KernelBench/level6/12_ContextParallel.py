import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Context Parallelism for Long Sequence Processing.
    
    Distributes long sequences across ranks, with each rank
    processing a portion. Uses ring communication for 
    KV sharing across context chunks.
    
    Based on: Ring Attention and Context Parallelism in DeepSpeed
    """
    def __init__(self, dim, num_heads, world_size, rank, head_dim=None):
        """
        :param dim: Model dimension
        :param num_heads: Number of attention heads
        :param world_size: Number of context parallel ranks
        :param rank: Current rank
        :param head_dim: Dimension per head
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.world_size = world_size
        self.rank = rank
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Local attention projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
    
    def ring_flash_attention(self, q_local, k_chunks, v_chunks):
        """
        Ring-based Flash Attention across context chunks.
        
        :param q_local: Local query chunk (batch, heads, seq_per_rank, head_dim)
        :param k_chunks: List of K chunks from all ranks
        :param v_chunks: List of V chunks from all ranks
        :return: Attention output for local chunk
        """
        batch, heads, seq_local, head_dim = q_local.shape
        
        # Initialize running max and sum for online softmax
        running_max = torch.full((batch, heads, seq_local), float('-inf'), 
                                  device=q_local.device)
        running_sum = torch.zeros(batch, heads, seq_local, device=q_local.device)
        output = torch.zeros_like(q_local)
        
        # Process KV chunks in ring order
        for step in range(self.world_size):
            # Get KV chunk from appropriate rank
            kv_rank = (self.rank - step) % self.world_size
            k_chunk = k_chunks[kv_rank]
            v_chunk = v_chunks[kv_rank]
            
            # Compute attention scores for this chunk
            attn_scores = torch.matmul(q_local, k_chunk.transpose(-2, -1)) * self.scale
            
            # Apply causal mask if needed (for chunks after current)
            if step > 0:
                # Full attention to all tokens in this chunk
                pass
            else:
                # Causal within local chunk
                seq_k = k_chunk.shape[2]
                causal_mask = torch.triu(
                    torch.ones(seq_local, seq_k, device=q_local.device),
                    diagonal=1
                ).bool()
                attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
            
            # Online softmax update
            chunk_max = attn_scores.max(dim=-1).values
            new_max = torch.maximum(running_max, chunk_max)
            
            # Rescale old values
            old_scale = torch.exp(running_max - new_max).unsqueeze(-1)
            new_scale = torch.exp(chunk_max - new_max).unsqueeze(-1)
            
            output = output * old_scale
            running_sum = running_sum * old_scale.squeeze(-1)
            
            # Add contribution from this chunk
            attn_probs = torch.exp(attn_scores - new_max.unsqueeze(-1))
            output = output + torch.matmul(attn_probs, v_chunk)
            running_sum = running_sum + attn_probs.sum(dim=-1)
            
            running_max = new_max
        
        # Final normalization
        output = output / running_sum.unsqueeze(-1).clamp(min=1e-9)
        
        return output
    
    def forward(self, x_chunks):
        """
        Context-parallel attention forward.
        
        :param x_chunks: List of input chunks, one per rank
        :return: List of output chunks
        """
        batch_size = x_chunks[0].shape[0]
        seq_per_rank = x_chunks[0].shape[1]
        
        # Project all chunks to Q, K, V
        q_chunks = []
        k_chunks = []
        v_chunks = []
        
        for chunk in x_chunks:
            q = self.q_proj(chunk).view(batch_size, seq_per_rank, 
                                        self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(chunk).view(batch_size, seq_per_rank, 
                                        self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(chunk).view(batch_size, seq_per_rank, 
                                        self.num_heads, self.head_dim).transpose(1, 2)
            q_chunks.append(q)
            k_chunks.append(k)
            v_chunks.append(v)
        
        # Each rank computes attention for its local Q using all K, V
        outputs = []
        for r in range(self.world_size):
            out = self.ring_flash_attention(q_chunks[r], k_chunks, v_chunks)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_per_rank, self.dim)
            out = self.out_proj(out)
            outputs.append(out)
        
        return outputs


# Context Parallel with KV Cache
class ContextParallelKVCache(nn.Module):
    """
    Context Parallelism with distributed KV Cache.
    
    Each rank maintains a portion of the KV cache for incremental decoding.
    """
    def __init__(self, dim, num_heads, world_size, rank, max_seq_per_rank):
        super(ContextParallelKVCache, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.world_size = world_size
        self.rank = rank
        self.head_dim = dim // num_heads
        self.max_seq_per_rank = max_seq_per_rank
        
        # Local KV cache
        self.register_buffer('k_cache', 
            torch.zeros(1, num_heads, max_seq_per_rank, self.head_dim))
        self.register_buffer('v_cache', 
            torch.zeros(1, num_heads, max_seq_per_rank, self.head_dim))
        self.register_buffer('cache_len', torch.tensor(0))
    
    def update_cache(self, new_k, new_v):
        """Update local KV cache."""
        seq_len = new_k.shape[2]
        start = self.cache_len.item()
        end = start + seq_len
        
        if end <= self.max_seq_per_rank:
            self.k_cache[:, :, start:end] = new_k
            self.v_cache[:, :, start:end] = new_v
            self.cache_len += seq_len
    
    def gather_kv_caches(self, all_k_caches, all_v_caches):
        """Gather KV caches from all ranks for full attention."""
        full_k = torch.cat(all_k_caches, dim=2)
        full_v = torch.cat(all_v_caches, dim=2)
        return full_k, full_v


# Test parameters
batch_size = 4
total_seq_len = 8192
dim = 2048
num_heads = 16
world_size = 8
rank = 0
seq_per_rank = total_seq_len // world_size

def get_inputs():
    # Input chunks for each rank
    x_chunks = [torch.randn(batch_size, seq_per_rank, dim) for _ in range(world_size)]
    return [x_chunks]

def get_init_inputs():
    return [dim, num_heads, world_size, rank]

