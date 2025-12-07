import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Distributed Attention Mechanisms.
    
    Collection of distributed attention patterns for large-scale
    inference including Ulysses, Ring, and Striped attention.
    
    Based on: Ulysses, Ring Attention, and DeepSpeed Ulysses
    """
    def __init__(self, dim, num_heads, world_size, rank, 
                 attention_type='ulysses', head_dim=None):
        """
        :param dim: Model dimension
        :param num_heads: Total attention heads
        :param world_size: Number of ranks
        :param rank: Current rank
        :param attention_type: 'ulysses', 'ring', or 'striped'
        :param head_dim: Dimension per head
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.world_size = world_size
        self.rank = rank
        self.attention_type = attention_type
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
    
    def ulysses_attention(self, x_chunks):
        """
        Ulysses-style sequence parallel attention.
        
        Each rank has a portion of sequence. All-to-all redistributes
        to head parallelism for attention, then back to sequence.
        
        :param x_chunks: List of input chunks per rank
        :return: List of output chunks per rank
        """
        batch = x_chunks[0].shape[0]
        seq_per_rank = x_chunks[0].shape[1]
        
        # Project to Q, K, V
        q_chunks = [self.q_proj(chunk) for chunk in x_chunks]
        k_chunks = [self.k_proj(chunk) for chunk in x_chunks]
        v_chunks = [self.v_proj(chunk) for chunk in x_chunks]
        
        # All-to-All: sequence parallel -> head parallel
        # Gather full sequence, split by heads
        q_full = torch.cat(q_chunks, dim=1)
        k_full = torch.cat(k_chunks, dim=1)
        v_full = torch.cat(v_chunks, dim=1)
        
        full_seq = q_full.shape[1]
        
        # Reshape for attention
        q = q_full.view(batch, full_seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_full.view(batch, full_seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_full.view(batch, full_seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Each rank processes subset of heads
        heads_per_rank = self.num_heads // self.world_size
        start_head = self.rank * heads_per_rank
        end_head = start_head + heads_per_rank
        
        q_local = q[:, start_head:end_head]
        k_local = k[:, start_head:end_head]
        v_local = v[:, start_head:end_head]
        
        # Attention on local heads
        attn = F.softmax(torch.matmul(q_local, k_local.transpose(-2, -1)) * self.scale, dim=-1)
        out_local = torch.matmul(attn, v_local)
        
        # All-gather heads, split sequence back
        # Simulate all-gather
        out_full = torch.zeros(batch, self.num_heads, full_seq, self.head_dim, 
                               device=x_chunks[0].device)
        out_full[:, start_head:end_head] = out_local
        
        # Reshape and project
        out_full = out_full.transpose(1, 2).contiguous().view(batch, full_seq, self.dim)
        out_full = self.out_proj(out_full)
        
        # Split back to sequence chunks
        return torch.split(out_full, seq_per_rank, dim=1)
    
    def ring_attention(self, x_chunks):
        """
        Ring attention for distributed long sequences.
        
        Each rank computes attention with local Q against all K, V
        using ring communication pattern.
        """
        batch = x_chunks[0].shape[0]
        seq_per_rank = x_chunks[0].shape[1]
        
        # Project to Q, K, V
        q_chunks = [self.q_proj(chunk).view(batch, seq_per_rank, self.num_heads, 
                    self.head_dim).transpose(1, 2) for chunk in x_chunks]
        k_chunks = [self.k_proj(chunk).view(batch, seq_per_rank, self.num_heads,
                    self.head_dim).transpose(1, 2) for chunk in x_chunks]
        v_chunks = [self.v_proj(chunk).view(batch, seq_per_rank, self.num_heads,
                    self.head_dim).transpose(1, 2) for chunk in x_chunks]
        
        outputs = []
        
        for local_rank in range(self.world_size):
            q_local = q_chunks[local_rank]
            
            # Initialize running stats for online softmax
            running_max = torch.full((batch, self.num_heads, seq_per_rank), 
                                    float('-inf'), device=q_local.device)
            running_sum = torch.zeros(batch, self.num_heads, seq_per_rank,
                                     device=q_local.device)
            output = torch.zeros_like(q_local)
            
            # Ring pass through all KV chunks
            for step in range(self.world_size):
                kv_rank = (local_rank + step) % self.world_size
                k = k_chunks[kv_rank]
                v = v_chunks[kv_rank]
                
                # Compute attention scores
                scores = torch.matmul(q_local, k.transpose(-2, -1)) * self.scale
                
                # Online softmax update
                chunk_max = scores.max(dim=-1).values
                new_max = torch.maximum(running_max, chunk_max)
                
                old_scale = torch.exp(running_max - new_max).unsqueeze(-1)
                new_scale = torch.exp(chunk_max - new_max).unsqueeze(-1)
                
                output = output * old_scale
                running_sum = running_sum * old_scale.squeeze(-1)
                
                exp_scores = torch.exp(scores - new_max.unsqueeze(-1))
                output = output + torch.matmul(exp_scores, v)
                running_sum = running_sum + exp_scores.sum(dim=-1)
                
                running_max = new_max
            
            # Normalize
            output = output / running_sum.unsqueeze(-1).clamp(min=1e-9)
            output = output.transpose(1, 2).contiguous().view(batch, seq_per_rank, self.dim)
            output = self.out_proj(output)
            
            outputs.append(output)
        
        return outputs
    
    def striped_attention(self, x_chunks):
        """
        Striped attention pattern for load-balanced distribution.
        
        Interleaves sequence positions across ranks for better balance.
        """
        # Interleave: rank i gets positions i, i+world_size, i+2*world_size, ...
        batch = x_chunks[0].shape[0]
        seq_per_rank = x_chunks[0].shape[1]
        full_seq = seq_per_rank * self.world_size
        
        # Gather and stripe
        x_full = torch.cat(x_chunks, dim=1)
        
        # Stripe pattern
        striped_indices = []
        for r in range(self.world_size):
            striped_indices.extend(range(r, full_seq, self.world_size))
        
        x_striped = x_full[:, striped_indices]
        
        # Standard attention on striped sequence
        q = self.q_proj(x_striped).view(batch, full_seq, self.num_heads, 
                                        self.head_dim).transpose(1, 2)
        k = self.k_proj(x_striped).view(batch, full_seq, self.num_heads,
                                        self.head_dim).transpose(1, 2)
        v = self.v_proj(x_striped).view(batch, full_seq, self.num_heads,
                                        self.head_dim).transpose(1, 2)
        
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, full_seq, self.dim)
        out = self.out_proj(out)
        
        # Unstripe and split
        unstriped = torch.zeros_like(out)
        for i, idx in enumerate(striped_indices):
            unstriped[:, idx] = out[:, i]
        
        return torch.split(unstriped, seq_per_rank, dim=1)
    
    def forward(self, x_chunks):
        """
        Apply distributed attention.
        
        :param x_chunks: List of input chunks per rank
        :return: List of output chunks per rank
        """
        if self.attention_type == 'ulysses':
            return self.ulysses_attention(x_chunks)
        elif self.attention_type == 'ring':
            return self.ring_attention(x_chunks)
        elif self.attention_type == 'striped':
            return self.striped_attention(x_chunks)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")


# Test parameters
batch_size = 4
total_seq_len = 8192
dim = 2048
num_heads = 16
world_size = 8
rank = 0
seq_per_rank = total_seq_len // world_size

def get_inputs():
    x_chunks = [torch.randn(batch_size, seq_per_rank, dim) for _ in range(world_size)]
    return [x_chunks]

def get_init_inputs():
    return [dim, num_heads, world_size, rank, 'ulysses']

