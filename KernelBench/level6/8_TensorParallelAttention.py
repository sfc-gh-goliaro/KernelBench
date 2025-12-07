import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Tensor Parallel Multi-Head Attention.
    
    Distributes attention heads across tensor parallel ranks.
    Each rank computes a subset of attention heads.
    
    Based on: Megatron-LM tensor parallelism
    """
    def __init__(self, dim, num_heads, world_size, rank, head_dim=None):
        """
        :param dim: Model dimension
        :param num_heads: Total number of attention heads
        :param world_size: Number of tensor parallel ranks
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
        
        # Each rank handles a subset of heads
        assert num_heads % world_size == 0
        self.num_heads_per_rank = num_heads // world_size
        self.hidden_per_rank = self.num_heads_per_rank * self.head_dim
        
        # Column-parallel Q, K, V projections
        self.q_proj = nn.Linear(dim, self.hidden_per_rank, bias=False)
        self.k_proj = nn.Linear(dim, self.hidden_per_rank, bias=False)
        self.v_proj = nn.Linear(dim, self.hidden_per_rank, bias=False)
        
        # Row-parallel output projection
        self.out_proj = nn.Linear(self.hidden_per_rank, dim, bias=False)
    
    def forward(self, x, attention_mask=None, all_rank_outputs=None):
        """
        Tensor-parallel attention forward.
        
        :param x: Input tensor (batch, seq, dim)
        :param attention_mask: Optional attention mask
        :param all_rank_outputs: For simulation, list for all-reduce
        :return: Attention output (batch, seq, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V (column parallel - each rank has subset)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads_per_rank, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads_per_rank, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads_per_rank, self.head_dim).transpose(1, 2)
        
        # Attention computation (local to this rank's heads)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)
        
        # Reshape
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_per_rank)
        
        # Output projection (row parallel - needs all-reduce)
        output = self.out_proj(attn_out)
        
        if all_rank_outputs is not None:
            all_rank_outputs.append(output)
            if len(all_rank_outputs) == self.world_size:
                # All-reduce: sum partial outputs
                return sum(all_rank_outputs)
        
        return output


# Tensor Parallel with GQA
class TensorParallelGQA(nn.Module):
    """
    Tensor Parallel Grouped Query Attention.
    
    Handles GQA where KV heads are shared across query head groups.
    Distributes query heads across ranks while replicating KV heads.
    """
    def __init__(self, dim, num_heads, num_kv_heads, world_size, rank, head_dim=None):
        super(TensorParallelGQA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.world_size = world_size
        self.rank = rank
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert num_heads % world_size == 0
        assert num_kv_heads % world_size == 0 or num_kv_heads < world_size
        
        self.num_heads_per_rank = num_heads // world_size
        self.num_kv_heads_per_rank = max(1, num_kv_heads // world_size)
        self.num_groups = self.num_heads_per_rank // self.num_kv_heads_per_rank
        
        self.q_proj = nn.Linear(dim, self.num_heads_per_rank * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads_per_rank * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads_per_rank * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads_per_rank * self.head_dim, dim, bias=False)
    
    def forward(self, x, all_rank_outputs=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads_per_rank, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads_per_rank, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads_per_rank, self.head_dim).transpose(1, 2)
        
        # Expand KV for GQA
        k = k.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(
            batch_size, self.num_heads_per_rank, seq_len, self.head_dim)
        v = v.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(
            batch_size, self.num_heads_per_rank, seq_len, self.head_dim)
        
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.out_proj(out)
        
        if all_rank_outputs is not None:
            all_rank_outputs.append(output)
            if len(all_rank_outputs) == self.world_size:
                return sum(all_rank_outputs)
        
        return output


# Test parameters
batch_size = 8
seq_len = 1024
dim = 4096
num_heads = 32
world_size = 8
rank = 0

def get_inputs():
    return [torch.randn(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, num_heads, world_size, rank]

