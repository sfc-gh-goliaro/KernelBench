import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DETRSelfAttention(nn.Module):
    """DETR decoder self-attention over object queries."""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, query_pos: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = k = x
        if query_pos is not None:
            q = q + query_pos
            k = k + query_pos
        
        q = self.q_proj(q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out)


class DETRCrossAttention(nn.Module):
    """DETR decoder cross-attention to encoder memory."""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, 
                query_pos: torch.Tensor = None, memory_pos: torch.Tensor = None) -> torch.Tensor:
        batch_size, tgt_len, _ = x.shape
        src_len = memory.shape[1]
        
        q = x + query_pos if query_pos is not None else x
        k = memory + memory_pos if memory_pos is not None else memory
        
        q = self.q_proj(q).view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, tgt_len, -1)
        return self.out_proj(out)


class DETRFFN(nn.Module):
    """DETR feed-forward network."""
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class Model(nn.Module):
    """
    DETR Decoder Layer
    
    The core repeated block in DETR decoder.
    Used by: DETR, Deformable DETR, DINO-DETR, DAB-DETR
    
    Architecture (Post-LN):
        queries -> Self-Attention (+ query_pos) -> + residual -> LayerNorm
                -> Cross-Attention (to memory + pos) -> + residual -> LayerNorm
                -> FFN -> + residual -> LayerNorm
    
    Key features:
    - Object queries attend to each other (self-attention)
    - Object queries attend to encoder memory (cross-attention)
    - Learnable query positional embeddings
    """
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        # Self-attention
        self.self_attn = DETRSelfAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        
        # Cross-attention
        self.cross_attn = DETRCrossAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # FFN
        self.ffn = DETRFFN(hidden_size, intermediate_size, dropout)
        self.norm3 = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                query_pos: torch.Tensor = None, memory_pos: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Object queries (batch, num_queries, hidden_size)
            memory: Encoder output (batch, src_len, hidden_size)
            query_pos: Query positional embedding
            memory_pos: Memory positional embedding
        """
        # Self-attention over queries
        residual = x
        x = self.self_attn(x, query_pos)
        x = self.dropout(x)
        x = self.norm1(residual + x)
        
        # Cross-attention to encoder memory
        residual = x
        x = self.cross_attn(x, memory, query_pos, memory_pos)
        x = self.dropout(x)
        x = self.norm2(residual + x)
        
        # FFN
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm3(residual + x)
        
        return x


# Benchmark configuration (DETR-R50)
batch_size = 2
num_queries = 100  # Object queries
memory_len = 850  # Encoder feature map
hidden_size = 256
num_heads = 8
intermediate_size = 2048
dropout = 0.0

def get_inputs():
    queries = torch.randn(batch_size, num_queries, hidden_size)
    memory = torch.randn(batch_size, memory_len, hidden_size)
    query_pos = torch.randn(batch_size, num_queries, hidden_size)
    memory_pos = torch.randn(batch_size, memory_len, hidden_size)
    return [queries, memory, query_pos, memory_pos]

def get_init_inputs():
    return [hidden_size, num_heads, intermediate_size, dropout]

