import torch
import torch.nn as nn
import torch.nn.functional as F
import math


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Sliding Window Attention (SWA).
    
    Each token attends only to tokens within a fixed window,
    enabling linear memory complexity for long sequences.
    
    Based on: "Longformer" and "Mistral" architectures
    """
    def __init__(self, dim, num_heads, window_size, head_dim=None, use_global=False):
        """
        :param dim: Model dimension
        :param num_heads: Number of attention heads
        :param window_size: Size of attention window (one side)
        :param head_dim: Dimension per head
        :param use_global: Whether to include global attention tokens
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.window_size = window_size
        self.use_global = use_global
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
        # Global attention projections (if enabled)
        if use_global:
            self.q_global = nn.Linear(dim, num_heads * self.head_dim, bias=False)
            self.k_global = nn.Linear(dim, num_heads * self.head_dim, bias=False)
            self.v_global = nn.Linear(dim, num_heads * self.head_dim, bias=False)
    
    def _compute_window_attention(self, q, k, v, attention_mask=None):
        """
        Compute sliding window attention.
        
        :param q, k, v: (batch, num_heads, seq_len, head_dim)
        :param attention_mask: Optional mask
        :return: Attention output
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        window = self.window_size
        
        # Pad sequence for windowing
        pad_len = window
        k_padded = F.pad(k, (0, 0, pad_len, pad_len))
        v_padded = F.pad(v, (0, 0, pad_len, pad_len))
        
        # Create sliding window views
        # For each position i, attend to [i-window, i+window]
        outputs = []
        
        for i in range(seq_len):
            # Extract window for position i
            start = i
            end = i + 2 * window + 1
            
            k_window = k_padded[:, :, start:end]  # (batch, heads, 2*window+1, head_dim)
            v_window = v_padded[:, :, start:end]
            q_i = q[:, :, i:i+1]  # (batch, heads, 1, head_dim)
            
            # Compute attention for this position
            attn_scores = torch.matmul(q_i, k_window.transpose(-2, -1)) * self.scale
            
            # Apply causal mask within window
            window_len = k_window.shape[2]
            causal_pos = window  # Current position in window
            causal_mask = torch.zeros(1, 1, 1, window_len, device=q.device)
            causal_mask[:, :, :, causal_pos+1:] = float('-inf')
            attn_scores = attn_scores + causal_mask
            
            attn_probs = F.softmax(attn_scores, dim=-1)
            out_i = torch.matmul(attn_probs, v_window)
            outputs.append(out_i)
        
        return torch.cat(outputs, dim=2)
    
    def _compute_chunked_attention(self, q, k, v):
        """
        Efficient chunked implementation of sliding window attention.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        window = self.window_size
        chunk_size = window * 2
        
        # Pad to multiple of chunk_size
        pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
        
        padded_len = q.shape[2]
        num_chunks = padded_len // chunk_size
        
        # Reshape into chunks
        q_chunks = q.view(batch_size, num_heads, num_chunks, chunk_size, head_dim)
        k_chunks = k.view(batch_size, num_heads, num_chunks, chunk_size, head_dim)
        v_chunks = v.view(batch_size, num_heads, num_chunks, chunk_size, head_dim)
        
        # Add overlapping context from adjacent chunks
        k_context = F.pad(k_chunks, (0, 0, 0, 0, 0, 1))[:, :, 1:]  # Next chunk
        v_context = F.pad(v_chunks, (0, 0, 0, 0, 0, 1))[:, :, 1:]
        
        k_extended = torch.cat([k_chunks, k_context], dim=3)
        v_extended = torch.cat([v_chunks, v_context], dim=3)
        
        # Compute attention per chunk
        attn_scores = torch.einsum('bhncd,bhnkd->bhnck', q_chunks, k_extended) * self.scale
        
        # Create sliding window mask
        chunk_len = q_chunks.shape[3]
        extended_len = k_extended.shape[3]
        mask = torch.ones(chunk_len, extended_len, device=q.device)
        for i in range(chunk_len):
            # Each position can attend to window positions around it
            mask[i, max(0, i-window):min(extended_len, i+window+1)] = 0
        mask = mask.bool()
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = attn_probs.nan_to_num(0)
        
        out_chunks = torch.einsum('bhnck,bhnkd->bhncd', attn_probs, v_extended)
        
        # Reshape back
        output = out_chunks.view(batch_size, num_heads, padded_len, head_dim)
        
        return output[:, :, :seq_len]
    
    def forward(self, x, global_mask=None):
        """
        Forward pass for Sliding Window Attention.
        
        :param x: Input tensor (batch, seq_len, dim)
        :param global_mask: Optional mask for global attention tokens
        :return: Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute sliding window attention
        if seq_len <= self.window_size * 4:
            # For short sequences, use simple implementation
            out = self._compute_window_attention(q, k, v)
        else:
            # For long sequences, use chunked implementation
            out = self._compute_chunked_attention(q, k, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out)


# Test parameters
batch_size = 8
seq_len = 4096
dim = 1024
num_heads = 16
window_size = 256
head_dim = 64

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    return [torch.randn(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, num_heads, window_size, head_dim]

