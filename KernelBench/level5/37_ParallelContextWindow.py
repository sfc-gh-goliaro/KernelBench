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
    Parallel Context Window (PCW) Attention.
    
    Extends context by processing multiple windows in parallel,
    with cross-window attention for information sharing.
    
    Based on: "Parallel Context Windows for Large Language Models"
    """
    def __init__(self, dim, num_heads, window_size, num_windows, head_dim=None):
        """
        :param dim: Model dimension
        :param num_heads: Number of attention heads
        :param window_size: Size of each context window
        :param num_windows: Number of parallel windows
        :param head_dim: Dimension per head
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.window_size = window_size
        self.num_windows = num_windows
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
        # Position embedding for window positions
        self.window_pos_embed = nn.Parameter(
            torch.randn(1, num_windows, 1, dim) * 0.02
        )
        
        # Cross-window attention
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        
        # Gate for combining windows
        self.window_gate = nn.Linear(dim, num_windows)
        
    def forward(self, x, window_mask=None):
        """
        Forward pass for Parallel Context Window attention.
        
        :param x: Input tensor (batch, total_seq_len, dim)
                  total_seq_len should equal num_windows * window_size
        :param window_mask: Optional mask for windows
        :return: Output tensor (batch, total_seq_len, dim)
        """
        batch_size, total_seq, _ = x.shape
        
        # Validate sequence length
        expected_len = self.num_windows * self.window_size
        if total_seq != expected_len:
            # Pad or truncate
            if total_seq < expected_len:
                x = F.pad(x, (0, 0, 0, expected_len - total_seq))
            else:
                x = x[:, :expected_len]
        
        # Reshape into windows
        x_windows = x.view(batch_size, self.num_windows, self.window_size, self.dim)
        
        # Add window position embedding
        x_windows = x_windows + self.window_pos_embed
        
        # Process each window independently first
        window_outputs = []
        
        for w in range(self.num_windows):
            x_w = x_windows[:, w]  # (batch, window_size, dim)
            
            # Project Q, K, V
            q = self.q_proj(x_w)
            k = self.k_proj(x_w)
            v = self.v_proj(x_w)
            
            # Reshape for attention
            q = q.view(batch_size, self.window_size, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, self.window_size, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, self.window_size, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Causal attention within window
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            causal_mask = torch.triu(
                torch.ones(self.window_size, self.window_size, device=x.device),
                diagonal=1
            ).bool()
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
            
            attn_probs = F.softmax(attn_scores, dim=-1)
            out_w = torch.matmul(attn_probs, v)
            
            out_w = out_w.transpose(1, 2).contiguous().view(batch_size, self.window_size, -1)
            out_w = self.out_proj(out_w)
            
            window_outputs.append(out_w)
        
        # Stack window outputs
        window_outputs = torch.stack(window_outputs, dim=1)  # (batch, num_windows, window_size, dim)
        
        # Cross-window attention: each position attends to same position in other windows
        cross_outputs = []
        
        for pos in range(self.window_size):
            # Gather same position from all windows
            pos_tokens = window_outputs[:, :, pos]  # (batch, num_windows, dim)
            
            # Cross-attention
            pos_normed = self.cross_norm(pos_tokens)
            cross_out, _ = self.cross_attn(pos_normed, pos_normed, pos_normed)
            cross_out = pos_tokens + cross_out
            
            cross_outputs.append(cross_out)
        
        # Reshape cross outputs
        cross_outputs = torch.stack(cross_outputs, dim=2)  # (batch, num_windows, window_size, dim)
        
        # Compute window importance weights
        window_logits = self.window_gate(cross_outputs.mean(dim=2))  # (batch, num_windows, num_windows)
        window_weights = F.softmax(window_logits, dim=-1)
        
        # Weighted combination across windows
        # Each window position gets weighted contribution from all windows
        output = torch.einsum('bwnl,bwld->bwld', window_weights.unsqueeze(2), cross_outputs)
        
        # Reshape back to sequence
        output = output.view(batch_size, -1, self.dim)
        
        return output[:, :total_seq]


# Test parameters
batch_size = 8
num_windows = 4
window_size = 512
total_seq = num_windows * window_size
dim = 768
num_heads = 12

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    return [torch.randn(batch_size, total_seq, dim)]

def get_init_inputs():
    return [dim, num_heads, window_size, num_windows]

