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
    Lightning Attention for efficient long-context processing.
    
    Combines linear attention with IO-aware optimizations for
    fast training and inference. Used in MiniMax-M2.
    
    Based on: "Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models"
    """
    def __init__(self, dim, num_heads, head_dim=None, use_decay=True):
        """
        :param dim: Model dimension
        :param num_heads: Number of attention heads
        :param head_dim: Dimension per head
        :param use_decay: Whether to use exponential decay
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.use_decay = use_decay
        
        # Projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
        # Decay factors (learnable)
        if use_decay:
            self.log_decay = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        
        # Output gate
        self.gate = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        
        # Layer norm
        self.norm = nn.LayerNorm(num_heads * self.head_dim)
    
    def _compute_decay_mask(self, seq_len, device):
        """Compute exponential decay mask."""
        if not self.use_decay:
            return None
        
        # Decay factor per position
        decay = torch.exp(-torch.exp(self.log_decay))  # (heads, head_dim)
        
        # Position-based decay
        positions = torch.arange(seq_len, device=device).float()
        decay_mask = decay.unsqueeze(-1) ** positions.unsqueeze(0).unsqueeze(0)
        
        return decay_mask  # (heads, head_dim, seq_len)
    
    def forward(self, x, attention_mask=None):
        """
        Forward pass with Lightning Attention.
        
        :param x: Input tensor (batch, seq_len, dim)
        :param attention_mask: Optional attention mask
        :return: Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Compute output gate
        g = torch.sigmoid(self.gate(x))
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply feature map (simple normalization + elu)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Compute decay mask
        decay_mask = self._compute_decay_mask(seq_len, x.device)
        
        if decay_mask is not None:
            # Apply decay to K
            k = k * decay_mask.unsqueeze(0).permute(0, 1, 3, 2)
        
        # Linear attention with cumulative sum (causal)
        # O(N) complexity
        kv = torch.einsum('bhnd,bhnm->bhdm', k, v)  # (batch, heads, head_dim, head_dim)
        
        # Cumulative KV for causal attention
        kv_cumsum = torch.zeros(batch_size, self.num_heads, self.head_dim, self.head_dim, 
                               device=x.device)
        outputs = []
        
        for t in range(seq_len):
            kt = k[:, :, t:t+1]  # (batch, heads, 1, head_dim)
            vt = v[:, :, t:t+1]  # (batch, heads, 1, head_dim)
            qt = q[:, :, t:t+1]
            
            # Update cumulative KV
            kv_t = torch.einsum('bhnd,bhnm->bhdm', kt, vt)
            kv_cumsum = kv_cumsum + kv_t
            
            # Compute output for this timestep
            out_t = torch.einsum('bhnd,bhdm->bhnm', qt, kv_cumsum)
            
            # Normalize
            k_sum = kt.sum(dim=-1, keepdim=True)
            k_cumsum = k[:, :, :t+1].sum(dim=2, keepdim=True)
            normalizer = torch.einsum('bhnd,bhnd->bhn', qt, k_cumsum).unsqueeze(-1)
            out_t = out_t / (normalizer + 1e-6)
            
            outputs.append(out_t)
        
        # Concatenate outputs
        out = torch.cat(outputs, dim=2)  # (batch, heads, seq, head_dim)
        
        # Apply gate
        out = out * g
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.norm(out)
        out = self.out_proj(out)
        
        return out


# Test parameters
batch_size = 8
seq_len = 4096
dim = 2048
num_heads = 16

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
    return [dim, num_heads]

