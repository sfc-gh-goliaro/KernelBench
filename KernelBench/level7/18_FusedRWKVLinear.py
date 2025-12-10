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
    Fused RWKV Linear Attention.
    
    RWKV combines RNN-like recurrence with attention-like operations:
    1. Time-mixing: combines current and previous state
    2. WKV computation: weighted key-value aggregation with decay
    3. Output projection
    
    This fuses the complete RWKV time-mixing block.
    
    Reference: RWKV-v5/v6
    """
    def __init__(self, hidden_dim, num_heads=None, head_dim=64):
        """
        :param hidden_dim: Model dimension
        :param num_heads: Number of heads
        :param head_dim: Dimension per head
        """
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.num_heads = num_heads if num_heads else hidden_dim // head_dim
        
        # Time mixing parameters
        self.time_mix_k = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.time_mix_g = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        
        # Projections
        self.receptance = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Time decay (trainable per head)
        self.time_decay = nn.Parameter(torch.randn(self.num_heads, self.head_dim) * 0.1)
        self.time_first = nn.Parameter(torch.randn(self.num_heads, self.head_dim) * 0.1)
        
        # LayerNorm
        self.ln_x = nn.GroupNorm(num_groups=self.num_heads, num_channels=hidden_dim)
    
    def _wkv_compute(self, r, k, v, state):
        """
        Compute WKV with state update.
        
        :param r: Receptance (batch, seq, heads, head_dim)
        :param k: Key (batch, seq, heads, head_dim)
        :param v: Value (batch, seq, heads, head_dim)
        :param state: Previous state (batch, heads, head_dim, head_dim)
        :return: Output and new state
        """
        batch, seq_len, heads, head_dim = r.shape
        device = r.device
        
        # Time decay
        w = torch.exp(-torch.exp(self.time_decay))  # (heads, head_dim)
        u = self.time_first  # (heads, head_dim)
        
        outputs = []
        
        if state is None:
            state = torch.zeros(batch, heads, head_dim, head_dim, device=device)
        
        for t in range(seq_len):
            rt = r[:, t]  # (batch, heads, head_dim)
            kt = k[:, t]
            vt = v[:, t]
            
            # WKV computation
            # wkv = (state + u * k * v) * r / (state.sum + u * k + eps)
            
            # Simplified formulation
            kv = kt.unsqueeze(-1) * vt.unsqueeze(-2)  # (batch, heads, head_dim, head_dim)
            
            # Output: receptance weighted sum
            wkv = (state + u.unsqueeze(0).unsqueeze(-1) * kv)
            wkv = torch.einsum('bhij,bhi->bhj', wkv, torch.sigmoid(rt))
            
            outputs.append(wkv)
            
            # State update: exponential decay
            state = state * w.unsqueeze(0).unsqueeze(-1) + kv
        
        return torch.stack(outputs, dim=1), state
    
    def forward(self, x, state=None):
        """
        Fused RWKV linear attention forward.
        
        :param x: Input (batch, seq, hidden_dim)
        :param state: Previous state for incremental decoding
        :return: Output and new state
        """
        batch, seq_len, _ = x.shape
        
        # === FUSED KERNEL START ===
        # Get previous token for time mixing (use zeros for first)
        if seq_len > 1:
            x_prev = F.pad(x[:, :-1], (0, 0, 1, 0))
        else:
            x_prev = torch.zeros_like(x)
        
        # Time mixing
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_prev * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + x_prev * (1 - self.time_mix_g)
        
        # Projections
        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))
        
        # Reshape for multi-head
        r = r.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim)
        
        # WKV computation
        wkv, new_state = self._wkv_compute(r, k, v, state)
        
        # Reshape and apply gate
        wkv = wkv.view(batch, seq_len, self.hidden_dim)
        
        # Group norm (in RWKV, this is channel-last group norm)
        wkv = wkv.transpose(1, 2)  # (batch, hidden, seq)
        wkv = self.ln_x(wkv)
        wkv = wkv.transpose(1, 2)
        
        # Gate and output
        output = self.output(wkv * g)
        # === FUSED KERNEL END ===
        
        return output, new_state


# Griffin-style Linear Attention
class FusedGriffinLinear(nn.Module):
    """
    Fused Griffin Linear Attention.
    
    Griffin uses a simpler linear attention variant.
    """
    def __init__(self, hidden_dim, expansion=2):
        super(FusedGriffinLinear, self).__init__()
        self.hidden_dim = hidden_dim
        self.expanded_dim = hidden_dim * expansion
        
        # Projections
        self.input_proj = nn.Linear(hidden_dim, 3 * self.expanded_dim, bias=False)
        self.conv = nn.Conv1d(self.expanded_dim, self.expanded_dim, 
                             kernel_size=4, padding=3, groups=self.expanded_dim)
        self.output_proj = nn.Linear(self.expanded_dim, hidden_dim, bias=False)
    
    def forward(self, x, state=None):
        batch, seq_len, _ = x.shape
        
        # Project and split
        projected = self.input_proj(x)
        gate, linear, value = projected.chunk(3, dim=-1)
        
        # Causal conv (for temporal mixing)
        linear = linear.transpose(1, 2)
        linear = self.conv(linear)[:, :, :seq_len]
        linear = linear.transpose(1, 2)
        
        # Gating
        output = F.silu(gate) * linear * torch.sigmoid(value)
        
        return self.output_proj(output), None


# Test parameters
batch_size = 8
seq_len = 2048
hidden_dim = 2048

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    return [torch.randn(batch_size, seq_len, hidden_dim)]

def get_init_inputs():
    return [hidden_dim]

