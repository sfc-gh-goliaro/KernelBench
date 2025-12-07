import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    RWKV (Receptance Weighted Key Value) Linear Attention mechanism.
    
    RWKV combines the parallelizable training of Transformers with the 
    efficient inference of RNNs through a linear attention formulation.
    
    Based on: "RWKV: Reinventing RNNs for the Transformer Era"
    """
    def __init__(self, dim, num_heads, seq_len):
        """
        :param dim: Model dimension
        :param num_heads: Number of attention heads
        :param seq_len: Sequence length
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.seq_len = seq_len
        
        # Time-mixing parameters
        self.time_decay = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.time_first = nn.Parameter(torch.randn(num_heads, self.head_dim))
        
        # Mixing coefficients
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, dim) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, dim) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, dim) * 0.5)
        
        # Projections
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x, state=None):
        """
        Forward pass for RWKV linear attention.
        
        :param x: Input tensor of shape (batch_size, seq_len, dim)
        :param state: Optional previous state for recurrent computation
        :return: Output tensor of shape (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Shift for time-mixing (using previous timestep info)
        x_shifted = F.pad(x, (0, 0, 1, -1))
        
        # Time-mixing
        xk = x * self.time_mix_k + x_shifted * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_shifted * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + x_shifted * (1 - self.time_mix_r)
        
        # Compute r, k, v
        r = torch.sigmoid(self.receptance(xr))
        k = self.key(xk)
        v = self.value(xv)
        
        # Reshape for multi-head
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # WKV computation (parallelized version)
        # This is a simplified version - full RWKV uses custom CUDA kernels
        w = torch.exp(-torch.exp(self.time_decay.unsqueeze(0).unsqueeze(0)))
        u = self.time_first.unsqueeze(0).unsqueeze(0)
        
        # Compute attention weights with exponential decay
        wkv = torch.zeros(batch_size, seq_len, self.num_heads, self.head_dim, device=x.device)
        
        for t in range(seq_len):
            kt = k[:, t:t+1]
            vt = v[:, t:t+1]
            
            if t == 0:
                wkv[:, t] = (u * kt * vt).squeeze(1)
                state_num = kt * vt
                state_den = kt
            else:
                wkv_t = (u * kt * vt + state_num) / (u * kt + state_den + 1e-9)
                wkv[:, t] = wkv_t.squeeze(1)
                state_num = w * state_num + kt * vt
                state_den = w * state_den + kt
        
        wkv = wkv.view(batch_size, seq_len, self.dim)
        
        # Apply receptance and output projection
        out = r * wkv
        return self.output(out)


# Test parameters
batch_size = 16
seq_len = 512
dim = 768
num_heads = 12

def get_inputs():
    return [torch.randn(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, num_heads, seq_len]

