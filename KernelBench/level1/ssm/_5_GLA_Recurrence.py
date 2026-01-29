import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Gated Linear Attention (GLA) Recurrence
    
    Used by: GLA models
    
    Gated Linear Attention recurrence with data-dependent gating
    for state updates. Combines linear attention efficiency with
    gating for better expressiveness.
    
    Shapes:
        Input: (batch, seq_len, hidden_size)
        Output: (batch, seq_len, hidden_size)
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, expand_k: float = 1.0, expand_v: float = 2.0):
        """
        Initialize GLA.
        
        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            expand_k: Key dimension expansion factor
            expand_v: Value dimension expansion factor
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.key_dim = int(self.head_dim * expand_k)
        self.value_dim = int(self.head_dim * expand_v)
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.value_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, num_heads * self.key_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.value_dim, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        GLA forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            
        Returns:
            Output tensor (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.key_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.key_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.value_dim)
        g = self.g_proj(x).view(batch_size, seq_len, self.num_heads, self.key_dim)
        
        # Apply gating (sigmoid gate for decay)
        g = torch.sigmoid(g)
        
        # Feature maps (simple normalization for linear attention)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Gate the keys
        k = k * g
        
        # Linear attention with recurrence
        # S_t = S_{t-1} + k_t^T v_t
        # o_t = q_t S_t
        
        # Initialize state: (batch, num_heads, key_dim, value_dim)
        S = torch.zeros(batch_size, self.num_heads, self.key_dim, self.value_dim, 
                       device=x.device, dtype=x.dtype)
        
        outputs = []
        for t in range(seq_len):
            # Update state: S += outer(k, v)
            S = S + torch.einsum('bhk,bhv->bhkv', k[:, t], v[:, t])
            
            # Compute output: o = q @ S
            o = torch.einsum('bhk,bhkv->bhv', q[:, t], S)
            
            # Normalize
            z = torch.einsum('bhk,bhk->bh', q[:, t], k[:, t].cumsum(dim=1)[:, t] if t > 0 else k[:, t])
            o = o / (z.unsqueeze(-1) + 1e-6)
            
            outputs.append(o)
        
        output = torch.stack(outputs, dim=1)  # (batch, seq, num_heads, value_dim)
        output = output.reshape(batch_size, seq_len, -1)
        
        return self.o_proj(output)


# ============================================================================
# Benchmark Configuration
# ============================================================================
