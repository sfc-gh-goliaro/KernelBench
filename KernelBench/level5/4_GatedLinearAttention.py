import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Gated Linear Attention (GLA) mechanism.
    
    GLA extends linear attention with data-dependent gating, achieving better
    performance while maintaining the linear complexity benefits.
    
    Based on: "Gated Linear Attention Transformers with Hardware-Efficient Training"
    """
    def __init__(self, dim, num_heads, seq_len, expansion_factor=2):
        """
        :param dim: Model dimension
        :param num_heads: Number of attention heads
        :param seq_len: Sequence length
        :param expansion_factor: Expansion factor for intermediate dimension
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.seq_len = seq_len
        self.expansion_factor = expansion_factor
        
        # Input projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        # Gating projections
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        
        # Forget gate (data-dependent decay)
        self.forget_gate = nn.Linear(dim, num_heads, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        Forward pass for Gated Linear Attention.
        
        :param x: Input tensor of shape (batch_size, seq_len, dim)
        :return: Output tensor of shape (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Compute gating
        gate = torch.sigmoid(self.gate_proj(x))
        
        # Compute data-dependent forget gates (decay)
        forget = torch.sigmoid(self.forget_gate(x))  # (batch, seq, heads)
        forget = forget.unsqueeze(-1)  # (batch, seq, heads, 1)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply feature map (simple normalization)
        q = F.silu(q)
        k = F.silu(k)
        
        # Chunk-wise computation for efficiency
        chunk_size = min(64, seq_len)
        num_chunks = seq_len // chunk_size
        
        outputs = []
        state = torch.zeros(batch_size, self.num_heads, self.head_dim, self.head_dim, 
                           device=x.device, dtype=x.dtype)
        
        for c in range(num_chunks):
            start = c * chunk_size
            end = start + chunk_size
            
            q_chunk = q[:, start:end]  # (batch, chunk, heads, head_dim)
            k_chunk = k[:, start:end]
            v_chunk = v[:, start:end]
            forget_chunk = forget[:, start:end]  # (batch, chunk, heads, 1)
            
            # Intra-chunk attention
            # Build causal mask within chunk
            intra_decay = torch.ones(chunk_size, chunk_size, device=x.device)
            intra_decay = torch.tril(intra_decay)
            
            # Q @ K^T within chunk
            attn_weights = torch.einsum('bchd,bcjd->bchj', q_chunk, k_chunk)
            attn_weights = attn_weights * intra_decay.unsqueeze(0).unsqueeze(0)
            
            # Normalize
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-9)
            
            # Intra-chunk output
            intra_out = torch.einsum('bchj,bcjd->bchd', attn_weights, v_chunk)
            
            # Inter-chunk: use state from previous chunks
            inter_out = torch.einsum('bchd,bhde->bche', q_chunk, state)
            
            # Combine with gating
            chunk_out = intra_out + inter_out
            
            outputs.append(chunk_out)
            
            # Update state with decay
            decay_factor = forget_chunk.mean(dim=1)  # (batch, heads, 1)
            kv_update = torch.einsum('bchd,bche->bhde', k_chunk, v_chunk)
            state = state * decay_factor.unsqueeze(-1) + kv_update / chunk_size
        
        # Concatenate chunks
        out = torch.cat(outputs, dim=1)  # (batch, seq, heads, head_dim)
        out = out.reshape(batch_size, seq_len, self.dim)
        
        # Apply output gating
        out = gate * out
        out = self.norm(out)
        
        return self.out_proj(out)


# Test parameters
batch_size = 16
seq_len = 512
dim = 512
num_heads = 8

def get_inputs():
    return [torch.randn(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, num_heads, seq_len]

