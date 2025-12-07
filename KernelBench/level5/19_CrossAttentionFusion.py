import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Cross-Attention Fusion for Multimodal Learning.
    
    Fuses information between two modalities (e.g., vision and language)
    using cross-attention mechanisms.
    
    Based on: "Flamingo" and "BLIP" architectures
    """
    def __init__(self, query_dim, kv_dim, hidden_dim, num_heads, num_layers=2):
        """
        :param query_dim: Dimension of query modality (e.g., language)
        :param kv_dim: Dimension of key/value modality (e.g., vision)
        :param hidden_dim: Internal hidden dimension
        :param num_heads: Number of attention heads
        :param num_layers: Number of cross-attention layers
        """
        super(Model, self).__init__()
        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Project KV modality to hidden dim if needed
        self.kv_proj = nn.Linear(kv_dim, hidden_dim) if kv_dim != hidden_dim else nn.Identity()
        
        # Cross-attention layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'q_proj': nn.Linear(query_dim, hidden_dim),
                'k_proj': nn.Linear(hidden_dim, hidden_dim),
                'v_proj': nn.Linear(hidden_dim, hidden_dim),
                'out_proj': nn.Linear(hidden_dim, query_dim),
                'q_norm': nn.LayerNorm(query_dim),
                'kv_norm': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(query_dim, query_dim * 4),
                    nn.GELU(),
                    nn.Linear(query_dim * 4, query_dim)
                ),
                'ffn_norm': nn.LayerNorm(query_dim)
            }))
        
        # Gating for residual (Flamingo-style)
        self.tanh_gate = nn.Parameter(torch.zeros(num_layers))
        
    def forward(self, query_features, kv_features, kv_mask=None):
        """
        Forward pass for cross-attention fusion.
        
        :param query_features: Query modality features (batch, query_seq, query_dim)
        :param kv_features: KV modality features (batch, kv_seq, kv_dim)
        :param kv_mask: Optional mask for KV sequence (batch, kv_seq)
        :return: Fused query features (batch, query_seq, query_dim)
        """
        batch_size, query_seq, _ = query_features.shape
        kv_seq = kv_features.shape[1]
        
        # Project KV features
        kv_features = self.kv_proj(kv_features)
        
        x = query_features
        
        for i, layer in enumerate(self.layers):
            # Pre-norm
            x_norm = layer['q_norm'](x)
            kv_norm = layer['kv_norm'](kv_features)
            
            # Cross-attention
            q = layer['q_proj'](x_norm)
            k = layer['k_proj'](kv_norm)
            v = layer['v_proj'](kv_norm)
            
            # Reshape for multi-head attention
            q = q.view(batch_size, query_seq, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, kv_seq, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, kv_seq, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Attention scores
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # Apply mask if provided
            if kv_mask is not None:
                attn_mask = kv_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, kv_seq)
                attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
            
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_out = torch.matmul(attn_probs, v)
            
            # Reshape back
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, query_seq, self.hidden_dim)
            attn_out = layer['out_proj'](attn_out)
            
            # Gated residual connection
            gate = torch.tanh(self.tanh_gate[i])
            x = x + gate * attn_out
            
            # FFN with residual
            x = x + layer['ffn'](layer['ffn_norm'](x))
        
        return x


# Test parameters
batch_size = 8
query_seq = 256  # Text sequence length
kv_seq = 196     # Image patches (14x14)
query_dim = 768  # Text embedding dim
kv_dim = 1024    # Vision embedding dim
hidden_dim = 768
num_heads = 12
num_layers = 2

def get_inputs():
    query_features = torch.randn(batch_size, query_seq, query_dim)
    kv_features = torch.randn(batch_size, kv_seq, kv_dim)
    return [query_features, kv_features]

def get_init_inputs():
    return [query_dim, kv_dim, hidden_dim, num_heads, num_layers]

