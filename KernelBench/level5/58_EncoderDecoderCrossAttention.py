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
    Encoder-Decoder Cross-Attention for sequence-to-sequence models.
    
    Implements the cross-attention mechanism between encoder outputs
    and decoder hidden states. Used in Whisper, T5, BART, etc.
    
    Based on: "Attention Is All You Need" encoder-decoder attention
    """
    def __init__(self, decoder_dim, encoder_dim, num_heads, dropout=0.0):
        """
        :param decoder_dim: Decoder hidden dimension
        :param encoder_dim: Encoder output dimension
        :param num_heads: Number of attention heads
        :param dropout: Dropout rate
        """
        super(Model, self).__init__()
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.num_heads = num_heads
        self.head_dim = decoder_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query projection (from decoder)
        self.q_proj = nn.Linear(decoder_dim, decoder_dim, bias=False)
        
        # Key and Value projections (from encoder)
        self.k_proj = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.v_proj = nn.Linear(encoder_dim, decoder_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(decoder_dim, decoder_dim, bias=False)
        
        # Layer norm and dropout
        self.norm = nn.LayerNorm(decoder_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Cached encoder KV for efficient decoding
        self.register_buffer('cached_k', None, persistent=False)
        self.register_buffer('cached_v', None, persistent=False)
    
    def cache_encoder_kv(self, encoder_output):
        """
        Pre-compute and cache encoder KV for efficient decoding.
        
        :param encoder_output: Encoder output (batch, enc_len, encoder_dim)
        """
        batch_size, enc_len, _ = encoder_output.shape
        
        k = self.k_proj(encoder_output)
        v = self.v_proj(encoder_output)
        
        k = k.view(batch_size, enc_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, enc_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        self.cached_k = k
        self.cached_v = v
    
    def forward(self, decoder_hidden, encoder_output=None, encoder_mask=None, 
                use_cache=False):
        """
        Cross-attention from decoder to encoder.
        
        :param decoder_hidden: Decoder hidden states (batch, dec_len, decoder_dim)
        :param encoder_output: Encoder output (batch, enc_len, encoder_dim)
        :param encoder_mask: Mask for encoder output (batch, enc_len)
        :param use_cache: Whether to use cached encoder KV
        :return: Cross-attended output (batch, dec_len, decoder_dim)
        """
        batch_size, dec_len, _ = decoder_hidden.shape
        
        # Pre-norm
        decoder_norm = self.norm(decoder_hidden)
        
        # Query projection
        q = self.q_proj(decoder_norm)
        q = q.view(batch_size, dec_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Key and Value (from cache or compute fresh)
        if use_cache and self.cached_k is not None:
            k = self.cached_k
            v = self.cached_v
        else:
            assert encoder_output is not None, "encoder_output required if not using cache"
            enc_len = encoder_output.shape[1]
            
            k = self.k_proj(encoder_output)
            v = self.v_proj(encoder_output)
            
            k = k.view(batch_size, enc_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, enc_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply encoder mask
        if encoder_mask is not None:
            attn_mask = encoder_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, enc_len)
            attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply to values
        out = torch.matmul(attn_probs, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, dec_len, self.decoder_dim)
        out = self.out_proj(out)
        
        # Residual connection
        return decoder_hidden + out


# Test parameters
batch_size = 8
dec_len = 256
enc_len = 1500
decoder_dim = 512
encoder_dim = 512
num_heads = 8

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    decoder_hidden = torch.randn(batch_size, dec_len, decoder_dim)
    encoder_output = torch.randn(batch_size, enc_len, encoder_dim)
    return [decoder_hidden, encoder_output]

def get_init_inputs():
    return [decoder_dim, encoder_dim, num_heads]

