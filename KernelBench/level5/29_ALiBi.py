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
    Attention with Linear Biases (ALiBi).
    
    Adds linear position-dependent biases to attention scores,
    enabling extrapolation to longer sequences than seen during training.
    
    Based on: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
    """
    def __init__(self, num_heads, max_seq_len):
        """
        :param num_heads: Number of attention heads
        :param max_seq_len: Maximum sequence length
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Compute slopes for each head
        # Slopes are powers of 2^(-8/n) for n heads
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)
        
        # Precompute bias matrix
        self._build_bias_cache(max_seq_len)
    
    def _get_slopes(self, num_heads):
        """Compute ALiBi slopes for each head."""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            # For non-power-of-2 heads, interpolate
            closest_power = 2 ** math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power)
            extra_slopes = get_slopes_power_of_2(2 * closest_power)[0::2]
            slopes = slopes + extra_slopes[:num_heads - closest_power]
        
        return torch.tensor(slopes).float()
    
    def _build_bias_cache(self, seq_len):
        """Build the position bias matrix."""
        # Distance matrix: bias[i,j] = -|i-j|
        positions = torch.arange(seq_len)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq, seq)
        
        # Apply slopes per head
        # bias shape: (num_heads, seq, seq)
        alibi = self.slopes.unsqueeze(1).unsqueeze(2) * distance.unsqueeze(0)
        
        self.register_buffer('alibi_cache', alibi, persistent=False)
    
    def get_bias(self, seq_len, device=None):
        """Get ALiBi bias for given sequence length."""
        if seq_len > self.alibi_cache.shape[1]:
            self._build_bias_cache(seq_len)
            if device is not None:
                self.alibi_cache = self.alibi_cache.to(device)
        
        return self.alibi_cache[:, :seq_len, :seq_len]
    
    def forward(self, attention_scores, seq_len=None):
        """
        Add ALiBi bias to attention scores.
        
        :param attention_scores: Attention scores (batch, num_heads, seq_q, seq_k)
        :param seq_len: Optional sequence length (if different from scores shape)
        :return: Biased attention scores
        """
        batch_size, num_heads, seq_q, seq_k = attention_scores.shape
        
        # Get appropriate bias
        if seq_len is None:
            seq_len = max(seq_q, seq_k)
        
        alibi = self.get_bias(seq_len, attention_scores.device)
        
        # Handle query-key length mismatch (for cross-attention or KV cache)
        alibi = alibi[:, :seq_q, :seq_k]
        
        # Add bias to attention scores
        return attention_scores + alibi.unsqueeze(0)
    
    def forward_with_attention(self, q, k, v, attention_mask=None):
        """
        Complete attention with ALiBi.
        
        :param q: Query tensor (batch, num_heads, seq_q, head_dim)
        :param k: Key tensor (batch, num_heads, seq_k, head_dim)
        :param v: Value tensor (batch, num_heads, seq_k, head_dim)
        :param attention_mask: Optional attention mask
        :return: Attention output (batch, num_heads, seq_q, head_dim)
        """
        head_dim = q.shape[-1]
        scale = head_dim ** -0.5
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Add ALiBi bias
        scores = self.forward(scores)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return output


# Test parameters
batch_size = 16
num_heads = 32
seq_len = 2048
head_dim = 128

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    attention_scores = torch.randn(batch_size, num_heads, seq_len, seq_len)
    return [attention_scores]

def get_init_inputs():
    return [num_heads, seq_len * 2]

