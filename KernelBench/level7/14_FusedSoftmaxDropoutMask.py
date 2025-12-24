import torch
import torch.nn as nn
import torch.nn.functional as F


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Fused Softmax + Dropout + Mask Application.
    
    The attention probability computation with all components fused:
    1. Apply attention mask (additive)
    2. Compute softmax
    3. Apply dropout (during training)
    
    This is the hot path in attention and fusion provides significant
    memory savings by not materializing intermediate tensors.
    
    Reference: FlashAttention, Apex fused softmax
    """
    def __init__(self, dim=-1, dropout_prob=0.0, scale=None):
        """
        :param dim: Softmax dimension
        :param dropout_prob: Dropout probability
        :param scale: Pre-softmax scaling factor
        """
        super(Model, self).__init__()
        self.dim = dim
        self.dropout_prob = dropout_prob
        self.scale = scale
    
    def forward(self, scores, attention_mask=None, training=True):
        """
        Fused softmax + dropout + mask.
        
        :param scores: Attention scores (batch, heads, seq_q, seq_k)
        :param attention_mask: Additive mask (broadcastable to scores shape)
        :param training: Whether in training mode
        :return: Attention probabilities
        """
        # === FUSED KERNEL START ===
        # Step 1: Scale (optional)
        if self.scale is not None:
            scores = scores * self.scale
        
        # Step 2: Apply attention mask
        if attention_mask is not None:
            # Additive mask: -inf for masked positions
            scores = scores + attention_mask
        
        # Step 3: Softmax
        attn_probs = F.softmax(scores, dim=self.dim)
        
        # Step 4: Dropout
        if training and self.dropout_prob > 0:
            attn_probs = F.dropout(attn_probs, p=self.dropout_prob, training=True)
        # === FUSED KERNEL END ===
        
        return attn_probs


# Causal softmax variant
class FusedCausalSoftmaxDropout(nn.Module):
    """
    Fused Causal Softmax + Dropout.
    
    Generates and applies causal mask within the kernel.
    """
    def __init__(self, dropout_prob=0.0):
        super(FusedCausalSoftmaxDropout, self).__init__()
        self.dropout_prob = dropout_prob
    
    def forward(self, scores, training=True):
        """
        Fused causal softmax + dropout.
        
        :param scores: Attention scores (batch, heads, seq_q, seq_k)
        :param training: Training mode
        :return: Causal attention probabilities
        """
        batch, heads, seq_q, seq_k = scores.shape
        device = scores.device
        
        # === FUSED KERNEL START ===
        # Generate causal mask in-kernel
        # In FlashAttention, this is done tile-by-tile
        causal_mask = torch.triu(
            torch.ones(seq_q, seq_k, dtype=torch.bool, device=device),
            diagonal=1
        )
        
        # Apply mask
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax
        attn_probs = F.softmax(scores, dim=-1)
        
        # Handle NaN from all-masked rows
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        
        # Dropout
        if training and self.dropout_prob > 0:
            attn_probs = F.dropout(attn_probs, p=self.dropout_prob, training=True)
        # === FUSED KERNEL END ===
        
        return attn_probs


# Sliding window variant
class FusedSlidingWindowSoftmax(nn.Module):
    """
    Fused Sliding Window Softmax + Dropout.
    
    For efficient attention with local window (Mistral, Longformer).
    """
    def __init__(self, window_size=4096, dropout_prob=0.0):
        super(FusedSlidingWindowSoftmax, self).__init__()
        self.window_size = window_size
        self.dropout_prob = dropout_prob
    
    def forward(self, scores, training=True):
        """
        Fused sliding window softmax.
        
        :param scores: Attention scores (batch, heads, seq_q, seq_k)
        :return: Windowed attention probabilities
        """
        batch, heads, seq_q, seq_k = scores.shape
        device = scores.device
        
        # === FUSED KERNEL START ===
        # Create sliding window mask
        row_idx = torch.arange(seq_q, device=device).unsqueeze(1)
        col_idx = torch.arange(seq_k, device=device).unsqueeze(0)
        
        # Mask positions outside window and future positions (causal)
        window_mask = (col_idx > row_idx) | (row_idx - col_idx > self.window_size)
        
        # Apply mask
        scores = scores.masked_fill(window_mask, float('-inf'))
        
        # Softmax
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        
        # Dropout
        if training and self.dropout_prob > 0:
            attn_probs = F.dropout(attn_probs, p=self.dropout_prob, training=True)
        # === FUSED KERNEL END ===
        
        return attn_probs


# Safe softmax with numerical stability
class FusedSafeSoftmaxDropout(nn.Module):
    """
    Fused Safe Softmax + Dropout with enhanced numerical stability.
    
    Handles edge cases like all-masked rows gracefully.
    """
    def __init__(self, dropout_prob=0.0, eps=1e-9):
        super(FusedSafeSoftmaxDropout, self).__init__()
        self.dropout_prob = dropout_prob
        self.eps = eps
    
    def forward(self, scores, attention_mask=None, training=True):
        """
        Safe fused softmax with stability handling.
        """
        # === FUSED KERNEL START ===
        # Apply mask
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Stable softmax: subtract max before exp
        max_scores = scores.max(dim=-1, keepdim=True).values
        max_scores = torch.where(
            torch.isinf(max_scores), 
            torch.zeros_like(max_scores), 
            max_scores
        )
        
        exp_scores = torch.exp(scores - max_scores)
        sum_exp = exp_scores.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        attn_probs = exp_scores / sum_exp
        
        # Dropout
        if training and self.dropout_prob > 0:
            attn_probs = F.dropout(attn_probs, p=self.dropout_prob, training=True)
        # === FUSED KERNEL END ===
        
        return attn_probs


# Sparse attention softmax
class FusedSparseSoftmax(nn.Module):
    """
    Fused Sparse Attention Softmax.
    
    For BigBird/Longformer-style sparse attention patterns.
    """
    def __init__(self, global_tokens=64, random_tokens=64, window_size=256):
        super(FusedSparseSoftmax, self).__init__()
        self.global_tokens = global_tokens
        self.random_tokens = random_tokens
        self.window_size = window_size
    
    def forward(self, scores, global_mask=None):
        """
        Sparse softmax with multiple attention patterns.
        """
        batch, heads, seq_q, seq_k = scores.shape
        device = scores.device
        
        # Create sparse mask combining:
        # 1. Global tokens (first N tokens attend to all)
        # 2. Window attention
        # 3. Random attention
        
        sparse_mask = torch.ones(seq_q, seq_k, dtype=torch.bool, device=device)
        
        # Global: first global_tokens can see everything
        sparse_mask[:self.global_tokens, :] = False
        sparse_mask[:, :self.global_tokens] = False
        
        # Window
        row_idx = torch.arange(seq_q, device=device).unsqueeze(1)
        col_idx = torch.arange(seq_k, device=device).unsqueeze(0)
        window_mask = (row_idx - col_idx).abs() <= self.window_size // 2
        sparse_mask = sparse_mask & ~window_mask
        
        # Apply sparse mask
        scores = scores.masked_fill(sparse_mask, float('-inf'))
        
        return F.softmax(scores, dim=-1)


# Test parameters
batch_size = 8
num_heads = 32
seq_len_q = 2048
seq_len_k = 2048

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    scores = torch.randn(batch_size, num_heads, seq_len_q, seq_len_k)
    # Causal mask
    attention_mask = torch.triu(
        torch.ones(seq_len_q, seq_len_k) * float('-inf'),
        diagonal=1
    )
    return [scores, attention_mask]

def get_init_inputs():
    return [-1, 0.1]  # dim, dropout_prob

