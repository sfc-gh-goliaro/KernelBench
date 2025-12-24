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
    Differential Attention (Diff Attention).
    
    Computes attention as difference of two softmax attention maps,
    reducing noise and improving signal-to-noise ratio.
    
    Based on: "Differential Transformer" (Microsoft Research)
    """
    def __init__(self, dim, num_heads, head_dim=None, lambda_init=0.8):
        """
        :param dim: Model dimension
        :param num_heads: Number of attention heads
        :param head_dim: Dimension per head
        :param lambda_init: Initial value for lambda parameter
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim else dim // num_heads
        # Each head is split into two sub-heads for differential attention
        self.sub_head_dim = self.head_dim // 2
        self.scale = self.sub_head_dim ** -0.5
        
        # Projections - each produces 2 sets of Q, K (for the two attention maps)
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
        # Learnable lambda for controlling the subtraction
        # Initialized close to 1 for near-standard attention initially
        self.lambda_q1 = nn.Parameter(torch.randn(num_heads, 1, self.sub_head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, 1, self.sub_head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(num_heads, 1, self.sub_head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, 1, self.sub_head_dim) * 0.1)
        self.lambda_init = nn.Parameter(torch.tensor(lambda_init))
        
        # Layer normalization for sub-attention outputs
        self.sublayer_norm = nn.LayerNorm(self.head_dim)
        
    def _compute_lambda(self):
        """Compute the lambda values for the two attention branches."""
        lambda1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lambda2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        return lambda1.unsqueeze(-1), lambda2.unsqueeze(-1)  # (heads, 1, 1)
    
    def forward(self, x, attention_mask=None):
        """
        Forward pass for Differential Attention.
        
        :param x: Input tensor (batch, seq_len, dim)
        :param attention_mask: Optional attention mask
        :return: Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape: split into heads and sub-heads
        q = q.view(batch_size, seq_len, self.num_heads, 2, self.sub_head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, 2, self.sub_head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention: (batch, heads, seq, ...)
        q1 = q[:, :, :, 0].transpose(1, 2)  # (batch, heads, seq, sub_head_dim)
        q2 = q[:, :, :, 1].transpose(1, 2)
        k1 = k[:, :, :, 0].transpose(1, 2)
        k2 = k[:, :, :, 1].transpose(1, 2)
        v = v.transpose(1, 2)  # (batch, heads, seq, head_dim)
        
        # Compute two attention score matrices
        attn_scores1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scale
        attn_scores2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores1 = attn_scores1 + attention_mask
            attn_scores2 = attn_scores2 + attention_mask
        
        # Compute attention probabilities
        attn_probs1 = F.softmax(attn_scores1, dim=-1)
        attn_probs2 = F.softmax(attn_scores2, dim=-1)
        
        # Get lambda values
        lambda1, lambda2 = self._compute_lambda()
        
        # Differential attention: weighted difference of attention maps
        # The subtraction cancels out common (noisy) patterns
        diff_attn = lambda1 * attn_probs1 - lambda2 * attn_probs2
        
        # Rescale to maintain magnitude (optional normalization)
        # Scale by (1 - lambda_init) to control the initial behavior
        diff_attn = diff_attn * (1 - self.lambda_init)
        
        # Apply to values
        out = torch.matmul(diff_attn, v)
        
        # Apply sublayer normalization per head
        out = out.transpose(1, 2)  # (batch, seq, heads, head_dim)
        out = self.sublayer_norm(out)
        
        # Reshape and project
        out = out.reshape(batch_size, seq_len, -1)
        out = self.out_proj(out)
        
        return out


# Test parameters
batch_size = 16
seq_len = 512
dim = 768
num_heads = 12
head_dim = 64

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
    return [dim, num_heads, head_dim]

