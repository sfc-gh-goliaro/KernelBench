import torch
import torch.nn as nn
import torch.nn.functional as F


TASK_CONFIG = {
    'comparison_mode': 'relative',
    'atol': 1e-5,
    'rtol': 1e-3,
}

class Model(nn.Module):
    """
    Fused Residual Add + LayerNorm/RMSNorm.
    
    A critical pattern in transformers that appears after every attention
    and FFN block:
        output = LayerNorm(x + sublayer_output)
    
    Fusing eliminates:
    1. Intermediate write of x + sublayer_output
    2. Read for LayerNorm input
    
    Reference: Apex FusedLayerNorm, FlashAttention residual fusion
    """
    def __init__(self, hidden_dim, eps=1e-6, norm_type='layernorm', bias=True):
        """
        :param hidden_dim: Hidden dimension
        :param eps: Normalization epsilon
        :param norm_type: 'layernorm' or 'rmsnorm'
        :param bias: Use bias in LayerNorm
        """
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.norm_type = norm_type
        
        # Normalization parameters
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        if bias and norm_type == 'layernorm':
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x, residual):
        """
        Fused residual add + normalization.
        
        :param x: Main input (batch, seq, hidden_dim)
        :param residual: Residual to add (batch, seq, hidden_dim)
        :return: Normalized output (batch, seq, hidden_dim)
        """
        # === FUSED KERNEL START ===
        # Step 1: Add residual (in-place in fused kernel)
        hidden = x + residual
        
        # Step 2: Normalize
        if self.norm_type == 'layernorm':
            # LayerNorm: subtract mean, divide by std
            mean = hidden.mean(dim=-1, keepdim=True)
            var = hidden.var(dim=-1, keepdim=True, unbiased=False)
            hidden = (hidden - mean) * torch.rsqrt(var + self.eps)
            output = hidden * self.weight
            if self.bias is not None:
                output = output + self.bias
        else:
            # RMSNorm: divide by RMS (no mean subtraction)
            variance = hidden.pow(2).mean(dim=-1, keepdim=True)
            output = hidden * torch.rsqrt(variance + self.eps) * self.weight
        # === FUSED KERNEL END ===
        
        return output


# Pre-norm variant (norm before sublayer)
class FusedLayerNormResidual(nn.Module):
    """
    Fused LayerNorm + Sublayer + Residual Add.
    
    For pre-norm architectures:
        output = x + Sublayer(LayerNorm(x))
    """
    def __init__(self, hidden_dim, eps=1e-6, norm_type='rmsnorm'):
        super(FusedLayerNormResidual, self).__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.norm_type = norm_type
        
        self.weight = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, x, sublayer_fn):
        """
        Fused forward with sublayer function.
        
        :param x: Input
        :param sublayer_fn: Function to apply between norm and residual
        :return: x + sublayer_fn(norm(x))
        """
        # === FUSED KERNEL START ===
        # Normalize
        if self.norm_type == 'rmsnorm':
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            normed = x * torch.rsqrt(variance + self.eps) * self.weight
        else:
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            normed = (x - mean) * torch.rsqrt(var + self.eps) * self.weight
        # === FUSED KERNEL END ===
        
        # Apply sublayer
        sublayer_out = sublayer_fn(normed)
        
        # Residual add
        return x + sublayer_out


# Dropout variant
class FusedResidualDropoutLayerNorm(nn.Module):
    """
    Fused Residual + Dropout + LayerNorm.
    
    Full post-norm sequence:
        output = LayerNorm(x + Dropout(sublayer_output))
    """
    def __init__(self, hidden_dim, dropout_prob=0.1, eps=1e-6):
        super(FusedResidualDropoutLayerNorm, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
    
    def forward(self, x, sublayer_output, training=True):
        """
        Fused residual + dropout + layernorm.
        
        :param x: Original input
        :param sublayer_output: Output from sublayer
        :param training: Whether in training mode (for dropout)
        :return: Normalized output
        """
        # === FUSED KERNEL START ===
        # Dropout (if training)
        if training and self.dropout_prob > 0:
            mask = torch.bernoulli(
                torch.full_like(sublayer_output, 1 - self.dropout_prob)
            ) / (1 - self.dropout_prob)
            sublayer_output = sublayer_output * mask
        
        # Residual add
        hidden = x + sublayer_output
        
        # LayerNorm
        mean = hidden.mean(dim=-1, keepdim=True)
        var = hidden.var(dim=-1, keepdim=True, unbiased=False)
        output = (hidden - mean) * torch.rsqrt(var + self.eps)
        output = output * self.weight + self.bias
        # === FUSED KERNEL END ===
        
        return output


# Bias + residual + norm (for some architectures)
class FusedBiasResidualNorm(nn.Module):
    """
    Fused Bias Add + Residual + Normalization.
    
    For layers where bias is fused with residual:
        output = Norm(x + (linear_out + bias))
    """
    def __init__(self, hidden_dim, eps=1e-6):
        super(FusedBiasResidualNorm, self).__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        
        self.norm_weight = nn.Parameter(torch.ones(hidden_dim))
        self.layer_bias = nn.Parameter(torch.zeros(hidden_dim))
    
    def forward(self, x, linear_out):
        """
        Fused bias + residual + RMSNorm.
        
        :param x: Residual input
        :param linear_out: Output from linear layer (without bias)
        :return: Normalized output
        """
        # === FUSED KERNEL START ===
        # Add bias and residual
        hidden = x + linear_out + self.layer_bias
        
        # RMSNorm
        variance = hidden.pow(2).mean(dim=-1, keepdim=True)
        output = hidden * torch.rsqrt(variance + self.eps) * self.norm_weight
        # === FUSED KERNEL END ===
        
        return output


# Test parameters
batch_size = 32
seq_len = 2048
hidden_dim = 4096

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    x = torch.randn(batch_size, seq_len, hidden_dim)
    residual = torch.randn(batch_size, seq_len, hidden_dim)
    return [x, residual]

def get_init_inputs():
    return [hidden_dim]

