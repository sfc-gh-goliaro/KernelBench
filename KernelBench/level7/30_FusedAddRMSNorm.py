import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused Add + RMSNorm (fused_add_rms_norm).
    
    The standard transformer residual pattern:
        output = RMSNorm(x + residual)
    
    This specific fusion is used extensively in vLLM and SGLang for
    post-attention and post-FFN normalization.
    
    Benefits:
    - Single memory read for x and residual
    - Single write for normalized output
    - Can optionally return the pre-norm hidden state for next residual
    
    Reference: vLLM fused_add_rms_norm, SGLang kernels
    """
    def __init__(self, hidden_dim, eps=1e-6):
        """
        :param hidden_dim: Hidden dimension
        :param eps: RMSNorm epsilon
        """
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, x, residual):
        """
        Fused add + RMSNorm.
        
        :param x: Input from sublayer (batch, seq, hidden_dim)
        :param residual: Residual connection (batch, seq, hidden_dim)
        :return: Tuple of (normalized_output, hidden_for_next_residual)
        """
        # === FUSED KERNEL START ===
        # Step 1: Residual add
        hidden = x + residual
        
        # Step 2: RMSNorm
        variance = hidden.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden * torch.rsqrt(variance + self.eps) * self.weight
        # === FUSED KERNEL END ===
        
        # Return both normalized output AND the hidden state
        # (hidden is needed for next layer's residual)
        return normalized, hidden


# In-place variant for memory efficiency
class FusedAddRMSNormInplace(nn.Module):
    """
    In-place Fused Add + RMSNorm.
    
    Modifies residual buffer in-place to save memory.
    """
    def __init__(self, hidden_dim, eps=1e-6):
        super(FusedAddRMSNormInplace, self).__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, x, residual):
        """
        In-place fused add + RMSNorm.
        
        WARNING: Modifies residual in-place!
        
        :param x: Sublayer output
        :param residual: Residual (modified in-place to become new residual)
        :return: Normalized output
        """
        # === FUSED KERNEL (in-place) ===
        # Update residual in-place
        residual.add_(x)
        
        # Compute RMSNorm on updated residual
        variance = residual.pow(2).mean(dim=-1, keepdim=True)
        output = residual * torch.rsqrt(variance + self.eps) * self.weight
        # === END FUSED KERNEL ===
        
        return output


# Variant with dropout
class FusedDropoutAddRMSNorm(nn.Module):
    """
    Fused Dropout + Add + RMSNorm.
    
    For training with dropout:
        output = RMSNorm(residual + Dropout(x))
    """
    def __init__(self, hidden_dim, dropout_prob=0.0, eps=1e-6):
        super(FusedDropoutAddRMSNorm, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, x, residual, training=True):
        """
        Fused dropout + add + RMSNorm.
        """
        # === FUSED KERNEL ===
        # Dropout
        if training and self.dropout_prob > 0:
            x = F.dropout(x, p=self.dropout_prob, training=True)
        
        # Add
        hidden = x + residual
        
        # RMSNorm
        variance = hidden.pow(2).mean(dim=-1, keepdim=True)
        output = hidden * torch.rsqrt(variance + self.eps) * self.weight
        # === END FUSED KERNEL ===
        
        return output, hidden


# Two-residual variant (for some architectures)
class FusedAddAddRMSNorm(nn.Module):
    """
    Fused Add + Add + RMSNorm for parallel sublayers.
    
    For architectures with parallel attention/FFN:
        output = RMSNorm(x + attn_out + ffn_out)
    """
    def __init__(self, hidden_dim, eps=1e-6):
        super(FusedAddAddRMSNorm, self).__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, residual, sublayer1_out, sublayer2_out):
        """
        Fused add + add + RMSNorm.
        """
        # === FUSED KERNEL ===
        hidden = residual + sublayer1_out + sublayer2_out
        
        variance = hidden.pow(2).mean(dim=-1, keepdim=True)
        output = hidden * torch.rsqrt(variance + self.eps) * self.weight
        # === END FUSED KERNEL ===
        
        return output, hidden


# LayerNorm variant
class FusedAddLayerNorm(nn.Module):
    """
    Fused Add + LayerNorm (for models using LayerNorm instead of RMSNorm).
    """
    def __init__(self, hidden_dim, eps=1e-6):
        super(FusedAddLayerNorm, self).__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
    
    def forward(self, x, residual):
        """
        Fused add + LayerNorm.
        """
        # Add
        hidden = x + residual
        
        # LayerNorm
        mean = hidden.mean(dim=-1, keepdim=True)
        var = hidden.var(dim=-1, keepdim=True, unbiased=False)
        output = (hidden - mean) * torch.rsqrt(var + self.eps)
        output = output * self.weight + self.bias
        
        return output, hidden


# Test parameters
batch_size = 32
seq_len = 2048
hidden_dim = 4096

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_dim)
    residual = torch.randn(batch_size, seq_len, hidden_dim)
    return [x, residual]

def get_init_inputs():
    return [hidden_dim]

