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
    Fused Attention + Output Quantization (FP8).
    
    Combines attention computation with FP8 quantization of the output:
    1. Compute scaled dot-product attention
    2. Apply output projection
    3. Quantize output to FP8 for next layer
    
    This enables FP8 inference pipelines where activations are quantized
    between layers.
    
    Reference: vLLM FP8, TensorRT-LLM, H100 FP8 inference
    """
    def __init__(self, dim, num_heads, num_kv_heads=None, head_dim=None,
                 quant_output=True, quant_dtype='fp8_e4m3'):
        """
        :param dim: Model dimension
        :param num_heads: Number of query heads
        :param num_kv_heads: Number of KV heads (for GQA)
        :param head_dim: Head dimension
        :param quant_output: Whether to quantize output
        :param quant_dtype: 'fp8_e4m3' or 'fp8_e5m2'
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.quant_output = quant_output
        self.quant_dtype = quant_dtype
        
        # Projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
    
    def _quantize_fp8(self, x, dtype='fp8_e4m3'):
        """Quantize tensor to FP8."""
        if dtype == 'fp8_e4m3':
            max_val = 448.0
        else:
            max_val = 57344.0
        
        absmax = x.abs().max()
        scale = max_val / absmax.clamp(min=1e-12)
        
        x_quant = (x * scale).clamp(-max_val, max_val)
        
        return x_quant.to(torch.float16), 1.0 / scale
    
    def forward(self, x, attention_mask=None, kv_cache=None):
        """
        Fused attention + output quantization.
        
        :param x: Input (batch, seq, dim)
        :param attention_mask: Optional attention mask
        :param kv_cache: Optional KV cache
        :return: Tuple of (output, scale) if quant_output else output
        """
        batch_size, seq_len, _ = x.shape
        
        # === FUSED KERNEL START ===
        # QKV projection
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Handle KV cache
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        
        # GQA expansion
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
                batch_size, self.num_heads, -1, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
                batch_size, self.num_heads, -1, self.head_dim)
        
        # Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Causal mask
        seq_k = k.shape[2]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_k, device=x.device, dtype=torch.bool),
            diagonal=seq_k - seq_len + 1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)
        
        # Reshape and output projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(attn_out)
        
        # Quantize output
        if self.quant_output:
            output, scale = self._quantize_fp8(output, self.quant_dtype)
            return output, scale
        # === FUSED KERNEL END ===
        
        return output


# FlashAttention-style with fused output quant
class FusedFlashAttentionOutputQuant(nn.Module):
    """
    Flash Attention with fused FP8 output quantization.
    
    Memory-efficient attention with inline quantization.
    """
    def __init__(self, dim, num_heads, head_dim=None, quant_dtype='fp8_e4m3'):
        super(FusedFlashAttentionOutputQuant, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.quant_dtype = quant_dtype
        
        self.qkv_proj = nn.Linear(dim, 3 * num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Fused QKV
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention (would be FlashAttention in practice)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Reshape and output
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(attn)
        
        # Fused FP8 quantization
        absmax = output.abs().max()
        max_val = 448.0 if 'e4m3' in self.quant_dtype else 57344.0
        scale = max_val / absmax.clamp(min=1e-12)
        output_quant = (output * scale).clamp(-max_val, max_val).to(torch.float16)
        
        return output_quant, 1.0 / scale


# Test parameters
batch_size = 8
seq_len = 2048
dim = 4096
num_heads = 32
num_kv_heads = 8

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
    return [dim, num_heads, num_kv_heads]

