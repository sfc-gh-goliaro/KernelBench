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
    Fused Dequantize + Attention (Quantized KV-Cache Attention).
    
    Performs attention with 4-bit or 8-bit quantized KV cache:
    1. Store K, V in quantized format (INT4/INT8/FP8)
    2. During attention, dequantize KV on-the-fly
    3. Compute scaled dot-product attention
    
    This reduces KV cache memory by 2-4x while maintaining accuracy
    through on-the-fly dequantization.
    
    Reference: SGLang quantized KV cache, KIVI, Atom
    """
    def __init__(self, num_heads, head_dim, max_seq_len, num_kv_heads=None,
                 kv_bits=8, kv_group_size=128):
        """
        :param num_heads: Number of query heads
        :param head_dim: Head dimension
        :param max_seq_len: Maximum sequence length
        :param num_kv_heads: Number of KV heads (for GQA)
        :param kv_bits: KV quantization bits (4 or 8)
        :param kv_group_size: Group size for quantization
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        self.max_seq_len = max_seq_len
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.scale = head_dim ** -0.5
        
        # Quantized KV cache
        if kv_bits == 4:
            # 4-bit: 2 values per byte
            k_cache_size = (1, self.num_kv_heads, max_seq_len, head_dim // 2)
            self.register_buffer('k_cache_int4', torch.zeros(*k_cache_size, dtype=torch.uint8))
            self.register_buffer('v_cache_int4', torch.zeros(*k_cache_size, dtype=torch.uint8))
        else:
            # 8-bit: 1 value per byte
            k_cache_size = (1, self.num_kv_heads, max_seq_len, head_dim)
            self.register_buffer('k_cache_int8', torch.zeros(*k_cache_size, dtype=torch.int8))
            self.register_buffer('v_cache_int8', torch.zeros(*k_cache_size, dtype=torch.int8))
        
        # Quantization scales (per-group)
        num_groups = head_dim // kv_group_size
        self.register_buffer('k_scales', torch.ones(1, self.num_kv_heads, max_seq_len, num_groups))
        self.register_buffer('v_scales', torch.ones(1, self.num_kv_heads, max_seq_len, num_groups))
        
        # Zero points for asymmetric quantization
        self.register_buffer('k_zeros', torch.zeros(1, self.num_kv_heads, max_seq_len, num_groups))
        self.register_buffer('v_zeros', torch.zeros(1, self.num_kv_heads, max_seq_len, num_groups))
        
        self.register_buffer('cache_len', torch.tensor(0))
    
    def _quantize_kv(self, k, v, position):
        """Quantize and store K, V at position."""
        batch_size = k.shape[0]
        
        if self.kv_bits == 8:
            # INT8 quantization per-group
            for g in range(self.head_dim // self.kv_group_size):
                g_start = g * self.kv_group_size
                g_end = g_start + self.kv_group_size
                
                k_group = k[..., g_start:g_end]
                v_group = v[..., g_start:g_end]
                
                # Compute scales
                k_absmax = k_group.abs().max(dim=-1, keepdim=True).values
                v_absmax = v_group.abs().max(dim=-1, keepdim=True).values
                
                k_scale = 127.0 / k_absmax.clamp(min=1e-12)
                v_scale = 127.0 / v_absmax.clamp(min=1e-12)
                
                # Quantize
                k_int8 = (k_group * k_scale).round().clamp(-128, 127).to(torch.int8)
                v_int8 = (v_group * v_scale).round().clamp(-128, 127).to(torch.int8)
                
                # Store
                self.k_cache_int8[:batch_size, :, position, g_start:g_end] = k_int8.squeeze(2)
                self.v_cache_int8[:batch_size, :, position, g_start:g_end] = v_int8.squeeze(2)
                self.k_scales[:batch_size, :, position, g:g+1] = 1.0 / k_scale.squeeze(-1)
                self.v_scales[:batch_size, :, position, g:g+1] = 1.0 / v_scale.squeeze(-1)
        
        else:  # 4-bit
            # INT4 quantization (pack 2 values per byte)
            for g in range(self.head_dim // self.kv_group_size):
                g_start = g * self.kv_group_size
                g_end = g_start + self.kv_group_size
                
                k_group = k[..., g_start:g_end]
                v_group = v[..., g_start:g_end]
                
                k_absmax = k_group.abs().max(dim=-1, keepdim=True).values
                v_absmax = v_group.abs().max(dim=-1, keepdim=True).values
                
                k_scale = 7.0 / k_absmax.clamp(min=1e-12)
                v_scale = 7.0 / v_absmax.clamp(min=1e-12)
                
                # Quantize to 4-bit (-8 to 7)
                k_int4 = (k_group * k_scale).round().clamp(-8, 7).to(torch.int8)
                v_int4 = (v_group * v_scale).round().clamp(-8, 7).to(torch.int8)
                
                # Pack (simplified - actual impl packs into uint8)
                self.k_scales[:batch_size, :, position, g:g+1] = 1.0 / k_scale.squeeze(-1)
                self.v_scales[:batch_size, :, position, g:g+1] = 1.0 / v_scale.squeeze(-1)
    
    def _dequantize_kv(self, seq_len):
        """Dequantize K, V from cache."""
        if self.kv_bits == 8:
            k_int8 = self.k_cache_int8[:, :, :seq_len]
            v_int8 = self.v_cache_int8[:, :, :seq_len]
            
            # Dequantize per-group
            k_float = k_int8.float()
            v_float = v_int8.float()
            
            # Apply scales (simplified - assumes aligned groups)
            k = k_float * self.k_scales[:, :, :seq_len].repeat_interleave(self.kv_group_size, dim=-1)
            v = v_float * self.v_scales[:, :, :seq_len].repeat_interleave(self.kv_group_size, dim=-1)
            
            return k, v
        
        else:  # 4-bit (simplified)
            # Would unpack and dequantize
            return None, None
    
    def forward(self, q, k_new, v_new, position=None):
        """
        Fused quantized KV cache attention.
        
        :param q: Query (batch, num_heads, 1, head_dim)
        :param k_new: New key (batch, num_kv_heads, 1, head_dim)
        :param v_new: New value (batch, num_kv_heads, 1, head_dim)
        :param position: Position index
        :return: Attention output
        """
        batch_size = q.shape[0]
        
        if position is None:
            position = self.cache_len.item()
        
        # === FUSED KERNEL START ===
        # Step 1: Quantize and store new K, V
        self._quantize_kv(k_new, v_new, position)
        self.cache_len = torch.tensor(position + 1)
        
        # Step 2: Dequantize full K, V
        k, v = self._dequantize_kv(position + 1)
        
        # Step 3: GQA expansion
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
                batch_size, self.num_heads, -1, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
                batch_size, self.num_heads, -1, self.head_dim)
        
        # Step 4: Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)
        # === FUSED KERNEL END ===
        
        return output


# FP8 KV Cache variant
class FusedFP8KVCacheAttention(nn.Module):
    """
    Fused Attention with FP8 Quantized KV Cache.
    
    Uses FP8 (E4M3) format for KV cache with better accuracy than INT8.
    """
    def __init__(self, num_heads, head_dim, max_seq_len, num_kv_heads=None):
        super(FusedFP8KVCacheAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        self.max_seq_len = max_seq_len
        self.scale = head_dim ** -0.5
        self.fp8_max = 448.0
        
        # FP8 KV cache (stored as float16 to simulate FP8)
        self.register_buffer('k_cache', 
            torch.zeros(1, self.num_kv_heads, max_seq_len, head_dim, dtype=torch.float16))
        self.register_buffer('v_cache',
            torch.zeros(1, self.num_kv_heads, max_seq_len, head_dim, dtype=torch.float16))
        self.register_buffer('k_scales', torch.ones(1, self.num_kv_heads, max_seq_len, 1))
        self.register_buffer('v_scales', torch.ones(1, self.num_kv_heads, max_seq_len, 1))
        self.register_buffer('cache_len', torch.tensor(0))
    
    def forward(self, q, k_new, v_new):
        """FP8 KV cache attention."""
        batch_size = q.shape[0]
        position = self.cache_len.item()
        
        # Quantize new K, V to FP8
        k_absmax = k_new.abs().max(dim=-1, keepdim=True).values
        v_absmax = v_new.abs().max(dim=-1, keepdim=True).values
        
        k_scale = self.fp8_max / k_absmax.clamp(min=1e-12)
        v_scale = self.fp8_max / v_absmax.clamp(min=1e-12)
        
        k_fp8 = (k_new * k_scale).clamp(-self.fp8_max, self.fp8_max).to(torch.float16)
        v_fp8 = (v_new * v_scale).clamp(-self.fp8_max, self.fp8_max).to(torch.float16)
        
        # Store in cache
        self.k_cache[:batch_size, :, position:position+1] = k_fp8
        self.v_cache[:batch_size, :, position:position+1] = v_fp8
        self.k_scales[:batch_size, :, position:position+1] = 1.0 / k_scale
        self.v_scales[:batch_size, :, position:position+1] = 1.0 / v_scale
        self.cache_len = torch.tensor(position + 1)
        
        # Dequantize and compute attention
        k = self.k_cache[:batch_size, :, :position+1].float() * self.k_scales[:batch_size, :, :position+1]
        v = self.v_cache[:batch_size, :, :position+1].float() * self.v_scales[:batch_size, :, :position+1]
        
        # GQA expansion
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(batch_size, self.num_heads, -1, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(batch_size, self.num_heads, -1, self.head_dim)
        
        # Attention
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        return torch.matmul(attn, v)


# Test parameters
batch_size = 8
num_heads = 32
num_kv_heads = 8
head_dim = 128
max_seq_len = 4096

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    q = torch.randn(batch_size, num_heads, 1, head_dim)
    k_new = torch.randn(batch_size, num_kv_heads, 1, head_dim)
    v_new = torch.randn(batch_size, num_kv_heads, 1, head_dim)
    return [q, k_new, v_new]

def get_init_inputs():
    return [num_heads, head_dim, max_seq_len, num_kv_heads]

