import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import math

class Model(nn.Module):
    """
    3D Rotary Position Embedding
    
    Used by: Qwen2-VL (M-RoPE)
    """
    
    def __init__(self, head_dim: int, max_temporal: int = 64, max_height: int = 64, 
                 max_width: int = 64, base: float = 10000.0):
        super(Model, self).__init__()
        self.head_dim = head_dim
        self.max_temporal = max_temporal
        self.max_height = max_height
        self.max_width = max_width
        
        assert head_dim % 3 == 0, "head_dim must be divisible by 3 for 3D RoPE"
        self.dim_t = self.dim_h = self.dim_w = head_dim // 3
        
        inv_freq_t = 1.0 / (base ** (torch.arange(0, self.dim_t, 2).float() / self.dim_t))
        inv_freq_h = 1.0 / (base ** (torch.arange(0, self.dim_h, 2).float() / self.dim_h))
        inv_freq_w = 1.0 / (base ** (torch.arange(0, self.dim_w, 2).float() / self.dim_w))
        
        self.register_buffer('inv_freq_t', inv_freq_t)
        self.register_buffer('inv_freq_h', inv_freq_h)
        self.register_buffer('inv_freq_w', inv_freq_w)
    
    def _compute_rope(self, positions: torch.Tensor, inv_freq: torch.Tensor) -> tuple:
        freqs = torch.outer(positions.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
    
    def _apply_rotary_1d(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        dim = x.shape[-1]
        x1 = x[..., :dim // 2]
        x2 = x[..., dim // 2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, 
                position_ids_t: torch.Tensor, position_ids_h: torch.Tensor, 
                position_ids_w: torch.Tensor) -> tuple:
        batch_size, seq_len = q.shape[:2]
        
        q_t = q[..., :self.dim_t]
        q_h = q[..., self.dim_t:self.dim_t + self.dim_h]
        q_w = q[..., self.dim_t + self.dim_h:]
        
        k_t = k[..., :self.dim_t]
        k_h = k[..., self.dim_t:self.dim_t + self.dim_h]
        k_w = k[..., self.dim_t + self.dim_h:]
        
        cos_t, sin_t = self._compute_rope(position_ids_t.view(-1), self.inv_freq_t)
        cos_t = cos_t.view(batch_size, seq_len, 1, -1)
        sin_t = sin_t.view(batch_size, seq_len, 1, -1)
        q_t = self._apply_rotary_1d(q_t, cos_t, sin_t)
        k_t = self._apply_rotary_1d(k_t, cos_t, sin_t)
        
        cos_h, sin_h = self._compute_rope(position_ids_h.view(-1), self.inv_freq_h)
        cos_h = cos_h.view(batch_size, seq_len, 1, -1)
        sin_h = sin_h.view(batch_size, seq_len, 1, -1)
        q_h = self._apply_rotary_1d(q_h, cos_h, sin_h)
        k_h = self._apply_rotary_1d(k_h, cos_h, sin_h)
        
        cos_w, sin_w = self._compute_rope(position_ids_w.view(-1), self.inv_freq_w)
        cos_w = cos_w.view(batch_size, seq_len, 1, -1)
        sin_w = sin_w.view(batch_size, seq_len, 1, -1)
        q_w = self._apply_rotary_1d(q_w, cos_w, sin_w)
        k_w = self._apply_rotary_1d(k_w, cos_w, sin_w)
        
        q = torch.cat([q_t, q_h, q_w], dim=-1)
        k = torch.cat([k_t, k_h, k_w], dim=-1)
        
        return q, k


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 576, "num_heads": 32, "head_dim": 96},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("vision", "6_RoPE_3D")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    q = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["num_heads"], p["head_dim"]), dtype=dtype, device=device)
    k = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["num_heads"], p["head_dim"]), dtype=dtype, device=device)
    
    h = w = int(math.sqrt(p["seq_length"]))
    position_ids_t = torch.zeros(p["batch_size"], p["seq_length"], dtype=torch.long, device=device)
    position_ids_h = torch.arange(h, device=device).repeat_interleave(w).unsqueeze(0).expand(p["batch_size"], -1)
    position_ids_w = torch.arange(w, device=device).repeat(h).unsqueeze(0).expand(p["batch_size"], -1)
    
    return [q, k, position_ids_t, position_ids_h, position_ids_w]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["head_dim"]]
