import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import math

class Model(nn.Module):
    """
    Llama-3 RoPE
    
    Used by: Llama-3, Llama-3.1, Llama-3.2
    
    Llama-3 specific RoPE with adjusted base frequency (500000) and
    optional scaling factors for extended context.
    
    Shapes:
        Input: (batch_size, seq_len, num_heads, head_dim) for q and k
        Output: (batch_size, seq_len, num_heads, head_dim) for q and k
    """
    
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 500000.0,
                 scaling_factor: float = 1.0, low_freq_factor: float = 1.0,
                 high_freq_factor: float = 4.0, original_max_seq_len: int = 8192):
        """
        Initialize Llama-3 RoPE.
        
        Args:
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length
            base: Base for frequency computation (500000 for Llama-3)
            scaling_factor: Context length scaling factor
            low_freq_factor: Factor for low frequency dimensions
            high_freq_factor: Factor for high frequency dimensions
            original_max_seq_len: Original training max sequence length
        """
        super(Model, self).__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Compute frequencies with Llama-3 specific adjustments
        inv_freq = self._compute_inv_freq(
            head_dim, base, scaling_factor, low_freq_factor, 
            high_freq_factor, original_max_seq_len
        )
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def _compute_inv_freq(self, dim: int, base: float, scaling_factor: float,
                          low_freq_factor: float, high_freq_factor: float,
                          original_max_seq_len: int) -> torch.Tensor:
        """Compute Llama-3 specific inverse frequencies."""
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        if scaling_factor == 1.0:
            return inv_freq
        
        # Llama-3.1+ uses a more complex scaling scheme
        low_freq_wavelen = original_max_seq_len / low_freq_factor
        high_freq_wavelen = original_max_seq_len / high_freq_factor
        
        wavelens = 2 * math.pi / inv_freq
        
        # Compute scaling based on wavelength
        new_inv_freq = []
        for i, (freq, wavelen) in enumerate(zip(inv_freq, wavelens)):
            if wavelen < high_freq_wavelen:
                # High frequency: no scaling
                new_inv_freq.append(freq)
            elif wavelen > low_freq_wavelen:
                # Low frequency: full scaling
                new_inv_freq.append(freq / scaling_factor)
            else:
                # Medium frequency: smooth interpolation
                smooth = (original_max_seq_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                new_inv_freq.append((1 - smooth) * freq / scaling_factor + smooth * freq)
        
        return torch.tensor(new_inv_freq)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor = None) -> tuple:
        """
        Apply Llama-3 RoPE to q and k.
        
        Args:
            q: Query tensor (batch, seq, num_heads, head_dim)
            k: Key tensor (batch, seq, num_heads, head_dim)
            position_ids: Optional position indices
            
        Returns:
            Tuple of rotated (q, k)
        """
        seq_len = q.shape[1]
        
        if position_ids is None:
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)
        else:
            cos = self.cos_cached[position_ids].unsqueeze(2)
            sin = self.sin_cached[position_ids].unsqueeze(2)
        
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        
        return q_rotated, k_rotated
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding."""
        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 8192, "num_heads": 32, "head_dim": 128},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("embeddings", "6_Llama3_RoPE")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["num_heads"], p["head_dim"])
    q = DISTRIBUTIONS["normal"](shape, dtype=dtype, device=device) if dist_name == "indices" else DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    k = DISTRIBUTIONS["normal"](shape, dtype=dtype, device=device) if dist_name == "indices" else DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [q, k]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["head_dim"], p["seq_length"], 500000.0]
