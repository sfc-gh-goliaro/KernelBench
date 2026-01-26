"""
RWKV-6 Linear Attention Model

Implements RWKV-6 architecture:
- Linear-complexity receptance-weighted key-value attention
- Time mixing and channel mixing blocks
- Data-dependent time decay

Variants from Table 5:
- RWKV-6-1.6B: 24 layers, d_model=2048
- RWKV-6-3B: 32 layers, d_model=2560
- RWKV-6-7B: 32 layers, d_model=4096

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any

# Import level1 operators (used directly - no wrapping needed)
from ..level1.normalization._6_LayerNorm import Model as LayerNorm
from ..level1.activations._7_Swish import Model as Swish
from ..level1.activations._3_Sigmoid import Model as Sigmoid
from ..level1.activations._5_Softmax import Model as Softmax
from ..level1.matmul._1_MatMul import Model as MatMul


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "1.6B": "RWKV/v6-Finch-1B6-HF",
    "3B": "RWKV/v6-Finch-3B-HF",
    "7B": "RWKV/v6-Finch-7B-HF",
}


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class RWKV6TimeMix(nn.Module):
    """RWKV-6 Time Mixing (attention substitute) using level1 operators."""
    def __init__(self, d_model: int, head_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.n_head = d_model // head_size
        
        self.time_maa_x = nn.Parameter(torch.zeros(1, 1, d_model))
        self.time_maa_w = nn.Parameter(torch.zeros(1, 1, d_model))
        self.time_maa_k = nn.Parameter(torch.zeros(1, 1, d_model))
        self.time_maa_v = nn.Parameter(torch.zeros(1, 1, d_model))
        self.time_maa_r = nn.Parameter(torch.zeros(1, 1, d_model))
        self.time_maa_g = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self.time_decay = nn.Parameter(torch.zeros(self.n_head, head_size))
        self.time_first = nn.Parameter(torch.zeros(self.n_head, head_size))
        
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        
        self.ln_x = nn.GroupNorm(self.n_head, d_model, eps=1e-5)
        
        self.sigmoid = Sigmoid()
        self.swish = Swish()

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        batch, seq_len, _ = x.shape
        
        # Time shift mixing
        xx = F.pad(x, (0, 0, 1, -1)) - x
        
        xxx = x + xx * self.time_maa_x
        xk = x + xx * self.time_maa_k
        xv = x + xx * self.time_maa_v
        xr = x + xx * self.time_maa_r
        xg = x + xx * self.time_maa_g
        
        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = self.swish(self.gate(xg))
        
        # Reshape for multi-head
        r = r.view(batch, seq_len, self.n_head, self.head_size)
        k = k.view(batch, seq_len, self.n_head, self.head_size)
        v = v.view(batch, seq_len, self.n_head, self.head_size)
        
        # Apply sigmoid to receptance
        r = self.sigmoid(r)
        
        # RWKV linear attention (simplified)
        w = torch.exp(-torch.exp(self.time_decay))
        
        # Compute weighted key-value
        wkv = torch.zeros(batch, seq_len, self.n_head, self.head_size, device=x.device, dtype=x.dtype)
        
        a = torch.zeros(batch, self.n_head, self.head_size, self.head_size, device=x.device, dtype=x.dtype)
        b = torch.zeros(batch, self.n_head, self.head_size, device=x.device, dtype=x.dtype)
        
        for t in range(seq_len):
            kt = k[:, t]  # (batch, n_head, head_size)
            vt = v[:, t]
            
            # Update recurrence
            kv = kt.unsqueeze(-1) * vt.unsqueeze(-2)
            a = w.unsqueeze(0).unsqueeze(-1) * a + kv
            b = w.unsqueeze(0) * b + kt
            
            # Apply time_first bonus for current token
            bonus = self.time_first.unsqueeze(0) * kt
            wkv[:, t] = ((a + bonus.unsqueeze(-1) * vt.unsqueeze(-2)).sum(-2) / 
                        (b + bonus + 1e-9))
        
        x = r * wkv
        x = x.view(batch, seq_len, -1)
        x = self.ln_x(x.transpose(1, 2)).transpose(1, 2)
        
        return self.output(x * g)


class RWKV6ChannelMix(nn.Module):
    """RWKV-6 Channel Mixing (FFN) using level1 operators."""
    def __init__(self, d_model: int, intermediate_size: Optional[int] = None):
        super().__init__()
        intermediate = intermediate_size or d_model * 4
        
        self.time_maa_k = nn.Parameter(torch.zeros(1, 1, d_model))
        self.time_maa_r = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self.key = nn.Linear(d_model, intermediate, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(intermediate, d_model, bias=False)
        
        self.sigmoid = Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = F.pad(x, (0, 0, 1, -1)) - x
        
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r
        
        k = self.key(xk)
        k = torch.relu(k) ** 2
        
        r = self.sigmoid(self.receptance(xr))
        
        return r * self.value(k)


class RWKV6Block(nn.Module):
    """RWKV-6 block using level1 operators."""
    def __init__(self, d_model: int, head_size: int = 64):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.time_mix = RWKV6TimeMix(d_model, head_size)
        self.channel_mix = RWKV6ChannelMix(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    RWKV-6 linear attention language model.
    
    Uses level1 operators from KernelBench:
    - LayerNorm from level1/normalization/6_LayerNorm
    - Swish (used directly)
    - Sigmoid from level1/activations/3_Sigmoid
    - MatMul from level1/matmul/1_MatMul
    """
    
    def __init__(
        self,
        d_model: int = 2048,
        num_layers: int = 24,
        vocab_size: int = 65536,
        head_size: int = 64,
        **kwargs  # Accept and ignore extra kwargs for flexibility
    ):
        super().__init__()
        
        # Store config values
        self.d_model = d_model
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.head_size = head_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.ln_in = LayerNorm(d_model)
        
        self.blocks = nn.ModuleList([
            RWKV6Block(d_model, head_size) for _ in range(num_layers)
        ])
        
        self.ln_out = LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.ln_in(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_out(x)
        return self.head(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 4
sequence_length = 1024
d_model = 2048
num_layers = 8
vocab_size = 65536
head_size = 64


def get_inputs():
    return [torch.randint(0, vocab_size, (batch_size, sequence_length))]


def get_init_inputs():
    return [{
        'd_model': d_model,
        'num_layers': num_layers,
        'vocab_size': vocab_size,
        'head_size': head_size,
    }]
