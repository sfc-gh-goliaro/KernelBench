"""
Gated Linear Attention (GLA) Model

Implements GLA architecture:
- Gated linear attention mechanism
- Sub-quadratic complexity
- Hardware-efficient training

Variants from Table 5:
- GLA-1B: d_model=2048, num_layers=24
- GLA-3B: d_model=3072, num_layers=32

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

# Import level1 operators
# Operators that need wrapping
from ..level1.normalization._4_RMSNorm import Model as RMSNormL1
# Operators used directly
from ..level1.activations._7_Swish import Model as Swish
from ..level1.activations._3_Sigmoid import Model as Sigmoid
from ..level1.matmul._1_MatMul import Model as MatMul


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "1B": "fla-hub/gla-1B-100B",
    "3B": "fla-hub/gla-3B-100B",
}


# ============================================================================
# Wrapper classes for level1 operators that need adaptation
# ============================================================================

class RMSNorm(nn.Module):
    """RMS normalization with learnable weight, using level1 operator."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self._rms_norm = RMSNormL1(dim, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self._rms_norm(x.transpose(1, -1)).transpose(1, -1)
        return normalized * self.weight


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class GatedLinearAttention(nn.Module):
    """Gated Linear Attention mechanism using level1 operators."""
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        expand_k: float = 1.0,
        expand_v: float = 2.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        
        self.q_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.g_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)
        
        self.gk_proj = nn.Linear(d_model, num_heads)
        
        self.swish = Swish()
        self.sigmoid = Sigmoid()
        self.matmul = MatMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.swish(self.g_proj(x))
        
        # Gate for keys
        gk = self.sigmoid(self.gk_proj(x))
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_k_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_k_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_v_dim)
        
        # Linear attention with gating
        q = q.transpose(1, 2)  # (B, H, L, Dk)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)  # (B, H, L, Dv)
        gk = gk.transpose(1, 2).unsqueeze(-1)  # (B, H, L, 1)
        
        # Apply key gating for data-dependent decay
        k = k * gk
        
        # Causal linear attention via cumulative sum
        kv = self.matmul(k.transpose(-2, -1), v)  # (B, H, Dk, Dv)
        
        # Recurrent form for causal linear attention
        output = torch.zeros_like(v)
        state = torch.zeros(batch_size, self.num_heads, self.head_k_dim, self.head_v_dim, 
                           device=x.device, dtype=x.dtype)
        
        for t in range(seq_len):
            kt = k[:, :, t:t+1, :]  # (B, H, 1, Dk)
            vt = v[:, :, t:t+1, :]  # (B, H, 1, Dv)
            qt = q[:, :, t:t+1, :]  # (B, H, 1, Dk)
            
            state = state + kt.transpose(-2, -1) @ vt
            output[:, :, t:t+1, :] = qt @ state
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = output * g
        
        return self.o_proj(output)


class GLAMLP(nn.Module):
    """GLA-style MLP using level1 operators."""
    def __init__(self, d_model: int, intermediate_size: Optional[int] = None):
        super().__init__()
        intermediate = intermediate_size or d_model * 4
        self.gate_proj = nn.Linear(d_model, intermediate * 2, bias=False)
        self.down_proj = nn.Linear(intermediate, d_model, bias=False)
        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(self.swish(gate) * up)


class GLABlock(nn.Module):
    """GLA block using level1 operators."""
    def __init__(self, d_model: int, num_heads: int, expand_k: float = 1.0, expand_v: float = 2.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GatedLinearAttention(d_model, num_heads, expand_k, expand_v)
        self.norm2 = RMSNorm(d_model)
        self.mlp = GLAMLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Gated Linear Attention language model.
    
    Uses level1 operators from KernelBench:
    - RMSNorm from level1/normalization/4_RMSNorm
    - Swish (used directly)
    - Sigmoid from level1/activations/3_Sigmoid
    - MatMul from level1/matmul/1_MatMul
    """
    
    def __init__(
        self,
        d_model: int = 2048,
        num_layers: int = 24,
        vocab_size: int = 50304,
        num_heads: int = 8,
        expand_k: float = 1.0,
        expand_v: float = 2.0,
        **kwargs  # Accept and ignore extra kwargs for flexibility
    ):
        super().__init__()
        
        # Store config values
        self.d_model = d_model
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.blocks = nn.ModuleList([
            GLABlock(d_model, num_heads, expand_k, expand_v)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return self.head(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 4
sequence_length = 1024
d_model = 2048
num_layers = 8
vocab_size = 50304
num_heads = 8


def get_inputs():
    return [torch.randint(0, vocab_size, (batch_size, sequence_length))]


def get_init_inputs():
    return [{
        'd_model': d_model,
        'num_layers': num_layers,
        'vocab_size': vocab_size,
        'num_heads': num_heads,
    }]
