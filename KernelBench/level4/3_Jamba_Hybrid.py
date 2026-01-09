"""
Jamba Hybrid Model (Mamba + Attention + MoE)

A hybrid architecture combining:
- Mamba-2 State Space Duality (SSD) layers
- Standard Transformer attention layers
- Mixture-of-Experts on selected layers
- Alternating Mamba and attention in a configurable pattern

Reference: Jamba-1.5-Mini
- Hidden: 4096, Heads: 32, KV Heads: 8, FFN: 14336, Layers: 32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""
    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention for Jamba attention layers."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class MambaMixer(nn.Module):
    """
    Mamba-2 State Space Duality (SSD) Mixer.
    
    Implements selective scan with:
    - Input projection to expand dimension
    - 1D convolution for local context
    - Selective scan (S6) with data-dependent A, B, C, D
    - Output projection
    """
    def __init__(
        self,
        hidden_size: int,
        state_size: int = 16,
        conv_kernel_size: int = 4,
        expand_factor: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.expand_factor = expand_factor
        self.inner_size = hidden_size * expand_factor
        self.num_heads = num_heads
        self.head_dim = self.inner_size // num_heads

        # Input projection: x -> (z, x_proj)
        self.in_proj = nn.Linear(hidden_size, self.inner_size * 2, bias=False)

        # 1D convolution
        self.conv1d = nn.Conv1d(
            self.inner_size, self.inner_size,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            groups=self.inner_size
        )

        # SSM parameters projection
        # Project to: dt, B, C (data-dependent)
        self.x_proj = nn.Linear(self.inner_size, num_heads * (1 + state_size * 2), bias=False)

        # Learnable A (log scale for stability)
        self.A_log = nn.Parameter(torch.randn(num_heads, state_size))

        # D (skip connection)
        self.D = nn.Parameter(torch.ones(num_heads))

        # dt projection
        self.dt_proj = nn.Linear(num_heads, num_heads, bias=True)

        # Output projection
        self.out_proj = nn.Linear(self.inner_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Input projection
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        # Convolution
        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        # SSM parameters
        ssm_params = self.x_proj(x_conv)
        ssm_params = ssm_params.view(batch_size, seq_len, self.num_heads, -1)

        dt = ssm_params[..., 0]  # (B, L, H)
        B = ssm_params[..., 1:1+self.state_size]  # (B, L, H, N)
        C = ssm_params[..., 1+self.state_size:]  # (B, L, H, N)

        # dt projection and softplus
        dt = self.dt_proj(dt)
        dt = F.softplus(dt)

        # A from log scale
        A = -torch.exp(self.A_log)  # (H, N)

        # Discretize: A_bar = exp(dt * A)
        A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, H, N)

        # Reshape x for SSM
        x_ssm = x_conv.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Selective scan (simplified sequential implementation)
        h = torch.zeros(batch_size, self.num_heads, self.state_size, device=x.device)
        outputs = []
        for t in range(seq_len):
            # State update: h = A_bar * h + B * x
            h = A_bar[:, t] * h + B[:, t] * x_ssm[:, t, :, 0:1]
            # Output: y = C * h + D * x
            y = (C[:, t] * h).sum(-1) + self.D * x_ssm[:, t, :, 0]
            outputs.append(y)

        y = torch.stack(outputs, dim=1)  # (B, L, H)
        y = y.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        y = y.reshape(batch_size, seq_len, self.inner_size)

        # Gate and output
        y = y * F.silu(z)
        return self.out_proj(y)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class JambaMoE(nn.Module):
    """MoE layer for Jamba."""
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int = 16, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUMLP(hidden_size, intermediate_size) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)

        router_logits = self.router(x_flat)
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1), self.top_k, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (selected_experts == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x_flat[mask])
                idx = torch.where(selected_experts[mask] == i)
                weight = routing_weights[mask][idx[0], idx[1]]
                output[mask] += weight.unsqueeze(-1) * expert_out

        return output.view(batch_size, seq_len, hidden_size)


class JambaLayer(nn.Module):
    """Jamba layer - can be Mamba or Attention based."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        use_mamba: bool = True,
        use_moe: bool = False,
        num_experts: int = 16,
    ):
        super().__init__()
        self.use_mamba = use_mamba
        self.input_layernorm = RMSNorm(hidden_size)

        if use_mamba:
            self.mixer = MambaMixer(hidden_size)
        else:
            self.mixer = GroupedQueryAttention(hidden_size, num_heads, num_kv_heads)

        self.post_mixer_layernorm = RMSNorm(hidden_size)

        if use_moe:
            self.mlp = JambaMoE(hidden_size, intermediate_size, num_experts)
        else:
            self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.mixer(x)
        x = residual + x

        residual = x
        x = self.post_mixer_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class Model(nn.Module):
    """Jamba Hybrid model."""
    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        intermediate_size: int = 14336,
        num_experts: int = 16,
        mamba_ratio: int = 7,  # Mamba:Attention ratio (7:1)
        moe_layer_freq: int = 2,  # MoE every N layers
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Alternating Mamba and Attention (7 Mamba : 1 Attention pattern)
            use_mamba = (i % (mamba_ratio + 1)) != mamba_ratio
            use_moe = (i % moe_layer_freq == 0) and (i > 0)
            self.layers.append(JambaLayer(
                hidden_size, num_heads, num_kv_heads, intermediate_size,
                use_mamba=use_mamba, use_moe=use_moe, num_experts=num_experts
            ))

        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.lm_head(x)


# Configuration (reduced for benchmarking)
batch_size = 2
sequence_length = 512
vocab_size = 65536
hidden_size = 2048
num_layers = 8
num_heads = 16
num_kv_heads = 4
intermediate_size = 5504


def get_inputs():
    return [torch.randint(0, vocab_size, (batch_size, sequence_length))]


def get_init_inputs():
    return [{
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'intermediate_size': intermediate_size,
        'num_experts': 8,
        'mamba_ratio': 3,
        'moe_layer_freq': 2,
    }]

