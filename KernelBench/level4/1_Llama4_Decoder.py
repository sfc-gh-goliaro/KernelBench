"""
Llama-4 Decoder-Only Transformer

A modern decoder-only transformer implementing Llama-4 style architecture with:
- RMSNorm normalization
- Grouped-Query Attention (GQA) with optional QK normalization
- Rotary Position Embeddings (RoPE) with NoPE layer support
- SwiGLU MLP (gate/up projection fused)
- SharedFusedMoE on interleaved layers
- Chunked Local Attention for efficiency

Reference dimensions (Llama-4-Scout style):
- Hidden: 5120, Heads: 40, KV Heads: 8, FFN: 13824, Layers: 48
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
    """Rotary Position Embedding (RoPE)."""
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class QKNorm(nn.Module):
    """Query-Key Normalization (Llama-4 feature)."""
    def __init__(self, head_dim: int):
        super().__init__()
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q_norm(q), self.k_norm(k)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention with optional QK normalization and RoPE."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        use_qk_norm: bool = True,
        use_rope: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        self.use_qk_norm = use_qk_norm
        self.use_rope = use_rope
        if use_qk_norm:
            self.qk_norm = QKNorm(self.head_dim)
        if use_rope:
            self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK Normalization
        if self.use_qk_norm:
            q, k = self.qk_norm(q, k)

        # RoPE
        if self.use_rope:
            cos, sin = self.rotary_emb(x, seq_len)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV heads for GQA
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP with fused gate/up projection."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class SharedExpertMoE(nn.Module):
    """Mixture of Experts with shared expert (Llama-4 style)."""
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Router
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList([
            SwiGLUMLP(hidden_size, intermediate_size) for _ in range(num_experts)
        ])

        # Shared expert
        self.shared_expert = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)

        # Router logits and top-k selection
        router_logits = self.router(x_flat)
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1), self.num_experts_per_tok, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Expert computation
        expert_output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (selected_experts == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x_flat[mask])
                weight = routing_weights[mask, (selected_experts[mask] == i).float().argmax(dim=-1)]
                expert_output[mask] += weight.unsqueeze(-1) * expert_out

        # Add shared expert
        shared_out = self.shared_expert(x_flat)
        expert_output = expert_output + shared_out

        return expert_output.view(batch_size, seq_len, hidden_size)


class Llama4DecoderLayer(nn.Module):
    """Single Llama-4 decoder layer."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        use_moe: bool = False,
        num_experts: int = 8,
        use_rope: bool = True,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = GroupedQueryAttention(
            hidden_size, num_heads, num_kv_heads, use_rope=use_rope
        )
        self.post_attention_layernorm = RMSNorm(hidden_size)

        if use_moe:
            self.mlp = SharedExpertMoE(hidden_size, intermediate_size, num_experts)
        else:
            self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask)
        x = residual + x

        # MLP with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class Model(nn.Module):
    """Llama-4 style decoder-only transformer."""
    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 5120,
        num_layers: int = 48,
        num_heads: int = 40,
        num_kv_heads: int = 8,
        intermediate_size: int = 13824,
        num_experts: int = 8,
        moe_layer_freq: int = 4,  # MoE every N layers
        nope_layer_freq: int = 8,  # NoPE every N layers
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            use_moe = (i % moe_layer_freq == 0) and (i > 0)
            use_rope = (i % nope_layer_freq != 0)  # NoPE on some layers
            self.layers.append(Llama4DecoderLayer(
                hidden_size, num_heads, num_kv_heads, intermediate_size,
                use_moe=use_moe, num_experts=num_experts, use_rope=use_rope
            ))

        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


# Configuration
batch_size = 2
sequence_length = 512
vocab_size = 128256
hidden_size = 4096  # Reduced for benchmarking
num_layers = 8  # Reduced for benchmarking
num_heads = 32
num_kv_heads = 8
intermediate_size = 11008


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
        'moe_layer_freq': 4,
        'nope_layer_freq': 8,
    }]

