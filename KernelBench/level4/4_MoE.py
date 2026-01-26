"""
Mixture-of-Experts Models

Implements DeepSeek-V2-Lite and Mixtral MoE architectures:
- DeepSeek: Multi-head Latent Attention (MLA) + shared experts
- Mixtral: Standard GQA + top-k expert routing

Variants from Table 5:
- DeepSeek-V2-Lite: MLA with kv_lora_rank=512, 64 experts, 6 active
- Mixtral-8x7B: 8 experts, 2 active, sliding_window=4096

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple

# Import level1 operators
# Operators that need wrapping (different interface in level4)
from ..level1.normalization._4_RMSNorm import Model as RMSNormL1
from ..level1.embeddings._1_RotaryEmbedding import Model as RotaryEmbeddingL1
# Operators used directly (no wrapping needed)
from ..level1.activations._7_Swish import Model as Swish
from ..level1.activations._5_Softmax import Model as Softmax
from ..level1.matmul._1_MatMul import Model as MatMul
from ..level1.moe._1_TopK_Router import Model as TopKRouter


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "DeepSeek-V2-Lite": "deepseek-ai/DeepSeek-V2-Lite",
    "Mixtral-8x7B": "mistralai/Mixtral-8x7B-v0.1",
}


# ============================================================================
# Wrapper classes for level1 operators that need adaptation
# ============================================================================

class RMSNorm(nn.Module):
    """RMS Normalization with learnable weight, using level1 operator."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self._rms_norm = RMSNormL1(hidden_size, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self._rms_norm(x.transpose(1, -1)).transpose(1, -1)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding using level1 operator (handles shape transpose)."""
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self._rope = RotaryEmbeddingL1(dim, max_seq_len, base)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        q_reshaped = q.transpose(1, 2)
        k_reshaped = k.transpose(1, 2)
        q_rotated, k_rotated = self._rope(q_reshaped, k_reshaped)
        return q_rotated.transpose(1, 2), k_rotated.transpose(1, 2)


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class SwiGLUMLP(nn.Module):
    """SwiGLU MLP using level1 operators."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swish = Swish()

    def forward(self, x):
        return self.down_proj(self.swish(self.gate_proj(x)) * self.up_proj(x))


class MultiHeadLatentAttention(nn.Module):
    """MLA from DeepSeek-V2/V3 with KV compression using level1 operators."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_nope_dim: int = 128,
        qk_rope_dim: int = 64,
        kv_lora_rank: int = 512,
        v_head_dim: int = 128,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_dim = qk_nope_dim
        self.qk_rope_dim = qk_rope_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.head_dim = qk_nope_dim + qk_rope_dim

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.kv_a_proj = nn.Linear(hidden_size, kv_lora_rank + qk_rope_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank)
        self.kv_b_proj = nn.Linear(kv_lora_rank, num_heads * (qk_nope_dim + v_head_dim), bias=False)
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(qk_rope_dim)
        self.matmul = MatMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q_nope, q_rope = q.split([self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        kv_a = self.kv_a_proj(x)
        kv_a, k_rope = kv_a.split([self.kv_lora_rank, self.qk_rope_dim], dim=-1)
        kv_a = self.kv_a_layernorm(kv_a)
        kv_b = self.kv_b_proj(kv_a)
        kv_b = kv_b.view(batch_size, seq_len, self.num_heads, self.qk_nope_dim + self.v_head_dim).transpose(1, 2)
        k_nope, v = kv_b.split([self.qk_nope_dim, self.v_head_dim], dim=-1)

        k_rope = k_rope.view(batch_size, seq_len, 1, self.qk_rope_dim).transpose(1, 2)
        k_rope = k_rope.expand(-1, self.num_heads, -1, -1)
        
        # Apply RoPE using level1 operator
        q_rope, k_rope = self.rotary_emb(q_rope, k_rope)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = self.matmul(q, k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = self.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class GroupedQueryAttention(nn.Module):
    """Standard GQA for Mixtral using level1 operators."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int, 
                 max_seq_len: int = 8192, rope_theta: float = 10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len, rope_theta)
        self.matmul = MatMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary_emb(q, k)

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = self.matmul(q, k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = self.matmul(attn_weights, v)

        return self.o_proj(attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))


class MoELayer(nn.Module):
    """Mixture of Experts layer using level1 operators."""
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        num_shared_experts: int = 0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts

        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUMLP(hidden_size, intermediate_size) for _ in range(num_experts)
        ])
        
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                SwiGLUMLP(hidden_size, intermediate_size) for _ in range(num_shared_experts)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)

        router_logits = self.router(x_flat)
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1), self.num_experts_per_tok, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        expert_output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (selected_experts == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x_flat[mask])
                indices = torch.where(selected_experts[mask] == i)
                weight = routing_weights[mask][indices[0], indices[1]]
                expert_output[mask] += weight.unsqueeze(-1) * expert_out

        if self.num_shared_experts > 0:
            for shared_expert in self.shared_experts:
                expert_output = expert_output + shared_expert(x_flat)

        return expert_output.view(batch_size, seq_len, hidden_size)


class DeepSeekMoEBlock(nn.Module):
    """DeepSeek-style decoder block with MLA + MoE using level1 operators."""
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int,
                 qk_nope_dim: int, qk_rope_dim: int, kv_lora_rank: int, v_head_dim: int,
                 num_experts: int, num_experts_per_tok: int, num_shared_experts: int):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = MultiHeadLatentAttention(
            hidden_size, num_heads, qk_nope_dim, qk_rope_dim, kv_lora_rank, v_head_dim
        )
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = MoELayer(hidden_size, intermediate_size, num_experts, 
                           num_experts_per_tok, num_shared_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.self_attn(self.input_layernorm(x))
        x = residual + x
        
        residual = x
        x = self.mlp(self.post_attention_layernorm(x))
        x = residual + x
        return x


class MixtralMoEBlock(nn.Module):
    """Mixtral-style decoder block with GQA + MoE using level1 operators."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int,
                 intermediate_size: int, num_experts: int, num_experts_per_tok: int,
                 max_seq_len: int = 8192, rope_theta: float = 1000000.0):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = GroupedQueryAttention(
            hidden_size, num_heads, num_kv_heads, head_dim, max_seq_len, rope_theta
        )
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = MoELayer(hidden_size, intermediate_size, num_experts, num_experts_per_tok, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.self_attn(self.input_layernorm(x))
        x = residual + x
        
        residual = x
        x = self.mlp(self.post_attention_layernorm(x))
        x = residual + x
        return x


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Mixture of Experts model supporting DeepSeek and Mixtral architectures.
    
    Uses level1 operators from KernelBench:
    - RMSNorm from level1/normalization/4_RMSNorm
    - RotaryEmbedding (wrapped - handles shape transpose)
    - Swish (used directly)
    - MatMul (used directly)
    - TopKRouter from level1/moe/1_TopK_Router
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 14336,
        max_seq_len: int = 4096,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        num_shared_experts: int = 0,
        use_mla: bool = False,
        kv_lora_rank: int = 512,
        qk_nope_dim: int = 128,
        qk_rope_dim: int = 64,
        v_head_dim: int = 128,
        rope_theta: float = 1000000.0,
        **kwargs  # Accept and ignore extra kwargs for flexibility
    ):
        super().__init__()
        
        # Store config values
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if use_mla:
                layer = DeepSeekMoEBlock(
                    hidden_size, num_heads, intermediate_size,
                    qk_nope_dim, qk_rope_dim, kv_lora_rank, v_head_dim,
                    num_experts, num_experts_per_tok, num_shared_experts
                )
            else:
                layer = MixtralMoEBlock(
                    hidden_size, num_heads, num_kv_heads, head_dim,
                    intermediate_size, num_experts, num_experts_per_tok,
                    max_seq_len, rope_theta
                )
            self.layers.append(layer)
        
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 2
sequence_length = 512
vocab_size = 32000
hidden_size = 4096
num_layers = 6
num_heads = 32
num_kv_heads = 8
head_dim = 128
intermediate_size = 14336
num_experts = 8
num_experts_per_tok = 2


def get_inputs():
    return [torch.randint(0, vocab_size, (batch_size, sequence_length))]


def get_init_inputs():
    return [{
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'head_dim': head_dim,
        'intermediate_size': intermediate_size,
        'num_experts': num_experts,
        'num_experts_per_tok': num_experts_per_tok,
    }]
