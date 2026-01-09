import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 32768, base: float = 1000000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    q_rotated = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
    k_rotated = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)
    return q_rotated, k_rotated


class MixtralAttention(nn.Module):
    """Mixtral uses sliding window attention."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, 
                 head_dim: int, max_seq_len: int, sliding_window: int = 4096):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.sliding_window = sliding_window
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len)
        self.scale = head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Sliding window causal mask
        mask = torch.ones(seq_len, seq_len, device=x.device)
        mask = torch.triu(mask, diagonal=1)
        mask += torch.tril(torch.ones_like(mask), diagonal=-self.sliding_window)
        mask = mask.bool()
        attn = attn.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)


class MixtralExpert(nn.Module):
    """Single expert MLP (SwiGLU)."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MixtralMoE(nn.Module):
    """Mixtral Mixture-of-Experts with Top-2 routing."""
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            MixtralExpert(hidden_size, intermediate_size) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        # Router
        router_logits = self.gate(x_flat)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Compute expert outputs
        output = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
            
            token_indices = expert_mask.nonzero(as_tuple=True)[0]
            expert_input = x_flat[token_indices]
            expert_output = self.experts[expert_idx](expert_input)
            
            # Get weights for this expert
            weights = torch.zeros(token_indices.shape[0], device=x.device)
            for k in range(self.top_k):
                mask_k = selected_experts[token_indices, k] == expert_idx
                weights[mask_k] = routing_weights[token_indices[mask_k], k]
            
            output[token_indices] += weights.unsqueeze(-1) * expert_output
        
        return output.view(batch_size, seq_len, hidden_size)


class Model(nn.Module):
    """
    Mixtral MoE Decoder Block
    
    The core repeated block in Mixtral-style MoE transformers.
    Used by: Mixtral-8x7B, Mixtral-8x22B
    
    Architecture:
        x -> RMSNorm -> Sliding Window Attention -> + residual
          -> RMSNorm -> Sparse MoE (Top-2 of 8 experts) -> + residual
    """
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 intermediate_size: int, num_experts: int = 8, top_k: int = 2,
                 max_seq_len: int = 32768):
        super().__init__()
        self.head_dim = hidden_size // num_heads
        
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = MixtralAttention(
            hidden_size, num_heads, num_kv_heads, self.head_dim, max_seq_len
        )
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.block_sparse_moe = MixtralMoE(hidden_size, intermediate_size, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x
        
        # MoE with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.block_sparse_moe(x)
        x = residual + x
        
        return x


# Benchmark configuration (Mixtral-8x7B dimensions)
batch_size = 4
seq_len = 2048
hidden_size = 4096
num_heads = 32
num_kv_heads = 8
intermediate_size = 14336
num_experts = 8
top_k = 2

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size)]

def get_init_inputs():
    return [hidden_size, num_heads, num_kv_heads, intermediate_size, num_experts, top_k]

