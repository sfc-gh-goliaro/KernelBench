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


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) from DeepSeek-V2/V3.
    
    Key innovation: Low-rank compression of KV cache.
    Instead of storing full K,V, we store compressed latent vectors
    and reconstruct K,V on-the-fly during attention.
    """
    def __init__(self, hidden_size: int, num_heads: int, qk_nope_head_dim: int,
                 qk_rope_head_dim: int, v_head_dim: int, kv_lora_rank: int,
                 max_seq_len: int = 8192):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim  # Non-RoPE Q/K dim
        self.qk_rope_head_dim = qk_rope_head_dim  # RoPE Q/K dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        
        # Q projection (full)
        self.q_proj = nn.Linear(hidden_size, num_heads * self.qk_head_dim, bias=False)
        self.q_norm = RMSNorm(self.qk_head_dim)
        
        # KV compression (low-rank)
        self.kv_down_proj = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        self.kv_up_proj = nn.Linear(kv_lora_rank, num_heads * (self.qk_head_dim + v_head_dim), bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank)
        
        # Output projection
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)
        
        # RoPE for the rope portion of Q/K
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, qk_rope_head_dim, 2).float() / qk_rope_head_dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
        
        self.scale = self.qk_head_dim ** -0.5

    def apply_rope(self, x, cos, sin):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Q projection
        q = self.q_proj(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        q = self.q_norm(q)
        q = q.transpose(1, 2)
        
        # KV low-rank decompression
        kv_compressed = self.kv_down_proj(x)
        kv_compressed = self.kv_norm(kv_compressed)
        kv = self.kv_up_proj(kv_compressed)
        kv = kv.view(batch_size, seq_len, self.num_heads, self.qk_head_dim + self.v_head_dim)
        k = kv[..., :self.qk_head_dim].transpose(1, 2)
        v = kv[..., self.qk_head_dim:].transpose(1, 2)
        
        # Apply RoPE to the rope portion only
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        
        q_nope, q_rope = q[..., :self.qk_nope_head_dim], q[..., self.qk_nope_head_dim:]
        k_nope, k_rope = k[..., :self.qk_nope_head_dim], k[..., self.qk_nope_head_dim:]
        
        q_rope = self.apply_rope(q_rope, cos, sin)
        k_rope = self.apply_rope(k_rope, cos, sin)
        
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)


class DeepSeekMoE(nn.Module):
    """DeepSeek MoE with shared expert."""
    def __init__(self, hidden_size: int, intermediate_size: int, 
                 num_experts: int = 64, num_shared_experts: int = 2, top_k: int = 6):
        super().__init__()
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        
        # Router
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Routed experts (simplified - real impl uses grouped weights)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size, bias=False),
                nn.SiLU(),
                nn.Linear(intermediate_size, hidden_size, bias=False)
            ) for _ in range(num_experts)
        ])
        
        # Shared experts (always activated)
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size, bias=False),
                nn.SiLU(),
                nn.Linear(intermediate_size, hidden_size, bias=False)
            ) for _ in range(num_shared_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        # Shared expert output (always computed)
        shared_output = sum(expert(x_flat) for expert in self.shared_experts)
        
        # Router
        router_logits = self.gate(x_flat)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Routed expert output
        routed_output = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
            
            token_indices = expert_mask.nonzero(as_tuple=True)[0]
            expert_input = x_flat[token_indices]
            expert_output = self.experts[expert_idx](expert_input)
            
            weights = torch.zeros(token_indices.shape[0], device=x.device)
            for k in range(self.top_k):
                mask_k = selected_experts[token_indices, k] == expert_idx
                weights[mask_k] = routing_weights[token_indices[mask_k], k]
            
            routed_output[token_indices] += weights.unsqueeze(-1) * expert_output
        
        output = shared_output + routed_output
        return output.view(batch_size, seq_len, hidden_size)


class Model(nn.Module):
    """
    DeepSeek-V2 Decoder Block
    
    The core repeated block in DeepSeek-V2/V3 transformers.
    
    Key innovations:
    - Multi-Head Latent Attention (MLA) for KV cache compression
    - Shared experts that are always activated
    - Fine-grained expert routing
    """
    def __init__(self, hidden_size: int, num_heads: int, qk_nope_head_dim: int,
                 qk_rope_head_dim: int, v_head_dim: int, kv_lora_rank: int,
                 intermediate_size: int, num_experts: int = 64, 
                 num_shared_experts: int = 2, top_k: int = 6):
        super().__init__()
        
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = MultiHeadLatentAttention(
            hidden_size, num_heads, qk_nope_head_dim, qk_rope_head_dim,
            v_head_dim, kv_lora_rank
        )
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = DeepSeekMoE(hidden_size, intermediate_size, num_experts, 
                                num_shared_experts, top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x
        
        # MoE with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


# Benchmark configuration (DeepSeek-V2-Lite dimensions, simplified)
batch_size = 4
seq_len = 1024
hidden_size = 2048
num_heads = 16
qk_nope_head_dim = 64
qk_rope_head_dim = 64
v_head_dim = 128
kv_lora_rank = 512
intermediate_size = 1408
num_experts = 16
num_shared_experts = 2
top_k = 6

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size)]

def get_init_inputs():
    return [hidden_size, num_heads, qk_nope_head_dim, qk_rope_head_dim, 
            v_head_dim, kv_lora_rank, intermediate_size, num_experts, 
            num_shared_experts, top_k]

