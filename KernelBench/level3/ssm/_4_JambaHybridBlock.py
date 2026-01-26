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


class MambaMixer(nn.Module):
    """Simplified Mamba mixer for hybrid architecture."""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                 padding=d_conv - 1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_inner = F.silu(x_conv)
        
        # Simplified SSM (sequential for correctness)
        x_dbl = self.x_proj(x_inner)
        dt = F.softplus(self.dt_proj(x_dbl[..., :1])).squeeze(-1)
        B = x_dbl[..., 1:self.d_state + 1]
        C = x_dbl[..., self.d_state + 1:]
        A = -torch.exp(self.A_log)
        
        # Sequential scan
        state = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device)
        outputs = []
        for t in range(seq_len):
            deltaA = torch.exp(dt[:, t].unsqueeze(-1) * A)
            deltaB_u = dt[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1) * x_inner[:, t].unsqueeze(-1)
            state = deltaA * state + deltaB_u
            y_t = (state * C[:, t].unsqueeze(1)).sum(-1) + x_inner[:, t] * self.D
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)
        y = y * F.silu(z)
        return self.out_proj(y)


class JambaAttention(nn.Module):
    """Standard attention for Jamba hybrid layers."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)


class JambaMLP(nn.Module):
    """SwiGLU MLP for Jamba."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class JambaMoE(nn.Module):
    """Jamba MoE layer."""
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int = 16, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            JambaMLP(hidden_size, intermediate_size) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        router_logits = self.gate(x_flat)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        output = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
            token_indices = expert_mask.nonzero(as_tuple=True)[0]
            expert_output = self.experts[expert_idx](x_flat[token_indices])
            
            weights = torch.zeros(token_indices.shape[0], device=x.device)
            for k in range(self.top_k):
                mask_k = selected_experts[token_indices, k] == expert_idx
                weights[mask_k] = routing_weights[token_indices[mask_k], k]
            output[token_indices] += weights.unsqueeze(-1) * expert_output
        
        return output.view(batch_size, seq_len, hidden_size)


class Model(nn.Module):
    """
    Jamba Hybrid Block
    
    The core repeated block in Jamba architecture (Mamba + Attention + MoE hybrid).
    Used by: Jamba-1.5-Mini, Jamba-1.5-Large
    
    Architecture alternates between:
    - Mamba layers (for long-range efficiency)
    - Attention layers (for high-quality token mixing)
    - MoE on selected layers (for capacity)
    
    This block represents a Mamba layer with MoE.
    """
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 intermediate_size: int, d_state: int = 16, use_mamba: bool = True,
                 use_moe: bool = True, num_experts: int = 16, top_k: int = 2):
        super().__init__()
        self.use_mamba = use_mamba
        self.use_moe = use_moe
        
        self.input_layernorm = RMSNorm(hidden_size)
        
        if use_mamba:
            self.mixer = MambaMixer(hidden_size, d_state=d_state)
        else:
            self.mixer = JambaAttention(hidden_size, num_heads, num_kv_heads)
        
        self.pre_moe_layernorm = RMSNorm(hidden_size)
        
        if use_moe:
            self.mlp = JambaMoE(hidden_size, intermediate_size, num_experts, top_k)
        else:
            self.mlp = JambaMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mixer (Mamba or Attention) with residual
        residual = x
        x = self.input_layernorm(x)
        x = self.mixer(x)
        x = residual + x
        
        # MLP/MoE with residual
        residual = x
        x = self.pre_moe_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


# Benchmark configuration (Jamba-1.5-Mini dimensions)
batch_size = 4
seq_len = 1024
hidden_size = 4096
num_heads = 32
num_kv_heads = 8
intermediate_size = 14336
d_state = 16
num_experts = 16
top_k = 2

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size)]

def get_init_inputs():
    return [hidden_size, num_heads, num_kv_heads, intermediate_size, d_state, True, True, num_experts, top_k]

