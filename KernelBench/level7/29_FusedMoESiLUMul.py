import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused MoE + SiLU + Mul (SwiGLU Expert Fusion).
    
    Combines the complete SwiGLU-based MoE expert computation:
    1. Route tokens to experts
    2. For each expert: gate_proj and up_proj
    3. Fused SiLU(gate) * up
    4. down_proj
    5. Combine expert outputs
    
    The SiLU + Mul fusion is critical as it's the bottleneck in SwiGLU experts.
    
    Reference: Mixtral, DeepSeek-MoE, vLLM fused MoE
    """
    def __init__(self, hidden_dim, expert_dim, num_experts, top_k=2):
        """
        :param hidden_dim: Input/output hidden dimension
        :param expert_dim: Expert intermediate dimension
        :param num_experts: Number of experts
        :param top_k: Number of experts per token
        """
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        
        # Expert weights: packed gate+up for efficiency
        self.expert_gate_up = nn.Parameter(
            torch.randn(num_experts, hidden_dim, 2 * expert_dim) * 0.02
        )
        self.expert_down = nn.Parameter(
            torch.randn(num_experts, expert_dim, hidden_dim) * 0.02
        )
    
    def _fused_expert_forward(self, tokens, expert_idx):
        """
        Fused SwiGLU expert computation.
        
        :param tokens: Input tokens for this expert
        :param expert_idx: Expert index
        :return: Expert output
        """
        # === FUSED SiLU + MUL KERNEL ===
        # Single matmul for gate and up
        gate_up = F.linear(tokens, self.expert_gate_up[expert_idx].t())
        
        # Split
        gate, up = gate_up.chunk(2, dim=-1)
        
        # Fused SiLU * Mul (this is the hot path)
        hidden = F.silu(gate) * up
        
        # Down projection
        output = F.linear(hidden, self.expert_down[expert_idx].t())
        # === END FUSED KERNEL ===
        
        return output
    
    def forward(self, x):
        """
        Fused MoE with SwiGLU experts.
        
        :param x: Input (batch * seq, hidden_dim)
        :return: Output (batch * seq, hidden_dim)
        """
        num_tokens = x.shape[0]
        device = x.device
        
        # Routing
        logits = self.gate(x)
        routing_weights, expert_indices = torch.topk(logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Initialize output
        output = torch.zeros(num_tokens, self.hidden_dim, device=device)
        
        # Process each top-k selection
        for k in range(self.top_k):
            for e in range(self.num_experts):
                # Get tokens for this expert
                mask = (expert_indices[:, k] == e)
                if not mask.any():
                    continue
                
                expert_tokens = x[mask]
                expert_weights = routing_weights[mask, k]
                
                # Fused expert forward
                expert_out = self._fused_expert_forward(expert_tokens, e)
                
                # Weighted accumulation
                output[mask] += expert_weights.unsqueeze(-1) * expert_out
        
        return output


# Grouped version for efficiency
class FusedGroupedMoESwiGLU(nn.Module):
    """
    Grouped MoE with fused SwiGLU using sorted tokens.
    
    More efficient: sorts tokens by expert and uses grouped GEMM.
    """
    def __init__(self, hidden_dim, expert_dim, num_experts, top_k=2):
        super(FusedGroupedMoESwiGLU, self).__init__()
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        
        # Packed gate+up weights
        self.expert_gate_up = nn.Parameter(
            torch.randn(num_experts, hidden_dim, 2 * expert_dim) * 0.02
        )
        self.expert_down = nn.Parameter(
            torch.randn(num_experts, expert_dim, hidden_dim) * 0.02
        )
    
    def forward(self, x):
        """Grouped MoE forward."""
        num_tokens = x.shape[0]
        device = x.device
        
        # Routing
        logits = self.gate(x)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        # Flatten assignments
        flat_weights = weights.view(-1)
        flat_indices = indices.view(-1)
        token_indices = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        
        # Sort by expert for grouped processing
        sorted_order = flat_indices.argsort()
        sorted_experts = flat_indices[sorted_order]
        sorted_tokens = token_indices[sorted_order]
        sorted_weights = flat_weights[sorted_order]
        
        # Compute expert boundaries
        expert_counts = torch.bincount(sorted_experts, minlength=self.num_experts)
        expert_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                                    expert_counts.cumsum(0)])
        
        # Initialize output
        output = torch.zeros(num_tokens, self.hidden_dim, device=device)
        
        # Process each expert's batch
        for e in range(self.num_experts):
            start, end = expert_offsets[e].item(), expert_offsets[e + 1].item()
            if end <= start:
                continue
            
            # Get this expert's tokens
            expert_token_ids = sorted_tokens[start:end]
            expert_weights_batch = sorted_weights[start:end]
            expert_inputs = x[expert_token_ids]
            
            # === FUSED SwiGLU ===
            gate_up = F.linear(expert_inputs, self.expert_gate_up[e].t())
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = F.silu(gate) * up
            expert_out = F.linear(hidden, self.expert_down[e].t())
            # === END FUSED ===
            
            # Scatter-add weighted outputs
            output.index_add_(0, expert_token_ids, 
                             expert_weights_batch.unsqueeze(-1) * expert_out)
        
        return output


# With GeGLU variant
class FusedMoEGeGLU(nn.Module):
    """
    Fused MoE with GeGLU (GELU-gated) experts.
    """
    def __init__(self, hidden_dim, expert_dim, num_experts, top_k=2):
        super(FusedMoEGeGLU, self).__init__()
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.expert_gate_up = nn.Parameter(
            torch.randn(num_experts, hidden_dim, 2 * expert_dim) * 0.02
        )
        self.expert_down = nn.Parameter(
            torch.randn(num_experts, expert_dim, hidden_dim) * 0.02
        )
    
    def forward(self, x):
        """MoE with GeGLU experts."""
        num_tokens = x.shape[0]
        device = x.device
        
        logits = self.gate(x)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        output = torch.zeros(num_tokens, self.hidden_dim, device=device)
        
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = (indices[:, k] == e)
                if not mask.any():
                    continue
                
                tokens = x[mask]
                
                # Fused GeGLU
                gate_up = F.linear(tokens, self.expert_gate_up[e].t())
                gate, up = gate_up.chunk(2, dim=-1)
                hidden = F.gelu(gate) * up  # GELU instead of SiLU
                expert_out = F.linear(hidden, self.expert_down[e].t())
                
                output[mask] += weights[mask, k].unsqueeze(-1) * expert_out
        
        return output


# Test parameters
batch_size = 32
seq_len = 512
hidden_dim = 4096
expert_dim = 11008
num_experts = 8
top_k = 2

def get_inputs():
    return [torch.randn(batch_size * seq_len, hidden_dim)]

def get_init_inputs():
    return [hidden_dim, expert_dim, num_experts, top_k]

