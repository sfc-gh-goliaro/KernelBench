import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Shared Expert MoE (Mixture of Experts with shared experts).
    
    Combines routed experts with always-active shared experts
    for better performance. Used in DeepSeek-V2/V3.
    
    Based on: "DeepSeek-V2: A Strong, Economical, and Efficient MoE LLM"
    """
    def __init__(self, dim, hidden_dim, num_routed_experts=64, 
                 num_shared_experts=2, top_k=6, aux_loss_weight=0.01):
        """
        :param dim: Input/output dimension
        :param hidden_dim: Expert hidden dimension
        :param num_routed_experts: Number of routed (sparse) experts
        :param num_shared_experts: Number of shared (always-active) experts
        :param top_k: Number of routed experts per token
        :param aux_loss_weight: Weight for auxiliary load balancing loss
        """
        super(Model, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        
        # Router for routed experts
        self.router = nn.Linear(dim, num_routed_experts, bias=False)
        
        # Routed experts (SwiGLU-style)
        self.routed_gate = nn.Parameter(
            torch.randn(num_routed_experts, dim, hidden_dim) * 0.02
        )
        self.routed_up = nn.Parameter(
            torch.randn(num_routed_experts, dim, hidden_dim) * 0.02
        )
        self.routed_down = nn.Parameter(
            torch.randn(num_routed_experts, hidden_dim, dim) * 0.02
        )
        
        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim, dim, bias=False)
            )
            for _ in range(num_shared_experts)
        ])
        
        # Normalization coefficient
        self.register_buffer('routed_scaling', 
            torch.tensor(1.0 / (top_k + num_shared_experts)))
    
    def forward(self, x):
        """
        Forward pass with shared + routed experts.
        
        :param x: Input tensor (batch, seq_len, dim)
        :return: Tuple of (output, aux_loss)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute shared expert outputs (always active)
        shared_out = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x)
        
        # Route to sparse experts
        router_logits = self.router(x)  # (batch, seq, num_routed)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute routed expert outputs
        x_flat = x.view(-1, self.dim)
        indices_flat = top_k_indices.view(-1, self.top_k)
        probs_flat = top_k_probs.view(-1, self.top_k)
        
        routed_out = torch.zeros_like(x_flat)
        
        for k in range(self.top_k):
            for e in range(self.num_routed_experts):
                mask = (indices_flat[:, k] == e)
                if not mask.any():
                    continue
                
                expert_input = x_flat[mask]
                
                # SwiGLU computation
                gate = F.silu(F.linear(expert_input, self.routed_gate[e].t()))
                up = F.linear(expert_input, self.routed_up[e].t())
                hidden = gate * up
                expert_output = F.linear(hidden, self.routed_down[e].t())
                
                # Weight by routing probability
                weight = probs_flat[mask, k:k+1]
                routed_out[mask] = routed_out[mask] + weight * expert_output
        
        routed_out = routed_out.view(batch_size, seq_len, self.dim)
        
        # Combine shared and routed outputs
        output = shared_out + routed_out
        
        # Compute auxiliary load balancing loss
        expert_counts = torch.zeros(self.num_routed_experts, device=x.device)
        for k in range(self.top_k):
            for e in range(self.num_routed_experts):
                expert_counts[e] += (indices_flat[:, k] == e).float().sum()
        
        expert_fraction = expert_counts / (batch_size * seq_len * self.top_k)
        router_prob_mean = router_probs.view(-1, self.num_routed_experts).mean(dim=0)
        aux_loss = (expert_fraction * router_prob_mean).sum() * self.num_routed_experts
        aux_loss = aux_loss * self.aux_loss_weight
        
        return output, aux_loss


# Test parameters
batch_size = 8
seq_len = 256
dim = 2048
hidden_dim = 1408  # DeepSeek intermediate size
num_routed_experts = 64
num_shared_experts = 2
top_k = 6

def get_inputs():
    return [torch.randn(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, hidden_dim, num_routed_experts, num_shared_experts, top_k]

