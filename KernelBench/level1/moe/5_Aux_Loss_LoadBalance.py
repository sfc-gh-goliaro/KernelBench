import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Auxiliary Load Balancing Loss for MoE
    
    Used by: All MoE training
    
    Auxiliary loss encouraging uniform expert utilization during training.
    Penalizes uneven distribution of tokens across experts.
    
    Shapes:
        router_logits: (batch*seq, num_experts)
        Output: scalar loss
    """
    
    def __init__(self, num_experts: int, aux_loss_coef: float = 0.01):
        """
        Initialize auxiliary loss.
        
        Args:
            num_experts: Number of experts
            aux_loss_coef: Coefficient for auxiliary loss
        """
        super(Model, self).__init__()
        self.num_experts = num_experts
        self.aux_loss_coef = aux_loss_coef
    
    def forward(self, router_logits: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.
        
        Args:
            router_logits: Router logits (batch*seq, num_experts)
            expert_indices: Selected experts (batch*seq, top_k)
            
        Returns:
            Auxiliary loss scalar
        """
        num_tokens = router_logits.shape[0]
        
        # Compute routing probabilities
        routing_probs = F.softmax(router_logits, dim=-1)  # (num_tokens, num_experts)
        
        # f_i: fraction of tokens routed to each expert
        # Count how many times each expert is selected
        expert_counts = torch.zeros(self.num_experts, device=router_logits.device)
        for expert_idx in range(self.num_experts):
            expert_counts[expert_idx] = (expert_indices == expert_idx).sum().float()
        
        # Normalize by total selections (num_tokens * top_k)
        total_selections = expert_indices.numel()
        f = expert_counts / total_selections
        
        # P_i: mean routing probability for each expert
        P = routing_probs.mean(dim=0)
        
        # Load balancing loss: num_experts * sum(f_i * P_i)
        # This encourages f and P to be uniform (1/num_experts each)
        aux_loss = self.num_experts * (f * P).sum()
        
        return self.aux_loss_coef * aux_loss


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
num_experts = 8
top_k = 2

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    num_tokens = batch_size * seq_length
    router_logits = torch.randn(num_tokens, num_experts, device='cuda')
    expert_indices = torch.randint(0, num_experts, (num_tokens, top_k), device='cuda')
    return [router_logits, expert_indices]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [num_experts]

