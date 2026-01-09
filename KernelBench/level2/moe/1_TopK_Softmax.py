import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused TopK + Softmax for MoE Routing
    
    Used by: Mixtral, DeepSeek, all MoE models
    
    Fuses the top-k selection with softmax normalization in MoE routing.
    This is more efficient than separate topk and softmax operations.
    
    Found in: vLLM (topk_softmax_kernels.cu), SGLang, Megatron-Core
    
    Shapes:
        Input: (num_tokens, num_experts) router logits
        Output: (num_tokens, top_k) routing weights and expert indices
    """
    
    def __init__(self, num_experts: int, top_k: int = 2, renormalize: bool = True):
        """
        Initialize fused TopK + Softmax.
        
        Args:
            num_experts: Number of experts
            top_k: Number of experts per token
            renormalize: Whether to renormalize weights after top-k
        """
        super(Model, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize = renormalize
    
    def forward(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fused TopK + Softmax routing.
        
        Args:
            router_logits: Router output (num_tokens, num_experts)
            
        Returns:
            Tuple of (routing_weights, expert_indices)
            - routing_weights: (num_tokens, top_k)
            - expert_indices: (num_tokens, top_k)
        """
        # Fused: get top-k logits and indices
        top_k_logits, expert_indices = torch.topk(router_logits, self.top_k, dim=-1)
        
        # Softmax on selected logits only (more efficient than full softmax)
        if self.renormalize:
            routing_weights = F.softmax(top_k_logits, dim=-1)
        else:
            # Softmax over all, then gather (less common)
            full_probs = F.softmax(router_logits, dim=-1)
            routing_weights = torch.gather(full_probs, -1, expert_indices)
        
        return routing_weights, expert_indices


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
num_experts = 64  # DeepSeek-V3 has 256 experts
top_k = 8

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    num_tokens = batch_size * seq_length
    router_logits = torch.randn(num_tokens, num_experts, device='cuda')
    return [router_logits]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [num_experts, top_k]

