import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ExpertMLP(nn.Module):
    """Single expert MLP with SwiGLU activation.
    
    Used as individual experts in the MoE layer.
    Weight names match HuggingFace DeepSeek checkpoint structure.
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SharedExpertMLP(nn.Module):
    """Shared expert MLP with SwiGLU activation.
    
    Always activated for all tokens. Can represent multiple shared experts
    by using a larger intermediate size (n_shared_experts * moe_intermediate_size).
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Model(nn.Module):
    """
    Shared Expert Fused MoE
    
    Used by: DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, Llama 4
    
    Mixture-of-Experts with shared experts that are always activated.
    Combines sparse expert computation with dense shared experts.
    Output = SharedExperts(x) + Σ(gate_i * Expert_i(x))
    
    Supports:
    - Multiple shared experts (combined into single larger MLP)
    - group_limited_greedy routing (for DeepSeek-V2)
    - routed_scaling_factor for expert weight scaling
    
    Shapes:
        Input: (batch, seq, hidden_size) or (num_tokens, hidden_size)
        Output: same shape as input
    """
    
    def __init__(
        self,
        hidden_size: int = 7168,
        intermediate_size: int = 2048,
        num_experts: int = 64,
        top_k: int = 6,
        n_shared_experts: int = 2,
        routed_scaling_factor: float = 1.0,
        topk_method: str = "greedy",
        n_group: int = 1,
        topk_group: int = 1,
    ):
        """
        Initialize Shared Fused MoE.
        
        Args:
            hidden_size: Input/output hidden dimension
            intermediate_size: Expert intermediate dimension (moe_intermediate_size)
            num_experts: Number of routed experts (n_routed_experts)
            top_k: Number of experts to select per token (num_experts_per_tok)
            n_shared_experts: Number of shared experts (combined into one larger MLP)
            routed_scaling_factor: Scaling factor for routed expert weights
            topk_method: Routing method - "greedy" or "group_limited_greedy"
            n_group: Number of expert groups for group_limited_greedy
            topk_group: Number of groups to select in group_limited_greedy
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_shared_experts = n_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        
        # Router gate for sparse experts
        # Named 'gate' to match DeepSeek HF checkpoint structure (mlp.gate.weight)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Routed experts as ModuleList to match HF checkpoint structure
        # (mlp.experts.0.gate_proj.weight, mlp.experts.0.up_proj.weight, etc.)
        self.experts = nn.ModuleList([
            ExpertMLP(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        
        # Shared experts (combined into single larger MLP)
        # Named 'shared_experts' to match HF checkpoint structure
        if n_shared_experts is not None and n_shared_experts > 0:
            shared_intermediate_size = intermediate_size * n_shared_experts
            self.shared_experts = SharedExpertMLP(hidden_size, shared_intermediate_size)
        else:
            self.shared_experts = None
    
    def route_tokens_to_experts(
        self,
        router_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to top-k experts.
        
        Supports two routing methods:
        - greedy: Simple top-k selection
        - group_limited_greedy: Select top groups first, then top-k within those groups
        
        Args:
            router_logits: (num_tokens, num_experts) raw routing scores
            
        Returns:
            topk_idx: (num_tokens, top_k) selected expert indices
            topk_weight: (num_tokens, top_k) expert weights (after scaling)
        """
        num_tokens = router_logits.shape[0]
        router_probs = router_logits.softmax(dim=-1, dtype=torch.float32)
        
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                router_probs, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "group_limited_greedy":
            # Group experts and select top groups first
            # Each group has (num_experts // n_group) experts
            group_scores = router_probs.view(num_tokens, self.n_group, -1).max(dim=-1).values
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            
            # Create mask for selected groups
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            
            # Expand mask to expert level
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
                .reshape(num_tokens, -1)
            )
            
            # Zero out experts from non-selected groups and select top-k
            tmp_scores = router_probs.masked_fill(~score_mask.bool(), 0.0)
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
        else:
            # Default to greedy
            topk_weight, topk_idx = torch.topk(
                router_probs, k=self.top_k, dim=-1, sorted=False
            )
        
        # Apply routing scaling factor
        topk_weight = topk_weight * self.routed_scaling_factor
        
        return topk_idx, topk_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Shared + Sparse MoE.
        
        Args:
            x: Input tensor (batch, seq, hidden_size) or (num_tokens, hidden_size)
            
        Returns:
            Output tensor with same shape as input
        """
        # Handle both 2D and 3D inputs
        input_shape = x.shape
        if len(input_shape) == 3:
            batch_size, seq_len, hidden_size = input_shape
            x_flat = x.view(-1, hidden_size)
        else:
            x_flat = x
            batch_size, seq_len = None, None
        
        num_tokens = x_flat.shape[0]
        residual = x_flat
        
        # Compute router logits (in float32 for numerical stability)
        router_logits = F.linear(
            x_flat.float(),
            self.gate.weight.float()
        )
        
        # Route tokens to experts
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        
        # Initialize output
        final_hidden_states = torch.zeros_like(x_flat)
        
        # Build expert mask for efficient dispatching
        with torch.no_grad():
            expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, top_k, num_tokens)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        
        # Process each active expert
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0].item()
            if expert_idx >= self.num_experts:
                continue
            
            # Find tokens routed to this expert and their positions in top_k
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = x_flat[token_idx]
            
            # Process through expert MLP
            current_hidden_states = self.experts[expert_idx](current_state)
            
            # Apply expert weights (in float32 for precision)
            weighted = current_hidden_states.float() * topk_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, weighted.to(final_hidden_states.dtype))
        
        # Add shared expert output (if shared experts exist)
        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(residual)
        
        # Reshape back to original shape if needed
        if batch_size is not None:
            return final_hidden_states.view(batch_size, seq_len, -1)
        return final_hidden_states


# ============================================================================
# Benchmark Configuration
# ============================================================================
