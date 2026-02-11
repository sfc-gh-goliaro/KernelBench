import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMLP(nn.Module):
    """Single expert MLP with SwiGLU activation.
    
    Weight names (w1, w2, w3) match HuggingFace Mixtral checkpoint structure.
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)  # gate
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)  # down
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)  # up
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: silu(gate) * up -> down
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Model(nn.Module):
    """
    Fused MoE (Mixture of Experts)
    
    Used by: Mixtral, Qwen3-Omni-MoE
    
    Fused MoE kernel computing all expert FFNs in single kernel
    with efficient memory access. This is a reference implementation;
    production uses optimized Triton/CUDA kernels.
    
    Supports two expert weight formats:
    - expert_format="module_list" (default): nn.ModuleList of ExpertMLP modules
      with separate w1, w2, w3 per expert. Matches HuggingFace Mixtral checkpoint
      structure (e.g., block_sparse_moe.experts.{i}.w1.weight).
    - expert_format="stacked_fused": Stacked nn.Parameter tensors with fused
      gate_up_proj (num_experts, 2*intermediate, hidden) and down_proj
      (num_experts, hidden, intermediate). Matches Qwen-MoE checkpoint structure.
    
    Supports two routing strategies:
    - norm_topk_prob=True (default): softmax -> topk -> renormalize by sum
    - norm_topk_prob=False: softmax -> topk (no renormalization)
    
    Shapes:
        Input: (batch_size, seq_len, hidden_size)
        Output: (batch_size, seq_len, hidden_size)
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int,
                 top_k: int = 2, expert_format: str = "module_list",
                 norm_topk_prob: bool = True):
        """
        Initialize fused MoE.
        
        Args:
            hidden_size: Model hidden dimension
            intermediate_size: FFN intermediate dimension
            num_experts: Number of experts
            top_k: Number of experts per token
            expert_format: "module_list" for Mixtral-style per-expert modules,
                           "stacked_fused" for Qwen-MoE-style stacked parameters
            norm_topk_prob: If True, renormalize top-k weights by their sum
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_format = expert_format
        self.norm_topk_prob = norm_topk_prob
        
        # Router gate
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Expert weights
        if expert_format == "module_list":
            # Mixtral: nn.ModuleList of ExpertMLP modules
            self.experts = nn.ModuleList([
                ExpertMLP(hidden_size, intermediate_size)
                for _ in range(num_experts)
            ])
        elif expert_format == "stacked_fused":
            # Qwen-MoE: stacked parameters with fused gate+up projection
            self.gate_up_proj = nn.Parameter(
                torch.empty(num_experts, 2 * intermediate_size, hidden_size)
            )
            self.down_proj = nn.Parameter(
                torch.empty(num_experts, hidden_size, intermediate_size)
            )
        else:
            raise ValueError(f"Unknown expert_format: {expert_format}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fused MoE forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            
        Returns:
            Output tensor (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        input_dtype = x.dtype
        x_flat = x.view(-1, self.hidden_size)
        
        # Router: softmax -> topk
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        
        # Select top-k experts
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize weights
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(input_dtype)
        
        # Initialize output
        final_hidden_states = torch.zeros_like(x_flat)
        
        # Build expert mask for efficient dispatching
        with torch.no_grad():
            expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, top_k, num_tokens)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        
        # Process each active expert
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx >= self.num_experts:
                continue
            
            # Find tokens routed to this expert and their positions in top_k
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = x_flat[token_idx]
            
            # Process through expert MLP
            if self.expert_format == "module_list":
                current_hidden_states = self.experts[expert_idx.item()](current_state)
            else:
                # stacked_fused: fused gate_up_proj + down_proj
                gate, up = (current_state @ self.gate_up_proj[expert_idx].T).chunk(2, dim=-1)
                current_hidden_states = F.silu(gate) * up
                current_hidden_states = current_hidden_states @ self.down_proj[expert_idx].T
            
            # Apply expert weights
            weighted = current_hidden_states * topk_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, weighted.to(final_hidden_states.dtype))
        
        return final_hidden_states.view(batch_size, seq_len, self.hidden_size)
