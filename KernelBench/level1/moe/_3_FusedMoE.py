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
    
    Used by: Mixtral, DeepSeek-V2/V3 (vLLM/SGLang)
    
    Fused MoE kernel computing all expert FFNs in single kernel
    with efficient memory access. This is a reference implementation;
    production uses optimized Triton/CUDA kernels.
    
    Uses nn.ModuleList for experts to match HuggingFace checkpoint structure
    (e.g., block_sparse_moe.experts.{i}.w1.weight).
    
    Shapes:
        Input: (batch_size, seq_len, hidden_size)
        Output: (batch_size, seq_len, hidden_size)
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int, top_k: int = 2):
        """
        Initialize fused MoE.
        
        Args:
            hidden_size: Model hidden dimension
            intermediate_size: FFN intermediate dimension
            num_experts: Number of experts
            top_k: Number of experts per token
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router gate
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Experts as ModuleList to match HF checkpoint structure
        self.experts = nn.ModuleList([
            ExpertMLP(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fused MoE forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            
        Returns:
            Output tensor (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.hidden_size)
        
        # Router
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        
        # Select top-k experts
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(x_flat.dtype)
        
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
            
            # Apply expert weights
            weighted = current_hidden_states * topk_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, weighted.to(final_hidden_states.dtype))
        
        return final_hidden_states.view(batch_size, seq_len, self.hidden_size)
