import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Top-K Expert Router
    
    Used by: Mixtral, DeepSeek-MoE, Qwen-MoE, DBRX
    
    Expert routing via softmax over expert logits + top-k selection.
    Returns indices of selected experts and their routing weights.
    
    Shapes:
        Input: (batch_size, seq_len, hidden_size)
        Output: (indices: (batch*seq, k), weights: (batch*seq, k))
    """
    
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        """
        Initialize top-k router.
        
        Args:
            hidden_size: Input hidden dimension
            num_experts: Total number of experts
            top_k: Number of experts to route to per token
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Route tokens to top-k experts.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            
        Returns:
            Tuple of (expert_indices, routing_weights):
                expert_indices: (batch*seq, top_k) selected expert IDs
                routing_weights: (batch*seq, top_k) normalized weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Flatten batch and sequence
        x_flat = x.view(-1, self.hidden_size)
        
        # Compute routing scores
        router_logits = self.gate(x_flat)  # (batch*seq, num_experts)
        
        # Get top-k experts
        routing_weights, expert_indices = torch.topk(router_logits, self.top_k, dim=-1)
        
        # Normalize weights with softmax over selected experts
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        return expert_indices, routing_weights


# ============================================================================
# Benchmark Configuration
# ============================================================================
