import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Expert Dispatch
    
    Used by: All MoE architectures
    
    Permute/gather tokens to route them to their assigned experts
    based on router output. Groups tokens by expert for efficient
    batched expert computation.
    
    Shapes:
        Input: (batch*seq, hidden_size), (batch*seq, top_k) expert indices
        Output: List of (num_tokens_for_expert, hidden_size) per expert
    """
    
    def __init__(self, num_experts: int, top_k: int = 2):
        """
        Initialize expert dispatch.
        
        Args:
            num_experts: Total number of experts
            top_k: Number of experts per token
        """
        super(Model, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
    
    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> tuple:
        """
        Dispatch tokens to experts.
        
        Args:
            x: Input tokens (batch*seq, hidden_size)
            expert_indices: Expert assignments (batch*seq, top_k)
            
        Returns:
            Tuple of (dispatched_inputs, dispatch_indices, expert_counts):
                dispatched_inputs: (total_tokens, hidden_size) permuted tokens
                dispatch_indices: Original positions for unpermuting
                expert_counts: Number of tokens per expert
        """
        num_tokens = x.shape[0]
        
        # Flatten expert indices: each token appears top_k times
        flat_expert_indices = expert_indices.view(-1)  # (batch*seq*top_k,)
        
        # Create token indices (each token repeated top_k times)
        token_indices = torch.arange(num_tokens, device=x.device).unsqueeze(1)
        token_indices = token_indices.expand(-1, self.top_k).reshape(-1)
        
        # Sort by expert to group tokens going to same expert
        sorted_expert_indices, sort_order = flat_expert_indices.sort()
        
        # Permute token indices according to sort order
        dispatch_indices = token_indices[sort_order]
        
        # Gather tokens in sorted order
        dispatched_inputs = x[dispatch_indices]
        
        # Count tokens per expert
        expert_counts = torch.bincount(sorted_expert_indices, minlength=self.num_experts)
        
        return dispatched_inputs, dispatch_indices, expert_counts
