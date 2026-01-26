import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
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


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096, "num_experts": 8, "top_k": 2},
    # DeepSeek-V2-Lite: hidden_size=2048, num_experts=64, top_k=6
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 2048, "num_experts": 64, "top_k": 6},
    # Mixtral-8x7B: hidden_size=4096, num_experts=8, top_k=2
    {"batch_size": 8, "seq_length": 4096, "hidden_size": 4096, "num_experts": 8, "top_k": 2},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("moe", "2_Expert_Dispatch")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"] * p["seq_length"], p["hidden_size"]), dtype=dtype, device=device)
    return [x, expert_indices]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_experts"], p["top_k"]]
