import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Last Token Pooling
    
    Used by: Reward models, some embedding models, decoder-only encoders
    
    Extracts the last (non-padding) token's hidden state.
    For causal LMs, the last token contains the most context.
    
    Shapes:
        Input: (batch_size, seq_length, hidden_size)
        Output: (batch_size, hidden_size)
    """
    
    def __init__(self):
        """Initialize Last Token Pooling."""
        super(Model, self).__init__()
    
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Extract last token's hidden state.
        
        Args:
            hidden_states: Token embeddings (batch_size, seq_length, hidden_size)
            attention_mask: Mask for valid tokens (batch_size, seq_length)
            
        Returns:
            Last token embeddings (batch_size, hidden_size)
        """
        if attention_mask is None:
            # No mask: just take the last token
            return hidden_states[:, -1, :]
        
        # Find the position of the last non-padding token
        # attention_mask: 1 for valid tokens, 0 for padding
        sequence_lengths = attention_mask.sum(dim=1) - 1  # (batch_size,)
        batch_size = hidden_states.shape[0]
        
        # Gather last token for each sequence
        return hidden_states[torch.arange(batch_size, device=hidden_states.device), 
                            sequence_lengths.long()]


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 32, "seq_length": 512, "hidden_size": 4096},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("pooling", "10_LastTokenPooling")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    hidden_states = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["hidden_size"]), dtype=dtype, device=device)
    attention_mask = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"]), dtype=dtype, device=device)
    return [hidden_states, attention_mask]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
