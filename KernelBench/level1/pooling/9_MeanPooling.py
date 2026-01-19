import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Mean Pooling for Sequences
    
    Used by: Sentence transformers, embedding models (E5, BGE, GTE)
    
    Computes mean of token embeddings, optionally with attention mask.
    Used to create sentence/document embeddings from token embeddings.
    
    Shapes:
        Input: (batch_size, seq_length, hidden_size)
        Output: (batch_size, hidden_size)
    """
    
    def __init__(self):
        """Initialize Mean Pooling."""
        super(Model, self).__init__()
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply mean pooling over sequence dimension.
        
        Args:
            hidden_states: Token embeddings (batch_size, seq_length, hidden_size)
            attention_mask: Mask for valid tokens (batch_size, seq_length)
            
        Returns:
            Pooled embeddings (batch_size, hidden_size)
        """
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        
        # Expand mask for broadcasting
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Sum embeddings and mask counts
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        return sum_embeddings / sum_mask


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 32, "seq_length": 512, "hidden_size": 768},
    # BERT-base: sentence embedding pooling (768 hidden, 512 seq)
    {"batch_size": 64, "seq_length": 512, "hidden_size": 768},
    # E5-large: embedding model pooling (1024 hidden, 512 seq)
    {"batch_size": 32, "seq_length": 512, "hidden_size": 1024},
    # BGE-large-en: embedding model pooling (1024 hidden, 8192 seq for long context)
    {"batch_size": 8, "seq_length": 8192, "hidden_size": 1024},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("pooling", "9_MeanPooling")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    hidden_states = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["hidden_size"]), dtype=dtype, device=device)
    attention_mask = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"]), dtype=dtype, device=device)
    return [hidden_states, attention_mask]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
