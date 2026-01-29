import os
import sys
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
