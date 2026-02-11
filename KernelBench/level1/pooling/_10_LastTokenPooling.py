import os
import sys
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
