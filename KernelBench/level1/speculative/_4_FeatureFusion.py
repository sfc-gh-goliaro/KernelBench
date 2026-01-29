import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Feature Fusion (Speculative Decoding)

    Used by: EAGLE, Lookahead Decoding

    Fuses features from base model with draft model for improved draft
    token prediction. Combines hidden states with token embeddings.

    Shapes:
        hidden_states: (batch, seq_len, hidden_size)
        token_embeds: (batch, seq_len, embed_size)
        Output: (batch, seq_len, hidden_size)
    """

    def __init__(self, hidden_size: int, embed_size: int):
        """
        Initialize feature fusion.

        Args:
            hidden_size: Model hidden dimension
            embed_size: Token embedding dimension
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        # Projection for token embeddings
        self.embed_proj = nn.Linear(embed_size, hidden_size)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Layer norm for output
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """
        Fuse hidden states with token embeddings.

        Args:
            hidden_states: Base model hidden states (batch, seq_len, hidden_size)
            token_embeds: Token embeddings (batch, seq_len, embed_size)

        Returns:
            Fused features (batch, seq_len, hidden_size)
        """
        # Project token embeddings
        projected_embeds = self.embed_proj(token_embeds)

        # Concatenate and fuse
        combined = torch.cat([hidden_states, projected_embeds], dim=-1)
        fused = self.fusion(combined)

        # Residual connection and normalize
        output = self.norm(hidden_states + fused)

        return output


# ============================================================================
# Benchmark Configuration
# ============================================================================
