import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
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


PARAMETERS = [
    {"batch_size": 8, "seq_len": 128, "hidden_size": 4096, "embed_size": 4096},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("speculative", "4_FeatureFusion")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    hidden_states = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_len"], p["hidden_size"]), dtype=dtype, device=device)
    token_embeds = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_len"], p["embed_size"]), dtype=dtype, device=device)
    return [hidden_states, token_embeds]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["embed_size"]]
