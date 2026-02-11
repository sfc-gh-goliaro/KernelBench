import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Model(nn.Module):
    """
    Draft Head (Speculative Decoding)

    Used by: EAGLE, EAGLE-2, Medusa, Hydra, Lookahead

    Unified draft prediction head that supports both:
    - Parallel mode (Medusa-style): Predict multiple future tokens independently
    - Autoregressive mode (EAGLE-style): Predict tokens sequentially with feedback

    This consolidates EagleHead and DraftHead into a single configurable operator.

    Shapes:
        hidden_states: (batch, seq_len, hidden_size) or (batch, hidden_size)
        Output: (batch, [seq_len,] num_draft_heads, vocab_size) logits
    """

    def __init__(self, hidden_size: int, vocab_size: int, num_draft_heads: int = 4,
                 mode: str = 'parallel', inner_dim: Optional[int] = None):
        """
        Initialize draft head.

        Args:
            hidden_size: Model hidden dimension
            vocab_size: Vocabulary size
            num_draft_heads: Number of draft token predictions
            mode: 'parallel' (Medusa-style) or 'autoregressive' (EAGLE-style)
            inner_dim: Inner dimension for autoregressive mode (default: hidden_size // 4)
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_draft_heads = num_draft_heads
        self.mode = mode
        self.inner_dim = inner_dim or hidden_size // 4

        if mode == 'parallel':
            # Medusa-style: independent heads for each draft position
            self.draft_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, vocab_size)
                ) for _ in range(num_draft_heads)
            ])
        elif mode == 'autoregressive':
            # EAGLE-style: sequential prediction with token feedback
            self.fc_in = nn.Linear(hidden_size, self.inner_dim)

            # Autoregressive layers that take previous prediction
            self.draft_layers = nn.ModuleList([
                nn.Linear(self.inner_dim + hidden_size, self.inner_dim)
                for _ in range(num_draft_heads)
            ])

            # Output heads for each position
            self.lm_heads = nn.ModuleList([
                nn.Linear(self.inner_dim, vocab_size, bias=False)
                for _ in range(num_draft_heads)
            ])

            # Embedding for feeding back draft tokens
            self.token_embed = nn.Embedding(vocab_size, hidden_size)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'parallel' or 'autoregressive'")

    def forward(self, hidden_states: torch.Tensor,
                base_hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate draft token logits.

        Args:
            hidden_states: Hidden states from base model
                - Parallel mode: (batch, seq_len, hidden_size)
                - Autoregressive mode: (batch, hidden_size)
            base_hidden: Optional base context for autoregressive mode

        Returns:
            Draft logits:
                - Parallel mode: (batch, seq_len, num_draft_heads, vocab_size)
                - Autoregressive mode: (batch, num_draft_heads, vocab_size)
        """
        if self.mode == 'parallel':
            return self._forward_parallel(hidden_states)
        else:
            return self._forward_autoregressive(hidden_states, base_hidden)

    def _forward_parallel(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Parallel (Medusa-style) forward pass."""
        # Get logits from each independent draft head
        draft_logits = []
        for head in self.draft_heads:
            logits = head(hidden_states)  # (batch, seq_len, vocab_size)
            draft_logits.append(logits)

        # Stack along new dimension: (batch, seq_len, num_heads, vocab)
        return torch.stack(draft_logits, dim=-2)

    def _forward_autoregressive(self, hidden_states: torch.Tensor,
                                  base_hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Autoregressive (EAGLE-style) forward pass."""
        # Handle both 2D and 3D input
        if hidden_states.dim() == 3:
            # Take last position for autoregressive
            hidden_states = hidden_states[:, -1, :]

        batch_size = hidden_states.shape[0]

        if base_hidden is None:
            base_hidden = hidden_states

        # Initial projection
        h = F.silu(self.fc_in(hidden_states))

        # Collect draft logits
        draft_logits = []

        for i in range(self.num_draft_heads):
            # Combine current state with base context
            combined = torch.cat([h, base_hidden], dim=-1)
            h = F.silu(self.draft_layers[i](combined))

            # Compute logits for this position
            logits = self.lm_heads[i](h)
            draft_logits.append(logits)

            # Sample token and embed for next iteration (greedy)
            token = logits.argmax(dim=-1)
            token_emb = self.token_embed(token)

            # Update base_hidden with token embedding
            base_hidden = base_hidden + token_emb

        # Stack: (batch, num_draft_heads, vocab_size)
        return torch.stack(draft_logits, dim=1)
