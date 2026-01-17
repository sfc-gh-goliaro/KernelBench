import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Model(nn.Module):
    """
    Early Exit (Speculative Decoding)

    Used by: CALM, LayerSkip, LITE

    Enables early exiting from transformer layers when confidence is high.
    Each layer has an exit classifier that decides whether to continue
    or exit with the current prediction.

    Shapes:
        hidden_states: (batch, seq_len, hidden_size)
        Output: (batch, seq_len, vocab_size) logits, (batch, seq_len) exit_layer
    """

    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int = 4,
                 confidence_threshold: float = 0.9):
        """
        Initialize early exit module.

        Args:
            hidden_size: Model hidden dimension
            vocab_size: Vocabulary size
            num_layers: Number of transformer layers with exit points
            confidence_threshold: Threshold for early exit decision
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.confidence_threshold = confidence_threshold

        # Exit classifiers for each layer
        self.exit_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size) for _ in range(num_layers)
        ])

        # Confidence predictors
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.SiLU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
            ) for _ in range(num_layers)
        ])

        # Simplified transformer layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, hidden_size)
            ) for _ in range(num_layers)
        ])

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with early exit.

        Args:
            hidden_states: Input hidden states (batch, seq_len, hidden_size)

        Returns:
            Tuple of:
                - logits: Output logits (batch, seq_len, vocab_size)
                - exit_layer: Layer at which each position exited (batch, seq_len)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Track outputs and exit layers
        final_logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)
        exit_layer = torch.full((batch_size, seq_len), self.num_layers - 1, device=device)
        exited = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        current_hidden = hidden_states

        for layer_idx in range(self.num_layers):
            # Apply transformer layer
            current_hidden = current_hidden + self.layers[layer_idx](current_hidden)

            # Get confidence for this layer
            confidence = self.confidence_heads[layer_idx](current_hidden).squeeze(-1)

            # Get logits for this layer
            layer_logits = self.exit_heads[layer_idx](current_hidden)

            # Determine which positions should exit
            should_exit = (confidence > self.confidence_threshold) & ~exited

            # Update outputs for exiting positions
            final_logits[should_exit] = layer_logits[should_exit]
            exit_layer[should_exit] = layer_idx
            exited = exited | should_exit

            # If all positions have exited, stop
            if exited.all():
                break

        # Fill in remaining positions with final layer output
        final_logits[~exited] = layer_logits[~exited]

        return final_logits, exit_layer


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_len = 128
hidden_size = 4096
vocab_size = 32000
num_layers = 4

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    return [hidden_states]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, vocab_size, num_layers]
