import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Presence and Frequency Penalty (Standalone Logits Processor)

    Used by: ChatGPT-style APIs, OpenAI-compatible endpoints

    OpenAI-style presence and frequency penalties for diverse generation.
    - Presence penalty: flat penalty for any token that appears in context
    - Frequency penalty: penalty proportional to token occurrence count

    This is a standalone processor that can be composed with other
    sampling strategies. For combined sampling, use UnifiedSampler.

    Shapes:
        logits: (batch_size, vocab_size) input logits
        input_ids: (batch_size, context_len) context token IDs
        Output: (batch_size, vocab_size) penalized logits
    """

    def __init__(self, presence_penalty: float = 0.0, frequency_penalty: float = 0.0):
        """
        Initialize presence/frequency penalties.

        Args:
            presence_penalty: Flat penalty for present tokens (subtracted from logits)
            frequency_penalty: Per-occurrence penalty (subtracted from logits)
        """
        super(Model, self).__init__()
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

    def forward(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply presence and frequency penalties.

        Args:
            logits: Logits tensor (batch_size, vocab_size)
            input_ids: Previous token IDs (batch_size, context_len)

        Returns:
            Penalized logits (batch_size, vocab_size)
        """
        batch_size, vocab_size = logits.shape
        device = logits.device

        # Count token frequencies in context
        token_counts = torch.zeros(batch_size, vocab_size, device=device)
        for b in range(batch_size):
            unique, counts = input_ids[b].unique(return_counts=True)
            token_counts[b, unique] = counts.float()

        # Presence penalty: apply to any token that appears (count > 0)
        presence_mask = (token_counts > 0).float()

        # Total penalty = presence_penalty * present + frequency_penalty * count
        penalty = (self.presence_penalty * presence_mask +
                   self.frequency_penalty * token_counts)

        # Subtract penalty from logits
        return logits - penalty


# ============================================================================
# Benchmark Configuration
# ============================================================================
