import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Model(nn.Module):
    """
    Unified Sampler with Configurable Filtering Strategies

    Used by: vLLM, SGLang, TensorRT-LLM, all LLM inference

    Combines temperature scaling, top-k, top-p (nucleus), and min-p filtering
    into a single configurable operator. Supports optional repetition and
    presence/frequency penalties when context tokens are provided.

    Filtering is applied in order: temperature -> top-k -> top-p -> min-p

    Shapes:
        logits: (batch_size, vocab_size) raw logits from model
        input_ids: Optional (batch_size, context_len) for penalties
        Output: (batch_size,) sampled token indices
    """

    def __init__(self,
                 temperature: float = 1.0,
                 top_k: int = 0,
                 top_p: float = 1.0,
                 min_p: float = 0.0,
                 repetition_penalty: float = 1.0,
                 presence_penalty: float = 0.0,
                 frequency_penalty: float = 0.0):
        """
        Initialize unified sampler with configurable parameters.

        Args:
            temperature: Sampling temperature (>0). Lower = more deterministic
            top_k: Number of top tokens to consider (0 = disabled)
            top_p: Cumulative probability threshold for nucleus sampling (1.0 = disabled)
            min_p: Minimum probability threshold relative to max prob (0.0 = disabled)
            repetition_penalty: Penalty for repeated tokens (1.0 = disabled)
            presence_penalty: Flat penalty for tokens present in context (0.0 = disabled)
            frequency_penalty: Per-occurrence penalty for context tokens (0.0 = disabled)
        """
        super(Model, self).__init__()
        assert temperature > 0, "Temperature must be positive"

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

    def _apply_repetition_penalty(self, logits: torch.Tensor,
                                   input_ids: torch.Tensor) -> torch.Tensor:
        """Apply multiplicative repetition penalty to context tokens."""
        batch_size, vocab_size = logits.shape
        device = logits.device

        # Vectorized: create penalty multipliers
        # For each batch, penalize tokens that appear in input_ids
        for b in range(batch_size):
            unique_tokens = input_ids[b].unique()
            # Positive logits: divide by penalty
            pos_mask = logits[b, unique_tokens] > 0
            logits[b, unique_tokens[pos_mask]] /= self.repetition_penalty
            # Negative logits: multiply by penalty
            neg_mask = logits[b, unique_tokens] < 0
            logits[b, unique_tokens[neg_mask]] *= self.repetition_penalty

        return logits

    def _apply_presence_frequency_penalty(self, logits: torch.Tensor,
                                           input_ids: torch.Tensor) -> torch.Tensor:
        """Apply additive presence and frequency penalties."""
        batch_size, vocab_size = logits.shape
        device = logits.device

        # Count token frequencies
        token_counts = torch.zeros(batch_size, vocab_size, device=device)
        for b in range(batch_size):
            unique, counts = input_ids[b].unique(return_counts=True)
            token_counts[b, unique] = counts.float()

        # Presence penalty: flat penalty for any token that appears
        presence_mask = (token_counts > 0).float()

        # Total penalty = presence_penalty * present + frequency_penalty * count
        penalty = (self.presence_penalty * presence_mask +
                   self.frequency_penalty * token_counts)

        return logits - penalty

    def _apply_top_k(self, logits: torch.Tensor) -> torch.Tensor:
        """Filter to keep only top-k tokens."""
        if self.top_k <= 0 or self.top_k >= logits.shape[-1]:
            return logits

        # Get the k-th largest value as threshold
        top_k_values = torch.topk(logits, self.top_k, dim=-1).values
        threshold = top_k_values[..., -1, None]

        # Mask out tokens below threshold
        return torch.where(logits < threshold,
                          torch.full_like(logits, float('-inf')), logits)

    def _apply_top_p(self, logits: torch.Tensor) -> torch.Tensor:
        """Filter using nucleus (top-p) sampling."""
        if self.top_p >= 1.0:
            return logits

        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # Compute cumulative probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff: remove tokens after cumulative prob exceeds p
        sorted_indices_to_remove = cumulative_probs > self.top_p
        # Shift to include the token that crosses threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter mask back to original order
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )

        return logits.masked_fill(indices_to_remove, float('-inf'))

    def _apply_min_p(self, logits: torch.Tensor) -> torch.Tensor:
        """Filter tokens below min_p * max_probability."""
        if self.min_p <= 0.0:
            return logits

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Compute threshold as min_p * max_prob
        max_probs = probs.max(dim=-1, keepdim=True).values
        threshold = self.min_p * max_probs

        # Mask out tokens below threshold
        return torch.where(probs < threshold,
                          torch.full_like(logits, float('-inf')), logits)

    def forward(self, logits: torch.Tensor,
                input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply configured sampling strategy and return sampled tokens.

        Args:
            logits: Raw logits from model (batch_size, vocab_size)
            input_ids: Optional context tokens for penalties (batch_size, context_len)

        Returns:
            Sampled token indices (batch_size,)
        """
        # Clone to avoid modifying input
        logits = logits.clone()

        # Apply penalties if context provided
        if input_ids is not None:
            if self.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, input_ids)

            if self.presence_penalty != 0.0 or self.frequency_penalty != 0.0:
                logits = self._apply_presence_frequency_penalty(logits, input_ids)

        # Apply temperature scaling
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Apply filtering in order: top-k -> top-p -> min-p
        logits = self._apply_top_k(logits)
        logits = self._apply_top_p(logits)
        logits = self._apply_min_p(logits)

        # Convert to probabilities and sample
        probs = F.softmax(logits, dim=-1)
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return sampled_tokens


# ============================================================================
# Benchmark Configuration
# ============================================================================
