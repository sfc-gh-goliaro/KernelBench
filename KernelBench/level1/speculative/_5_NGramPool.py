import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Model(nn.Module):
    """
    N-Gram Pool (Speculative Decoding)

    Used by: Prompt Lookup Decoding, REST

    Retrieves matching n-grams from prompt/context to use as draft tokens.
    Looks up recent tokens in the context and returns continuation candidates.

    Shapes:
        context_ids: (batch, context_len) token IDs
        query_ids: (batch, query_len) recent token IDs to match
        Output: (batch, max_matches, ngram_len) matching continuations
    """

    def __init__(self, ngram_len: int = 4, max_matches: int = 5):
        """
        Initialize n-gram pool.

        Args:
            ngram_len: Length of n-gram to match
            max_matches: Maximum number of matching continuations to return
        """
        super(Model, self).__init__()
        self.ngram_len = ngram_len
        self.max_matches = max_matches

    def forward(self, context_ids: torch.Tensor, query_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find n-gram matches and return continuations.

        Args:
            context_ids: Context token IDs (batch, context_len)
            query_ids: Query token IDs to match (batch, query_len)

        Returns:
            Tuple of:
                - matches: Matching continuations (batch, max_matches, ngram_len)
                - match_mask: Valid match mask (batch, max_matches)
        """
        batch_size = context_ids.shape[0]
        context_len = context_ids.shape[1]
        query_len = query_ids.shape[1]
        device = context_ids.device

        # Output tensors
        matches = torch.zeros(batch_size, self.max_matches, self.ngram_len,
                            dtype=context_ids.dtype, device=device)
        match_mask = torch.zeros(batch_size, self.max_matches, dtype=torch.bool, device=device)

        # For each batch
        for b in range(batch_size):
            match_count = 0

            # Slide through context looking for query match
            for i in range(context_len - query_len - self.ngram_len + 1):
                # Check if this position matches query
                if torch.equal(context_ids[b, i:i+query_len], query_ids[b]):
                    # Get continuation
                    continuation = context_ids[b, i+query_len:i+query_len+self.ngram_len]
                    matches[b, match_count] = continuation
                    match_mask[b, match_count] = True
                    match_count += 1

                    if match_count >= self.max_matches:
                        break

        return matches, match_mask


# ============================================================================
# Benchmark Configuration
# ============================================================================
