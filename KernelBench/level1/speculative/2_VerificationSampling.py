import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class Model(nn.Module):
    """
    Verification Sampling (Speculative Decoding)

    Used by: All speculative decoding methods (vLLM, SGLang, EAGLE, Medusa, etc.)

    Unified verification module that accepts/rejects draft tokens by comparing
    draft vs target model distributions using rejection sampling. Supports
    optional bonus token sampling from the target distribution.

    This consolidates TokenVerification, DraftTargetCoord, and RejectionSampling
    into a single operator.

    Shapes:
        draft_probs: (batch, num_draft, vocab_size) draft model probabilities
        target_probs: (batch, num_draft + 1, vocab_size) target model probabilities
        draft_tokens: (batch, num_draft) drafted token IDs
        Output: verified tokens, acceptance info, optional bonus token
    """

    def __init__(self, sample_bonus_token: bool = True):
        """
        Initialize verification sampling.

        Args:
            sample_bonus_token: Whether to sample a bonus token from target
                               distribution at the first rejected position
        """
        super(Model, self).__init__()
        self.sample_bonus_token = sample_bonus_token

    def forward(self, draft_probs: torch.Tensor, target_probs: torch.Tensor,
                draft_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Verify draft tokens using rejection sampling.

        Algorithm:
        1. For each draft token, compute acceptance probability = min(1, p_target/p_draft)
        2. Accept token if uniform_random < acceptance_probability
        3. Find first rejection (all subsequent tokens are invalid)
        4. Optionally sample bonus token from target at rejection point

        Args:
            draft_probs: Draft model probabilities (batch, num_draft, vocab_size)
            target_probs: Target model probabilities (batch, num_draft+1, vocab_size)
                         Extra position is for bonus token sampling
            draft_tokens: Draft token IDs (batch, num_draft)

        Returns:
            Tuple of:
                - accepted_tokens: Verified token sequence (batch, max_len)
                - accept_mask: Boolean acceptance mask (batch, num_draft)
                - num_accepted: Number of accepted tokens per batch (batch,)
                - bonus_tokens: Bonus tokens sampled at rejection (batch,) or None
        """
        batch_size, num_draft, vocab_size = draft_probs.shape
        device = draft_probs.device

        # Get probabilities for the specific drafted tokens
        # Using gather for efficient indexing
        draft_token_probs = torch.gather(
            draft_probs, 2, draft_tokens.unsqueeze(-1)
        ).squeeze(-1)  # (batch, num_draft)

        target_token_probs = torch.gather(
            target_probs[:, :num_draft, :], 2, draft_tokens.unsqueeze(-1)
        ).squeeze(-1)  # (batch, num_draft)

        # Acceptance probability: min(1, p_target / p_draft)
        # Add epsilon to avoid division by zero
        acceptance_probs = torch.clamp(
            target_token_probs / (draft_token_probs + 1e-10),
            max=1.0
        )

        # Sample uniform random for rejection test
        uniform_samples = torch.rand_like(acceptance_probs)

        # Accept where uniform < acceptance_prob
        token_accepted = uniform_samples < acceptance_probs

        # Find cumulative acceptance (must accept all previous to accept current)
        cumulative_accepted = torch.cumprod(token_accepted.float(), dim=1)
        accept_mask = cumulative_accepted > 0.5

        # Count accepted tokens per sequence
        num_accepted = accept_mask.sum(dim=1).long()  # (batch,)

        # Build output token sequence
        if self.sample_bonus_token:
            # Maximum output length is num_accepted + 1 (for bonus token)
            max_output_len = num_accepted.max().item() + 1
            accepted_tokens = torch.zeros(
                batch_size, max_output_len, dtype=draft_tokens.dtype, device=device
            )

            # Fill in accepted draft tokens and sample bonus token
            bonus_tokens = torch.zeros(batch_size, dtype=draft_tokens.dtype, device=device)

            for b in range(batch_size):
                n = num_accepted[b].item()
                if n > 0:
                    accepted_tokens[b, :n] = draft_tokens[b, :n]

                # Sample bonus token from target distribution at position n
                # Use modified distribution: max(0, p_target - p_draft) normalized
                # Or simplified: just sample from target
                if n < num_draft:
                    # Sample from adjusted distribution
                    p_target = target_probs[b, n]
                    p_draft = draft_probs[b, n]
                    adjusted = F.relu(p_target - p_draft)
                    adjusted_sum = adjusted.sum()
                    if adjusted_sum > 1e-10:
                        adjusted = adjusted / adjusted_sum
                    else:
                        adjusted = p_target
                    bonus = torch.multinomial(adjusted, 1).squeeze()
                else:
                    # All accepted, sample bonus from target at position num_draft
                    bonus = torch.multinomial(target_probs[b, n], 1).squeeze()

                bonus_tokens[b] = bonus
                accepted_tokens[b, n] = bonus

            return accepted_tokens, accept_mask, num_accepted, bonus_tokens
        else:
            # No bonus token, just return accepted draft tokens
            accepted_tokens = draft_tokens.clone()
            accepted_tokens[~accept_mask] = 0

            return accepted_tokens, accept_mask, num_accepted, None


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 32, "num_draft": 5, "vocab_size": 32000},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("speculative", "2_VerificationSampling")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    draft_logits = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_draft"], p["vocab_size"]), dtype=dtype, device=device)
    target_logits = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_draft"] + 1, p["vocab_size"]), dtype=dtype, device=device)
    return [draft_probs, target_probs, draft_tokens]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return [True]
