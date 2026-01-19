import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Model(nn.Module):
    """
    Tree Expansion (Speculative Decoding)

    Used by: SpecInfer, Sequoia, EAGLE-2

    Expands draft token tree by selecting top-k candidates at each node.
    Manages tree structure for speculative decoding verification.

    Shapes:
        logits: (batch, num_nodes, vocab_size)
        parent_indices: (num_nodes,) parent index for each node
        Output: expanded tree tokens and structure
    """

    def __init__(self, top_k: int = 4, max_depth: int = 5):
        """
        Initialize tree expansion.

        Args:
            top_k: Number of top candidates to expand at each node
            max_depth: Maximum tree depth
        """
        super(Model, self).__init__()
        self.top_k = top_k
        self.max_depth = max_depth

    def forward(self, logits: torch.Tensor, current_depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expand tree with top-k tokens at each leaf node.

        Args:
            logits: Logits for each current leaf (batch, num_leaves, vocab_size)
            current_depth: Current depth of each leaf (num_leaves,)

        Returns:
            Tuple of:
                - expanded_tokens: New token candidates (batch, num_leaves, top_k)
                - expanded_probs: Probabilities (batch, num_leaves, top_k)
                - expand_mask: Which leaves to expand (num_leaves,)
        """
        batch_size, num_leaves, vocab_size = logits.shape
        device = logits.device

        # Get probabilities
        probs = F.softmax(logits, dim=-1)

        # Get top-k for each leaf
        top_probs, top_tokens = torch.topk(probs, self.top_k, dim=-1)

        # Determine which leaves can be expanded (not at max depth)
        expand_mask = current_depth < self.max_depth

        # Zero out expansions for nodes at max depth
        expanded_tokens = top_tokens.clone()
        expanded_probs = top_probs.clone()

        # Mask out expansions at max depth
        expanded_tokens[:, ~expand_mask, :] = 0
        expanded_probs[:, ~expand_mask, :] = 0.0

        return expanded_tokens, expanded_probs, expand_mask


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    # SpecInfer tree expansion (Llama-2-7B)
    {"batch_size": 8, "num_leaves": 16, "vocab_size": 32000, "top_k": 4, "max_depth": 5},
    # Sequoia optimal tree (Llama-3.1-8B)
    {"batch_size": 4, "num_leaves": 32, "vocab_size": 128256, "top_k": 6, "max_depth": 7},
    # EAGLE-2 dynamic tree (Vicuna-7B)
    {"batch_size": 8, "num_leaves": 24, "vocab_size": 32000, "top_k": 5, "max_depth": 6},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("speculative", "8_TreeExpansion")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    logits = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_leaves"], p["vocab_size"]), dtype=dtype, device=device)
    return [logits, current_depth]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["top_k"], p["max_depth"]]
