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
    Tree Pruning (Speculative Decoding)

    Used by: Sequoia, SpecInfer, Optimal Tree

    Prunes low-probability branches from draft token tree to limit
    verification cost. Uses cumulative probability or beam-like scoring.

    Shapes:
        node_probs: (batch, num_nodes) probability of each node
        parent_indices: (num_nodes,) parent index for each node
        Output: pruned tree mask and surviving nodes
    """

    def __init__(self, max_nodes: int = 64, prob_threshold: float = 0.01):
        """
        Initialize tree pruning.

        Args:
            max_nodes: Maximum number of nodes to keep
            prob_threshold: Minimum cumulative probability threshold
        """
        super(Model, self).__init__()
        self.max_nodes = max_nodes
        self.prob_threshold = prob_threshold

    def forward(self, node_probs: torch.Tensor, parent_indices: torch.Tensor,
                node_depths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prune tree to keep most promising branches.

        Args:
            node_probs: Probability of each node (batch, num_nodes)
            parent_indices: Parent index for each node (num_nodes,)
            node_depths: Depth of each node (num_nodes,)

        Returns:
            Tuple of:
                - keep_mask: Which nodes to keep (batch, num_nodes)
                - cumulative_probs: Cumulative probability for each node (batch, num_nodes)
        """
        batch_size, num_nodes = node_probs.shape
        device = node_probs.device

        # Compute cumulative probabilities along each path
        cumulative_probs = node_probs.clone()

        # Sort by depth to process in order
        depth_order = torch.argsort(node_depths)

        for node_idx in depth_order:
            if node_idx == 0:
                continue  # Root node
            parent_idx = parent_indices[node_idx]
            if parent_idx >= 0:
                cumulative_probs[:, node_idx] *= cumulative_probs[:, parent_idx]

        # Prune by threshold
        above_threshold = cumulative_probs > self.prob_threshold

        # Keep top-k by cumulative probability
        _, top_indices = torch.topk(cumulative_probs, min(self.max_nodes, num_nodes), dim=-1)
        top_k_mask = torch.zeros_like(node_probs, dtype=torch.bool)
        top_k_mask.scatter_(1, top_indices, True)

        # Combine both criteria
        keep_mask = above_threshold & top_k_mask

        # Always keep root
        keep_mask[:, 0] = True

        return keep_mask, cumulative_probs


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    # Sequoia optimal tree pruning (Llama-2-7B)
    {"batch_size": 8, "num_nodes": 128, "max_nodes": 64},
    # SpecInfer aggressive pruning (Llama-3.1-8B)
    {"batch_size": 4, "num_nodes": 256, "max_nodes": 128},
    # Optimal tree budget pruning (Mistral-7B)
    {"batch_size": 16, "num_nodes": 64, "max_nodes": 32},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("speculative", "9_TreePruning")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    node_probs = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_nodes"]), dtype=dtype, device=device)
    parent_indices = DISTRIBUTIONS[dist_name]((p["num_nodes"]), dtype=dtype, device=device)
    node_depths = DISTRIBUTIONS[dist_name]((p["num_nodes"]), dtype=dtype, device=device)
    return [node_probs, parent_indices, node_depths]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["max_nodes"]]
