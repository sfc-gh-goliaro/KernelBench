import os
import sys
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
