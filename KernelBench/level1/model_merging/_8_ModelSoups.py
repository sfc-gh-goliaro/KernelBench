import os
import sys
import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):
    """
    Model Soups (Model Merging)

    Used by: WiSE-FT, Model Soups

    Averages weights from multiple fine-tuned checkpoints (model soup).
    Can use uniform averaging or greedy soup selection based on validation.

    Shapes:
        weights_list: List of (param_shape) weight tensors
        scores: Optional (num_models,) validation scores for weighted averaging
        Output: (param_shape) merged weights
    """

    def __init__(self, mode: str = 'uniform'):
        """
        Initialize model soups.

        Args:
            mode: 'uniform' for equal weights, 'weighted' for score-based
        """
        super(Model, self).__init__()
        self.mode = mode

    def forward(self, *weights_and_scores) -> torch.Tensor:
        """
        Merge multiple model weights.

        Args:
            *weights_and_scores: Variable number of weight tensors,
                                optionally followed by scores tensor

        Returns:
            Merged weights (param_shape)
        """
        # Check if last argument is scores (1D tensor with length matching num weights)
        if len(weights_and_scores) >= 2:
            potential_scores = weights_and_scores[-1]
            if potential_scores.dim() == 1 and len(potential_scores) == len(weights_and_scores) - 1:
                weights = weights_and_scores[:-1]
                scores = potential_scores
            else:
                weights = weights_and_scores
                scores = None
        else:
            weights = weights_and_scores
            scores = None

        num_models = len(weights)

        if self.mode == 'uniform' or scores is None:
            # Simple uniform averaging
            merged = sum(weights) / num_models
        else:
            # Weighted averaging based on scores
            # Normalize scores to sum to 1
            normalized_scores = torch.softmax(scores, dim=0)

            merged = torch.zeros_like(weights[0])
            for w, s in zip(weights, normalized_scores):
                merged += w * s

        return merged
