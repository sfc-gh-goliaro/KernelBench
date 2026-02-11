import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Dot Product Similarity (Recommendation)

    Used by: Two-Tower Models, Matrix Factorization, DSSM

    Computes dot product similarity between user and item embeddings
    for recommendation scoring. Core operation in retrieval models.

    Shapes:
        user_embeddings: (batch, embedding_dim) or (batch, num_users, embedding_dim)
        item_embeddings: (batch, embedding_dim) or (num_items, embedding_dim)
        Output: (batch,) or (batch, num_items) similarity scores
    """

    def __init__(self, normalize: bool = False, temperature: float = 1.0):
        """
        Initialize dot product similarity.

        Args:
            normalize: Whether to L2 normalize embeddings before dot product
            temperature: Temperature scaling for similarity scores
        """
        super(Model, self).__init__()
        self.normalize = normalize
        self.temperature = temperature

    def forward(self, user_embeddings: torch.Tensor,
                item_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute dot product similarity.

        Args:
            user_embeddings: User embeddings (batch, embedding_dim)
            item_embeddings: Item embeddings (num_items, embedding_dim) or
                           (batch, embedding_dim) for single item per user

        Returns:
            Similarity scores (batch, num_items) or (batch,)
        """
        if self.normalize:
            user_embeddings = F.normalize(user_embeddings, p=2, dim=-1)
            item_embeddings = F.normalize(item_embeddings, p=2, dim=-1)

        if user_embeddings.dim() == 2 and item_embeddings.dim() == 2:
            if user_embeddings.shape[0] == item_embeddings.shape[0]:
                # Paired: (batch, dim) @ (batch, dim) -> (batch,)
                scores = (user_embeddings * item_embeddings).sum(dim=-1)
            else:
                # All pairs: (batch, dim) @ (num_items, dim)^T -> (batch, num_items)
                scores = torch.matmul(user_embeddings, item_embeddings.t())
        else:
            # General case
            scores = torch.matmul(user_embeddings, item_embeddings.transpose(-2, -1))

        return scores / self.temperature
