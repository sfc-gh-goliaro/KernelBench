import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Cosine Similarity (Recommendation)

    Used by: Semantic Search, Content-Based Filtering, CLIP

    Computes cosine similarity between embedding vectors. Normalized
    dot product that measures directional alignment independent of magnitude.

    Shapes:
        query_embeddings: (batch, embedding_dim) or (batch, num_queries, embedding_dim)
        candidate_embeddings: (num_candidates, embedding_dim) or (batch, num_candidates, embedding_dim)
        Output: (batch, num_candidates) similarity scores in [-1, 1]
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-8):
        """
        Initialize cosine similarity.

        Args:
            temperature: Temperature scaling for similarity scores
            eps: Small constant for numerical stability
        """
        super(Model, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, query_embeddings: torch.Tensor,
                candidate_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity.

        Args:
            query_embeddings: Query embeddings (batch, embedding_dim)
            candidate_embeddings: Candidate embeddings (num_candidates, embedding_dim)

        Returns:
            Cosine similarity scores (batch, num_candidates)
        """
        # L2 normalize
        query_norm = F.normalize(query_embeddings, p=2, dim=-1, eps=self.eps)
        candidate_norm = F.normalize(candidate_embeddings, p=2, dim=-1, eps=self.eps)

        # Compute cosine similarity via matrix multiplication
        if query_norm.dim() == 2 and candidate_norm.dim() == 2:
            # (batch, dim) @ (candidates, dim)^T -> (batch, candidates)
            similarity = torch.matmul(query_norm, candidate_norm.t())
        else:
            # General batched case
            similarity = torch.matmul(query_norm, candidate_norm.transpose(-2, -1))

        return similarity / self.temperature
