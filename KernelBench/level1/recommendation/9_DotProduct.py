import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
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


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 256, "num_items": 10000, "embedding_dim": 128},
    # Two-Tower: YouTube DNN retrieval scoring
    {"batch_size": 512, "num_items": 50000, "embedding_dim": 256},
    # NCF: neural collaborative filtering dot product
    {"batch_size": 1024, "num_items": 5000, "embedding_dim": 64},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("recommendation", "9_DotProduct")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    user_embeddings = DISTRIBUTIONS[dist_name]((p["batch_size"], p["embedding_dim"]), dtype=dtype, device=device)
    item_embeddings = DISTRIBUTIONS[dist_name]((p["num_items"], p["embedding_dim"]), dtype=dtype, device=device)
    return [user_embeddings, item_embeddings]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return [True]
