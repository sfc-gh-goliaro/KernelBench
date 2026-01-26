import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
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


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 256, "num_candidates": 10000, "embedding_dim": 768},
    # CLIP: vision-language similarity matching
    {"batch_size": 512, "num_candidates": 50000, "embedding_dim": 512},
    # Sentence-BERT: semantic text similarity
    {"batch_size": 128, "num_candidates": 100000, "embedding_dim": 384},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("recommendation", "10_CosineSimilarity")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    query_embeddings = DISTRIBUTIONS[dist_name]((p["batch_size"], p["embedding_dim"]), dtype=dtype, device=device)
    candidate_embeddings = DISTRIBUTIONS[dist_name]((p["num_candidates"], p["embedding_dim"]), dtype=dtype, device=device)
    return [query_embeddings, candidate_embeddings]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
