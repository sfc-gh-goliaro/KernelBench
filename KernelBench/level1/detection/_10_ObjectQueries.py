import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Object Queries (Object Detection)

    Used by: DETR, Deformable DETR, DINO

    Learnable object queries that are decoded into bounding box predictions.
    Each query learns to attend to and detect one object in the image.

    Shapes:
        batch_size: number of images
        Output: (batch, num_queries, hidden_size) query embeddings
    """

    def __init__(self, num_queries: int = 300, hidden_size: int = 256,
                 with_position: bool = True):
        """
        Initialize object queries.

        Args:
            num_queries: Number of object queries (max detections)
            hidden_size: Query embedding dimension
            with_position: Whether to add positional embeddings
        """
        super(Model, self).__init__()
        self.num_queries = num_queries
        self.hidden_size = hidden_size
        self.with_position = with_position

        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_size)

        if with_position:
            # Learnable positional embeddings
            self.query_pos = nn.Embedding(num_queries, hidden_size)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.query_embed.weight)
        if self.with_position:
            nn.init.normal_(self.query_pos.weight)

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Generate object queries for a batch.

        Args:
            batch_size: Number of images in batch

        Returns:
            Query embeddings (batch, num_queries, hidden_size)
            If with_position, also returns positional embeddings
        """
        device = self.query_embed.weight.device

        # Get query indices
        query_indices = torch.arange(self.num_queries, device=device)

        # Get embeddings
        queries = self.query_embed(query_indices)  # (num_queries, hidden_size)

        # Expand for batch
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)

        if self.with_position:
            query_pos = self.query_pos(query_indices)
            query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
            return queries + query_pos

        return queries
