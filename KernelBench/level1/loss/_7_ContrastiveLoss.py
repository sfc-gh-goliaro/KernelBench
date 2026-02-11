import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Contrastive Loss (InfoNCE / SimCLR style)
    
    Used by: CLIP, SimCLR, embedding models, contrastive learning
    
    Computes contrastive loss between positive pairs against negative samples.
    Maximizes similarity of positive pairs, minimizes similarity with negatives.
    
    Shapes:
        Input embeddings: (batch_size, embedding_dim)
        Output: scalar loss
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize Contrastive Loss.
        
        Args:
            temperature: Temperature scaling for softmax
        """
        super(Model, self).__init__()
        self.temperature = temperature
    
    def forward(self, embeddings_a: torch.Tensor, 
                embeddings_b: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between two sets of embeddings.
        
        Assumes embeddings_a[i] and embeddings_b[i] are positive pairs.
        All other combinations are treated as negatives.
        
        Args:
            embeddings_a: First set of embeddings (batch_size, embedding_dim)
            embeddings_b: Second set of embeddings (batch_size, embedding_dim)
            
        Returns:
            Scalar contrastive loss
        """
        batch_size = embeddings_a.shape[0]
        
        # Normalize embeddings
        embeddings_a = F.normalize(embeddings_a, p=2, dim=-1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=-1)
        
        # Compute similarity matrix
        # (batch_size, batch_size)
        similarity = torch.matmul(embeddings_a, embeddings_b.T) / self.temperature
        
        # Labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=embeddings_a.device)
        
        # Cross-entropy loss (InfoNCE)
        # Loss from a -> b
        loss_a = F.cross_entropy(similarity, labels)
        
        # Loss from b -> a
        loss_b = F.cross_entropy(similarity.T, labels)
        
        # Symmetric loss
        loss = (loss_a + loss_b) / 2
        
        return loss
