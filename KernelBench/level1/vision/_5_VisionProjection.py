import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Vision-to-LLM Projection
    
    Used by: LLaVA, Qwen-VL, InternVL
    
    MLP (typically 2-layer) projecting vision encoder features to
    LLM embedding space.
    
    Shapes:
        Input: (batch, num_patches, vision_dim)
        Output: (batch, num_patches, llm_dim)
    """
    
    def __init__(self, vision_dim: int, llm_dim: int, hidden_dim: int = None):
        """
        Initialize vision projection.
        
        Args:
            vision_dim: Vision encoder output dimension
            llm_dim: LLM embedding dimension
            hidden_dim: Hidden dimension (default: llm_dim)
        """
        super(Model, self).__init__()
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        self.hidden_dim = hidden_dim or llm_dim
        
        # 2-layer MLP with GELU
        self.proj1 = nn.Linear(vision_dim, self.hidden_dim)
        self.proj2 = nn.Linear(self.hidden_dim, llm_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to LLM space.
        
        Args:
            x: Vision features (batch, num_patches, vision_dim)
            
        Returns:
            Projected features (batch, num_patches, llm_dim)
        """
        x = self.proj1(x)
        x = F.gelu(x)
        x = self.proj2(x)
        return x
