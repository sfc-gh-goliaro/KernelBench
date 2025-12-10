import torch
import torch.nn as nn
import torch.nn.functional as F
import math


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Vision-Language Embedding Fusion Layer.
    
    Combines image patches and text tokens into a unified
    embedding space for multimodal processing.
    
    Based on: LLaVA, GPT-4V, and similar VLM architectures
    """
    def __init__(self, vision_dim, text_dim, hidden_dim, num_image_tokens, 
                 vision_layers=2, use_resampler=True):
        """
        :param vision_dim: Dimension of vision features
        :param text_dim: Dimension of text features
        :param hidden_dim: Unified hidden dimension
        :param num_image_tokens: Number of image tokens to produce
        :param vision_layers: Number of vision projection layers
        :param use_resampler: Whether to use a resampler for vision tokens
        """
        super(Model, self).__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_image_tokens = num_image_tokens
        self.use_resampler = use_resampler
        
        # Vision projection (MLP)
        vision_proj_layers = []
        in_dim = vision_dim
        for i in range(vision_layers):
            out_dim = hidden_dim if i == vision_layers - 1 else vision_dim
            vision_proj_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU() if i < vision_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim
        self.vision_proj = nn.Sequential(*vision_proj_layers)
        
        # Text projection (if dimensions differ)
        if text_dim != hidden_dim:
            self.text_proj = nn.Linear(text_dim, hidden_dim)
        else:
            self.text_proj = nn.Identity()
        
        # Resampler (like Q-Former) for controlling number of image tokens
        if use_resampler:
            self.query_tokens = nn.Parameter(torch.randn(1, num_image_tokens, hidden_dim) * 0.02)
            self.resampler = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            )
        
        # Modality embeddings
        self.vision_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.text_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Learnable separator tokens
        self.image_start = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.image_end = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
    
    def forward(self, vision_features, text_features, image_positions=None):
        """
        Forward pass for vision-language embedding fusion.
        
        :param vision_features: Vision features (batch, num_patches, vision_dim)
        :param text_features: Text features (batch, text_len, text_dim)
        :param image_positions: Positions to insert images in text (optional)
        :return: Fused embeddings (batch, total_len, hidden_dim)
        """
        batch_size = vision_features.shape[0]
        
        # Project vision features
        vision_proj = self.vision_proj(vision_features)
        
        # Apply resampler if enabled
        if self.use_resampler:
            queries = self.query_tokens.expand(batch_size, -1, -1)
            vision_proj = self.resampler(queries, vision_proj)
        
        # Add modality embedding
        vision_proj = vision_proj + self.vision_embed
        
        # Add separator tokens
        image_start = self.image_start.expand(batch_size, -1, -1)
        image_end = self.image_end.expand(batch_size, -1, -1)
        vision_tokens = torch.cat([image_start, vision_proj, image_end], dim=1)
        
        # Project text features
        text_proj = self.text_proj(text_features)
        text_proj = text_proj + self.text_embed
        
        # Combine vision and text
        if image_positions is not None:
            # Insert image tokens at specified positions
            output = []
            for b in range(batch_size):
                pos = image_positions[b] if isinstance(image_positions, list) else image_positions
                text_b = text_proj[b]
                vision_b = vision_tokens[b]
                
                # Split text at image position
                text_before = text_b[:pos]
                text_after = text_b[pos:]
                
                # Concatenate
                seq = torch.cat([text_before, vision_b, text_after], dim=0)
                output.append(seq)
            
            # Pad to same length
            max_len = max(o.shape[0] for o in output)
            output = torch.stack([
                F.pad(o, (0, 0, 0, max_len - o.shape[0]))
                for o in output
            ])
        else:
            # Default: prepend vision to text
            output = torch.cat([vision_tokens, text_proj], dim=1)
        
        return self.norm(output)


# Test parameters
batch_size = 4
num_patches = 256  # e.g., 16x16 patches from ViT
vision_dim = 1024
text_len = 512
text_dim = 4096
hidden_dim = 4096
num_image_tokens = 64

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    vision_features = torch.randn(batch_size, num_patches, vision_dim)
    text_features = torch.randn(batch_size, text_len, text_dim)
    return [vision_features, text_features]

def get_init_inputs():
    return [vision_dim, text_dim, hidden_dim, num_image_tokens]

