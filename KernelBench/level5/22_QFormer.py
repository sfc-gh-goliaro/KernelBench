import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Q-Former (Querying Transformer) for Vision-Language Alignment.
    
    Uses learned queries to extract visual information through
    cross-attention, producing fixed-length representations.
    
    Based on: "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"
    """
    def __init__(self, vision_dim, hidden_dim, num_queries, num_heads, 
                 num_layers, vocab_size=None):
        """
        :param vision_dim: Dimension of visual features
        :param hidden_dim: Hidden dimension of Q-Former
        :param num_queries: Number of learned query tokens
        :param num_heads: Number of attention heads
        :param num_layers: Number of transformer layers
        :param vocab_size: Vocabulary size for text (None if not needed)
        """
        super(Model, self).__init__()
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        
        # Learned query tokens
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        
        # Vision projection
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        
        # Q-Former transformer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                # Self-attention for queries
                'self_attn': nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True),
                'self_norm': nn.LayerNorm(hidden_dim),
                
                # Cross-attention to vision
                'cross_attn': nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True),
                'cross_norm': nn.LayerNorm(hidden_dim),
                
                # FFN
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                ),
                'ffn_norm': nn.LayerNorm(hidden_dim)
            }))
        
        # Optional: text encoder for image-text matching
        if vocab_size is not None:
            self.text_embed = nn.Embedding(vocab_size, hidden_dim)
            self.text_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, 
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    batch_first=True
                )
                for _ in range(num_layers // 2)
            ])
        else:
            self.text_embed = None
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, vision_features, text_ids=None, text_mask=None):
        """
        Forward pass for Q-Former.
        
        :param vision_features: Visual features (batch, num_patches, vision_dim)
        :param text_ids: Optional text token IDs (batch, text_len)
        :param text_mask: Optional text attention mask (batch, text_len)
        :return: Query outputs (batch, num_queries, hidden_dim)
        """
        batch_size = vision_features.shape[0]
        
        # Project vision features
        vision = self.vision_proj(vision_features)
        
        # Expand queries for batch
        queries = self.queries.expand(batch_size, -1, -1)
        
        # Process through layers
        for layer in self.layers:
            # Self-attention
            q_norm = layer['self_norm'](queries)
            self_out, _ = layer['self_attn'](q_norm, q_norm, q_norm)
            queries = queries + self_out
            
            # Cross-attention to vision
            q_norm = layer['cross_norm'](queries)
            cross_out, _ = layer['cross_attn'](q_norm, vision, vision)
            queries = queries + cross_out
            
            # FFN
            queries = queries + layer['ffn'](layer['ffn_norm'](queries))
        
        # Output projection
        output = self.output_proj(queries)
        
        # Optional: image-text matching with text input
        if text_ids is not None and self.text_embed is not None:
            text_embeds = self.text_embed(text_ids)
            for text_layer in self.text_layers:
                text_embeds = text_layer(text_embeds, 
                    src_key_padding_mask=~text_mask if text_mask is not None else None)
            
            # Compute similarity between queries and text
            query_mean = output.mean(dim=1)  # (batch, hidden_dim)
            text_mean = text_embeds.mean(dim=1)  # (batch, hidden_dim)
            similarity = F.cosine_similarity(query_mean, text_mean, dim=-1)
            
            return output, similarity
        
        return output


# Test parameters
batch_size = 8
num_patches = 196  # 14x14 patches
vision_dim = 1408  # e.g., from EVA-CLIP
hidden_dim = 768
num_queries = 32
num_heads = 12
num_layers = 12

def get_inputs():
    vision_features = torch.randn(batch_size, num_patches, vision_dim)
    return [vision_features]

def get_init_inputs():
    return [vision_dim, hidden_dim, num_queries, num_heads, num_layers]

