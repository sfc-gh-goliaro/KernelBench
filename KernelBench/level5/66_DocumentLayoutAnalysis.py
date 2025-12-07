import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Document Layout Analysis for OCR Models.
    
    Detects and classifies document regions (text, tables, figures)
    with bounding box regression. Used in HunyuanOCR, PaddleOCR-VL.
    
    Based on: LayoutLM, DiT for documents, and detection architectures
    """
    def __init__(self, hidden_dim, num_classes=10, num_queries=100):
        """
        :param hidden_dim: Hidden dimension
        :param num_classes: Number of layout element classes
        :param num_queries: Number of detection queries (DETR-style)
        """
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # Learnable object queries
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        
        # Cross-attention decoder for detection
        self.decoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(hidden_dim, 8, batch_first=True),
                'cross_attn': nn.MultiheadAttention(hidden_dim, 8, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                ),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
                'norm3': nn.LayerNorm(hidden_dim),
            })
            for _ in range(6)
        ])
        
        # Classification head
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        
        # Bounding box head (normalized coordinates)
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # (x_center, y_center, width, height)
        )
        
        # Reading order prediction (pointer network)
        self.order_query = nn.Linear(hidden_dim, hidden_dim)
        self.order_key = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, image_features, image_mask=None):
        """
        Detect document layout elements.
        
        :param image_features: Encoded image features (batch, num_patches, hidden_dim)
        :param image_mask: Optional mask for image features
        :return: Dict with 'logits', 'boxes', 'order_logits'
        """
        batch_size = image_features.shape[0]
        
        # Expand queries for batch
        queries = self.query_embed.expand(batch_size, -1, -1)
        
        # Decode with cross-attention
        for layer in self.decoder_layers:
            # Self-attention over queries
            q_norm = layer['norm1'](queries)
            self_attn_out, _ = layer['self_attn'](q_norm, q_norm, q_norm)
            queries = queries + self_attn_out
            
            # Cross-attention to image features
            q_norm = layer['norm2'](queries)
            cross_attn_out, _ = layer['cross_attn'](
                q_norm, image_features, image_features,
                key_padding_mask=image_mask
            )
            queries = queries + cross_attn_out
            
            # FFN
            queries = queries + layer['ffn'](layer['norm3'](queries))
        
        # Predict classes and boxes
        class_logits = self.class_head(queries)  # (batch, num_queries, num_classes+1)
        bbox_pred = self.bbox_head(queries)      # (batch, num_queries, 4)
        bbox_pred = torch.sigmoid(bbox_pred)     # Normalize to [0, 1]
        
        # Predict reading order (which element comes next)
        order_q = self.order_query(queries)
        order_k = self.order_key(queries)
        order_logits = torch.matmul(order_q, order_k.transpose(-2, -1))
        order_logits = order_logits / math.sqrt(self.hidden_dim)
        
        return {
            'logits': class_logits,
            'boxes': bbox_pred,
            'order_logits': order_logits
        }


# Test parameters
batch_size = 4
num_patches = 196  # 14x14 patches
hidden_dim = 768
num_classes = 10  # text, title, table, figure, list, etc.
num_queries = 100

def get_inputs():
    image_features = torch.randn(batch_size, num_patches, hidden_dim)
    return [image_features]

def get_init_inputs():
    return [hidden_dim, num_classes, num_queries]

