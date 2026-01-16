import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Perceiver-style Resampler
    
    Used by: Qwen-VL, Idefics, Flamingo-style VLMs
    """
    
    def __init__(self, vision_dim: int, output_dim: int, num_queries: int = 64, 
                 num_heads: int = 8, num_layers: int = 6):
        super(Model, self).__init__()
        self.vision_dim = vision_dim
        self.output_dim = output_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        
        self.queries = nn.Parameter(torch.randn(1, num_queries, output_dim) * 0.02)
        
        self.input_proj = nn.Linear(vision_dim, output_dim) if vision_dim != output_dim else nn.Identity()
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(output_dim, num_heads, batch_first=True),
                'ff': nn.Sequential(
                    nn.Linear(output_dim, output_dim * 4),
                    nn.GELU(),
                    nn.Linear(output_dim * 4, output_dim)
                ),
                'norm1': nn.LayerNorm(output_dim),
                'norm2': nn.LayerNorm(output_dim)
            })
            for _ in range(num_layers)
        ])
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        batch_size = vision_features.shape[0]
        
        vision_features = self.input_proj(vision_features)
        queries = self.queries.expand(batch_size, -1, -1)
        
        for layer in self.layers:
            residual = queries
            queries = layer['norm1'](queries)
            queries, _ = layer['cross_attn'](queries, vision_features, vision_features)
            queries = queries + residual
            
            residual = queries
            queries = layer['norm2'](queries)
            queries = layer['ff'](queries) + residual
        
        return queries


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "num_vision_tokens": 576, "vision_dim": 1024, "output_dim": 4096, "num_queries": 64},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("vision", "7_Resampler")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    vision_features = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_vision_tokens"], p["vision_dim"]), dtype=dtype, device=device)
    return [vision_features]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["vision_dim"], p["output_dim"], p["num_queries"]]
