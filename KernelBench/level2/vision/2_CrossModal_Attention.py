import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Fused Cross-Modal Attention
    
    Used by: LLaVA, Qwen-VL, InternVL, all Vision-Language Models
    
    Fuses the cross-attention mechanism between visual and text modalities.
    Combines visual feature projection, cross-attention, and output projection.
    
    Shapes:
        Text input: (batch_size, text_seq_len, hidden_size)
        Visual input: (batch_size, num_visual_tokens, visual_dim)
        Output: (batch_size, text_seq_len, hidden_size)
    """
    
    def __init__(self, hidden_size: int, visual_dim: int, num_heads: int = 16,
                 num_visual_tokens: int = 576):
        """
        Initialize fused cross-modal attention.
        
        Args:
            hidden_size: Text model hidden dimension
            visual_dim: Visual encoder output dimension
            num_heads: Number of attention heads
            num_visual_tokens: Number of visual tokens (e.g., 576 for 336x336 @ 14px)
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.visual_dim = visual_dim
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Visual projection to match text hidden size
        self.visual_proj = nn.Linear(visual_dim, hidden_size)
        
        # Cross-attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, text_hidden: torch.Tensor, visual_features: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Fused cross-modal attention.
        
        Args:
            text_hidden: Text hidden states (batch, text_len, hidden_size)
            visual_features: Visual features (batch, num_visual, visual_dim)
            attention_mask: Optional attention mask
            
        Returns:
            Cross-attended text features (batch, text_len, hidden_size)
        """
        batch_size, text_len, _ = text_hidden.shape
        num_visual = visual_features.shape[1]
        
        # Project visual features to text space
        visual_hidden = self.visual_proj(visual_features)
        
        # Compute Q from text, K/V from visual
        q = self.q_proj(text_hidden)
        k = self.k_proj(visual_hidden)
        v = self.v_proj(visual_hidden)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, text_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_visual, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_visual, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, text_len, self.hidden_size)
        
        return self.o_proj(attn_output)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 4
text_seq_len = 512
num_visual_tokens = 576  # 24x24 patches
hidden_size = 4096
visual_dim = 1024
num_heads = 32

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    text_hidden = torch.randn(batch_size, text_seq_len, hidden_size, device='cuda')
    visual_features = torch.randn(batch_size, num_visual_tokens, visual_dim, device='cuda')
    return [text_hidden, visual_features]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, visual_dim, num_heads, num_visual_tokens]

