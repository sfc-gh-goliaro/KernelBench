import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class VLMCrossAttentionLayer(nn.Module):
    """Cross-attention from text to visual features."""
    def __init__(self, text_hidden_size: int, vision_hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = text_hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q from text, K/V from vision
        self.q_proj = nn.Linear(text_hidden_size, text_hidden_size, bias=False)
        self.k_proj = nn.Linear(vision_hidden_size, text_hidden_size, bias=False)
        self.v_proj = nn.Linear(vision_hidden_size, text_hidden_size, bias=False)
        self.o_proj = nn.Linear(text_hidden_size, text_hidden_size, bias=False)

    def forward(self, text_hidden: torch.Tensor, vision_hidden: torch.Tensor) -> torch.Tensor:
        batch_size, text_len, _ = text_hidden.shape
        vision_len = vision_hidden.shape[1]
        
        q = self.q_proj(text_hidden).view(batch_size, text_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(vision_hidden).view(batch_size, vision_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(vision_hidden).view(batch_size, vision_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, text_len, -1)
        return self.o_proj(out)


class GatedMLP(nn.Module):
    """SwiGLU MLP."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Model(nn.Module):
    """
    VLM Cross-Attention Block
    
    The core visual-text fusion block in Vision-Language Models.
    Used by: LLaVA, Qwen-VL, InternVL, Idefics, and other VLMs
    
    Architecture:
        text_hidden -> RMSNorm -> Cross-Attention (to vision) -> + residual
                    -> RMSNorm -> MLP -> + residual
    
    This block enables the language model to attend to visual features.
    In practice, these blocks may be interleaved with regular self-attention
    layers or used as a perceiver/resampler module.
    """
    def __init__(self, text_hidden_size: int, vision_hidden_size: int, 
                 num_heads: int, intermediate_size: int):
        super().__init__()
        
        # Cross-attention
        self.cross_attn_norm = RMSNorm(text_hidden_size)
        self.cross_attn = VLMCrossAttentionLayer(text_hidden_size, vision_hidden_size, num_heads)
        
        # MLP
        self.mlp_norm = RMSNorm(text_hidden_size)
        self.mlp = GatedMLP(text_hidden_size, intermediate_size)

    def forward(self, text_hidden: torch.Tensor, vision_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_hidden: Text model hidden states (batch, text_seq_len, text_hidden_size)
            vision_hidden: Vision encoder output (batch, num_visual_tokens, vision_hidden_size)
        
        Returns:
            Updated text hidden states with visual information
        """
        # Cross-attention to vision
        residual = text_hidden
        text_hidden = self.cross_attn_norm(text_hidden)
        text_hidden = self.cross_attn(text_hidden, vision_hidden)
        text_hidden = residual + text_hidden
        
        # MLP
        residual = text_hidden
        text_hidden = self.mlp_norm(text_hidden)
        text_hidden = self.mlp(text_hidden)
        text_hidden = residual + text_hidden
        
        return text_hidden


# Benchmark configuration (LLaVA-style dimensions)
batch_size = 4
text_seq_len = 512
num_visual_tokens = 576  # 24x24 patches for 336x336 image at 14px
text_hidden_size = 4096
vision_hidden_size = 1024
num_heads = 32
intermediate_size = 14336

def get_inputs():
    text_hidden = torch.randn(batch_size, text_seq_len, text_hidden_size)
    vision_hidden = torch.randn(batch_size, num_visual_tokens, vision_hidden_size)
    return [text_hidden, vision_hidden]

def get_init_inputs():
    return [text_hidden_size, vision_hidden_size, num_heads, intermediate_size]

