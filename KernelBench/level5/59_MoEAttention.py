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
    Mixture of Experts Attention.
    
    Applies MoE to attention heads, allowing dynamic routing
    of different attention patterns. Used in some DeepSeek models.
    
    Based on: MoE attention variants in recent LLMs
    """
    def __init__(self, dim, num_heads, num_experts=4, top_k=2, 
                 head_dim=None, dropout=0.0):
        """
        :param dim: Model dimension
        :param num_heads: Number of base attention heads
        :param num_experts: Number of attention experts
        :param top_k: Number of experts to route to
        :param head_dim: Head dimension (default: dim // num_heads)
        :param dropout: Dropout rate
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Router
        self.router = nn.Linear(dim, num_experts, bias=False)
        
        # Expert attention heads (each expert has full set of projections)
        self.q_experts = nn.Parameter(
            torch.randn(num_experts, dim, num_heads * self.head_dim) * 0.02
        )
        self.k_experts = nn.Parameter(
            torch.randn(num_experts, dim, num_heads * self.head_dim) * 0.02
        )
        self.v_experts = nn.Parameter(
            torch.randn(num_experts, dim, num_heads * self.head_dim) * 0.02
        )
        self.o_experts = nn.Parameter(
            torch.randn(num_experts, num_heads * self.head_dim, dim) * 0.02
        )
        
        # Shared components
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attention_mask=None):
        """
        Forward pass with MoE attention.
        
        :param x: Input tensor (batch, seq_len, dim)
        :param attention_mask: Optional attention mask
        :return: Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Normalize
        x_norm = self.norm(x)
        
        # Route tokens to experts
        router_logits = self.router(x_norm)  # (batch, seq, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Get top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # Normalize
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each expert
        for k in range(self.top_k):
            for e in range(self.num_experts):
                # Find tokens routed to this expert at position k
                mask = (top_k_indices[:, :, k] == e)  # (batch, seq)
                if not mask.any():
                    continue
                
                # Get weights for this expert
                weights = top_k_probs[:, :, k] * mask.float()  # (batch, seq)
                
                # Compute Q, K, V using this expert's projections
                q = F.linear(x_norm, self.q_experts[e].t())
                k_proj = F.linear(x_norm, self.k_experts[e].t())
                v = F.linear(x_norm, self.v_experts[e].t())
                
                # Reshape for attention
                q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                k_proj = k_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                
                # Attention
                attn_scores = torch.matmul(q, k_proj.transpose(-2, -1)) * self.scale
                
                if attention_mask is not None:
                    attn_scores = attn_scores + attention_mask
                
                attn_probs = F.softmax(attn_scores, dim=-1)
                attn_probs = self.dropout(attn_probs)
                
                attn_out = torch.matmul(attn_probs, v)
                attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
                
                # Output projection
                expert_out = F.linear(attn_out, self.o_experts[e].t())
                
                # Weight by routing probability
                output = output + weights.unsqueeze(-1) * expert_out
        
        # Residual
        return x + output


# Test parameters
batch_size = 8
seq_len = 512
dim = 768
num_heads = 12
num_experts = 4
top_k = 2

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    return [torch.randn(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, num_heads, num_experts, top_k]

