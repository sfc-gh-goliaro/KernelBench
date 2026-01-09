import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Tree Attention (Speculative Decoding)
    
    Used by: EAGLE, Medusa, SpecInfer
    
    Attention with tree-structured causal mask for parallel verification
    of draft token trees in speculative decoding.
    
    Shapes:
        query: (batch, num_draft_tokens, hidden_size)
        tree_mask: (num_draft_tokens, num_draft_tokens) tree structure
        Output: (batch, num_draft_tokens, hidden_size)
    """
    
    def __init__(self, hidden_size: int, num_heads: int):
        """
        Initialize tree attention.
        
        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                tree_mask: torch.Tensor) -> torch.Tensor:
        """
        Tree attention forward pass.
        
        Args:
            query: Query tensor (batch, num_draft, hidden_size)
            key: Key tensor (batch, context_len + num_draft, hidden_size)
            value: Value tensor (batch, context_len + num_draft, hidden_size)
            tree_mask: Tree structure mask (num_draft, context_len + num_draft)
                       1 where attention is allowed, 0 where blocked
            
        Returns:
            Output tensor (batch, num_draft, hidden_size)
        """
        batch_size, num_draft, _ = query.shape
        total_len = key.shape[1]
        
        # Project
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape to (batch, num_heads, seq, head_dim)
        q = q.view(batch_size, num_draft, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, total_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, total_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply tree mask (1 = attend, 0 = block)
        # tree_mask: (num_draft, total_len) -> (1, 1, num_draft, total_len)
        mask = tree_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_draft, self.hidden_size)
        
        return self.o_proj(attn_output)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
context_len = 2048
num_draft_tokens = 64  # Tree of draft tokens
hidden_size = 4096
num_heads = 32

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    query = torch.randn(batch_size, num_draft_tokens, hidden_size, device='cuda')
    key = torch.randn(batch_size, context_len + num_draft_tokens, hidden_size, device='cuda')
    value = torch.randn(batch_size, context_len + num_draft_tokens, hidden_size, device='cuda')
    
    # Create tree mask: each draft token can attend to context + its ancestors
    # For simplicity, use a lower triangular structure
    tree_mask = torch.tril(torch.ones(num_draft_tokens, context_len + num_draft_tokens, device='cuda'))
    tree_mask[:, :context_len] = 1  # All draft tokens can attend to full context
    
    return [query, key, value, tree_mask]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, num_heads]

