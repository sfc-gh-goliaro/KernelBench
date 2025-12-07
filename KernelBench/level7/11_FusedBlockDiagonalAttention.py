import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Fused Block-Diagonal Masked Attention.
    
    Computes attention with block-diagonal structure, where each block
    is a separate context. Used in:
    1. Ring Attention: Each rank processes a block
    2. Document batching: Different documents have separate contexts
    3. Efficient prefix caching
    
    The fused kernel avoids materializing the full N×N attention matrix.
    
    Reference: Ring Attention, FlexAttention block masks
    """
    def __init__(self, dim, num_heads, head_dim=None, block_size=512):
        """
        :param dim: Model dimension
        :param num_heads: Number of attention heads
        :param head_dim: Head dimension
        :param block_size: Size of each attention block
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.block_size = block_size
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
    
    def _block_diagonal_attention(self, q, k, v, block_boundaries):
        """
        Block-diagonal attention computation.
        
        :param q: Query (batch, heads, seq, head_dim)
        :param k: Key (batch, heads, seq, head_dim)
        :param v: Value (batch, heads, seq, head_dim)
        :param block_boundaries: List of (start, end) for each block
        :return: Attention output
        """
        batch, heads, seq_len, head_dim = q.shape
        device = q.device
        
        output = torch.zeros_like(q)
        
        # === FUSED KERNEL ===
        # Process each block independently
        for start, end in block_boundaries:
            if end <= start:
                continue
            
            block_q = q[:, :, start:end]
            block_k = k[:, :, start:end]
            block_v = v[:, :, start:end]
            
            # Standard attention within block
            attn_scores = torch.matmul(block_q, block_k.transpose(-2, -1)) * self.scale
            attn_probs = F.softmax(attn_scores, dim=-1)
            block_out = torch.matmul(attn_probs, block_v)
            
            output[:, :, start:end] = block_out
        
        return output
    
    def forward(self, x, block_boundaries=None):
        """
        Block-diagonal attention forward.
        
        :param x: Input (batch, seq, dim)
        :param block_boundaries: List of (start, end) tuples for blocks
                                If None, uses uniform blocks of block_size
        :return: Output (batch, seq, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Default: uniform blocks
        if block_boundaries is None:
            block_boundaries = []
            for start in range(0, seq_len, self.block_size):
                end = min(start + self.block_size, seq_len)
                block_boundaries.append((start, end))
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Block-diagonal attention
        output = self._block_diagonal_attention(q, k, v, block_boundaries)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(output)
        
        return output


# Ring Attention variant with KV passing
class RingBlockAttention(nn.Module):
    """
    Ring Attention with block-diagonal structure.
    
    Each "rank" processes its local block and passes KV to neighbors.
    """
    def __init__(self, dim, num_heads, num_blocks, head_dim=None):
        super(RingBlockAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
    
    def _ring_step_attention(self, q_block, k_blocks, v_blocks, block_idx, causal=True):
        """
        Single ring step: compute attention for one Q block against multiple KV blocks.
        
        Uses online softmax for memory efficiency.
        """
        batch, heads, block_len, head_dim = q_block.shape
        device = q_block.device
        
        running_max = torch.full((batch, heads, block_len), float('-inf'), device=device)
        running_sum = torch.zeros(batch, heads, block_len, device=device)
        output = torch.zeros_like(q_block)
        
        for kv_idx, (k, v) in enumerate(zip(k_blocks, v_blocks)):
            # Compute attention scores
            scores = torch.matmul(q_block, k.transpose(-2, -1)) * self.scale
            
            # Apply causal mask if this block is from current or future position
            if causal and kv_idx >= block_idx:
                if kv_idx == block_idx:
                    # Within-block causal mask
                    mask = torch.triu(torch.ones(block_len, k.shape[2], device=device), diagonal=1).bool()
                    scores = scores.masked_fill(mask, float('-inf'))
                else:
                    # Future block - mask all
                    continue
            
            # Online softmax update
            block_max = scores.max(dim=-1).values
            new_max = torch.maximum(running_max, block_max)
            
            old_scale = torch.exp(running_max - new_max).unsqueeze(-1)
            output = output * old_scale
            running_sum = running_sum * old_scale.squeeze(-1)
            
            exp_scores = torch.exp(scores - new_max.unsqueeze(-1))
            output = output + torch.matmul(exp_scores, v)
            running_sum = running_sum + exp_scores.sum(dim=-1)
            
            running_max = new_max
        
        # Normalize
        output = output / running_sum.unsqueeze(-1).clamp(min=1e-9)
        
        return output
    
    def forward(self, x_blocks):
        """
        Ring attention forward.
        
        :param x_blocks: List of input blocks
        :return: List of output blocks
        """
        # Project all blocks
        q_blocks = [self.q_proj(x).view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2) 
                   for x in x_blocks]
        k_blocks = [self.k_proj(x).view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2) 
                   for x in x_blocks]
        v_blocks = [self.v_proj(x).view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2) 
                   for x in x_blocks]
        
        # Compute attention for each block
        outputs = []
        for block_idx in range(len(x_blocks)):
            out = self._ring_step_attention(q_blocks[block_idx], k_blocks, v_blocks, block_idx)
            out = out.transpose(1, 2).contiguous().view(out.shape[0], -1, self.dim)
            out = self.out_proj(out)
            outputs.append(out)
        
        return outputs


# Test parameters
batch_size = 4
seq_len = 2048
dim = 2048
num_heads = 16
block_size = 512

def get_inputs():
    x = torch.randn(batch_size, seq_len, dim)
    # Create block boundaries for different sequence lengths
    block_boundaries = [(0, 512), (512, 1024), (1024, 1536), (1536, 2048)]
    return [x, block_boundaries]

def get_init_inputs():
    return [dim, num_heads, None, block_size]

