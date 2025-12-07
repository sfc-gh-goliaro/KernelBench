import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Flash Attention-style Chunked Computation.
    
    Computes attention in blocks/chunks to reduce memory usage
    from O(N^2) to O(N). This is a reference implementation 
    demonstrating the algorithmic structure.
    
    Based on: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
    """
    def __init__(self, num_heads, head_dim, block_size=64, causal=True):
        """
        :param num_heads: Number of attention heads
        :param head_dim: Dimension per head
        :param block_size: Size of blocks for chunked computation
        :param causal: Whether to apply causal masking
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.causal = causal
        self.scale = head_dim ** -0.5
    
    def forward(self, q, k, v, attention_mask=None):
        """
        Forward pass using tiled/chunked attention computation.
        
        :param q: Query tensor (batch, num_heads, seq_q, head_dim)
        :param k: Key tensor (batch, num_heads, seq_k, head_dim)
        :param v: Value tensor (batch, num_heads, seq_k, head_dim)
        :param attention_mask: Optional attention mask
        :return: Attention output (batch, num_heads, seq_q, head_dim)
        """
        batch_size, num_heads, seq_q, head_dim = q.shape
        seq_k = k.shape[2]
        
        # Determine block sizes
        block_q = min(self.block_size, seq_q)
        block_k = min(self.block_size, seq_k)
        
        num_blocks_q = (seq_q + block_q - 1) // block_q
        num_blocks_k = (seq_k + block_k - 1) // block_k
        
        # Initialize output and running statistics
        output = torch.zeros_like(q)
        row_max = torch.full((batch_size, num_heads, seq_q), float('-inf'), device=q.device)
        row_sum = torch.zeros(batch_size, num_heads, seq_q, device=q.device)
        
        # Process in blocks
        for i in range(num_blocks_q):
            q_start = i * block_q
            q_end = min(q_start + block_q, seq_q)
            q_block = q[:, :, q_start:q_end]  # (batch, heads, block_q, head_dim)
            
            # Running max and sum for this block
            block_max = torch.full((batch_size, num_heads, q_end - q_start), 
                                  float('-inf'), device=q.device)
            block_sum = torch.zeros(batch_size, num_heads, q_end - q_start, device=q.device)
            block_out = torch.zeros(batch_size, num_heads, q_end - q_start, head_dim, device=q.device)
            
            for j in range(num_blocks_k):
                k_start = j * block_k
                k_end = min(k_start + block_k, seq_k)
                
                # Causal check: skip future blocks
                if self.causal and k_start > q_end:
                    continue
                
                k_block = k[:, :, k_start:k_end]
                v_block = v[:, :, k_start:k_end]
                
                # Compute attention scores for this block
                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale
                
                # Apply causal mask
                if self.causal:
                    q_positions = torch.arange(q_start, q_end, device=q.device)
                    k_positions = torch.arange(k_start, k_end, device=q.device)
                    causal_mask = q_positions.unsqueeze(1) < k_positions.unsqueeze(0)
                    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                
                # Apply attention mask if provided
                if attention_mask is not None:
                    mask_block = attention_mask[:, :, q_start:q_end, k_start:k_end]
                    scores = scores + mask_block
                
                # Online softmax update
                # New max
                block_max_new = torch.maximum(block_max, scores.max(dim=-1).values)
                
                # Rescale old values
                scale_old = torch.exp(block_max - block_max_new).unsqueeze(-1)
                block_out = block_out * scale_old
                block_sum = block_sum * scale_old.squeeze(-1)
                
                # Add new values
                scores_scaled = torch.exp(scores - block_max_new.unsqueeze(-1))
                block_out = block_out + torch.matmul(scores_scaled, v_block)
                block_sum = block_sum + scores_scaled.sum(dim=-1)
                
                block_max = block_max_new
            
            # Normalize this block
            block_out = block_out / block_sum.unsqueeze(-1).clamp(min=1e-9)
            
            # Update output
            output[:, :, q_start:q_end] = block_out
            row_max[:, :, q_start:q_end] = block_max
            row_sum[:, :, q_start:q_end] = block_sum
        
        return output


# Test parameters
batch_size = 8
num_heads = 32
seq_len = 2048
head_dim = 64
block_size = 64

def get_inputs():
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    return [q, k, v]

def get_init_inputs():
    return [num_heads, head_dim, block_size]

