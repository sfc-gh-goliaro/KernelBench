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
    Stripmining Attention for Memory-Efficient Computation.
    
    Processes attention in strips/tiles to minimize peak memory usage
    while maintaining numerical precision.
    
    Based on memory-efficient attention implementations
    """
    def __init__(self, num_heads, head_dim, query_chunk_size=512, key_chunk_size=1024):
        """
        :param num_heads: Number of attention heads
        :param head_dim: Dimension per head
        :param query_chunk_size: Size of query chunks
        :param key_chunk_size: Size of key chunks
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.query_chunk_size = query_chunk_size
        self.key_chunk_size = key_chunk_size
        self.scale = head_dim ** -0.5
    
    def _summarize_chunk(self, query, key, value, mask=None):
        """
        Compute attention for a single query-key chunk.
        Returns weighted values and log-sum-exp for later normalization.
        """
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_scores = attn_scores + mask
        
        # Compute max for numerical stability
        chunk_max = attn_scores.max(dim=-1, keepdim=True).values
        
        # Compute exp(scores - max)
        exp_scores = torch.exp(attn_scores - chunk_max)
        
        # Sum of exponentials
        exp_sum = exp_scores.sum(dim=-1, keepdim=True)
        
        # Weighted values (not yet normalized)
        weighted_values = torch.matmul(exp_scores, value)
        
        return weighted_values, chunk_max.squeeze(-1), exp_sum.squeeze(-1)
    
    def forward(self, q, k, v, attention_mask=None):
        """
        Forward pass with stripmined attention computation.
        
        :param q: Query tensor (batch, num_heads, seq_q, head_dim)
        :param k: Key tensor (batch, num_heads, seq_k, head_dim)
        :param v: Value tensor (batch, num_heads, seq_k, head_dim)
        :param attention_mask: Optional attention mask
        :return: Attention output
        """
        batch_size, num_heads, seq_q, head_dim = q.shape
        seq_k = k.shape[2]
        device = q.device
        
        # Chunk sizes
        q_chunk = min(self.query_chunk_size, seq_q)
        k_chunk = min(self.key_chunk_size, seq_k)
        
        num_q_chunks = (seq_q + q_chunk - 1) // q_chunk
        num_k_chunks = (seq_k + k_chunk - 1) // k_chunk
        
        # Output accumulator
        output = torch.zeros_like(q)
        
        # Process query chunks
        for q_idx in range(num_q_chunks):
            q_start = q_idx * q_chunk
            q_end = min(q_start + q_chunk, seq_q)
            q_block = q[:, :, q_start:q_end]
            
            # Running statistics for this query block
            running_max = torch.full(
                (batch_size, num_heads, q_end - q_start), 
                float('-inf'), device=device
            )
            running_sum = torch.zeros(
                batch_size, num_heads, q_end - q_start, 
                device=device
            )
            running_output = torch.zeros(
                batch_size, num_heads, q_end - q_start, head_dim,
                device=device
            )
            
            # Process key chunks for this query block
            for k_idx in range(num_k_chunks):
                k_start = k_idx * k_chunk
                k_end = min(k_start + k_chunk, seq_k)
                
                k_block = k[:, :, k_start:k_end]
                v_block = v[:, :, k_start:k_end]
                
                # Get mask for this chunk if provided
                chunk_mask = None
                if attention_mask is not None:
                    chunk_mask = attention_mask[:, :, q_start:q_end, k_start:k_end]
                
                # Compute chunk attention
                chunk_output, chunk_max, chunk_sum = self._summarize_chunk(
                    q_block, k_block, v_block, chunk_mask
                )
                
                # Online softmax update
                # Update max
                new_max = torch.maximum(running_max, chunk_max)
                
                # Rescale old values
                old_scale = torch.exp(running_max - new_max)
                new_scale = torch.exp(chunk_max - new_max)
                
                # Update running output
                running_output = (
                    running_output * old_scale.unsqueeze(-1) + 
                    chunk_output * new_scale.unsqueeze(-1)
                )
                
                # Update running sum
                running_sum = running_sum * old_scale + chunk_sum * new_scale
                
                # Update running max
                running_max = new_max
            
            # Normalize output for this query block
            output[:, :, q_start:q_end] = running_output / running_sum.unsqueeze(-1).clamp(min=1e-9)
        
        return output


# Test parameters
batch_size = 4
num_heads = 16
seq_len = 4096
head_dim = 64
query_chunk_size = 256
key_chunk_size = 512

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    return [q, k, v]

def get_init_inputs():
    return [num_heads, head_dim, query_chunk_size, key_chunk_size]

