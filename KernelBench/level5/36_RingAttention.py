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
    Ring Attention for Distributed Long Context.
    
    Distributes sequence across devices in a ring topology,
    computing attention through collective communication.
    
    Based on: "Ring Attention with Blockwise Transformers for Near-Infinite Context"
    
    Note: This is a single-GPU simulation of the algorithm.
    """
    def __init__(self, num_heads, head_dim, ring_size=4, block_size=256, causal=True):
        """
        :param num_heads: Number of attention heads
        :param head_dim: Dimension per head
        :param ring_size: Number of simulated ring partitions
        :param block_size: Size of each block
        :param causal: Whether to apply causal masking
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ring_size = ring_size
        self.block_size = block_size
        self.causal = causal
        self.scale = head_dim ** -0.5
    
    def _ring_pass(self, tensor, step):
        """
        Simulate ring-passing of tensors.
        In actual distributed setting, this would be a collective communication.
        
        :param tensor: Tensor to pass
        :param step: Current ring step
        :return: Tensor from previous device in ring
        """
        # In simulation, we just return the appropriate partition
        # based on the step. In real distributed, this would be send/recv.
        return tensor
    
    def forward(self, q, k, v):
        """
        Forward pass simulating ring attention.
        
        :param q: Query tensor (batch, num_heads, seq_len, head_dim)
        :param k: Key tensor (batch, num_heads, seq_len, head_dim)
        :param v: Value tensor (batch, num_heads, seq_len, head_dim)
        :return: Attention output
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        
        # Partition sequence into ring_size blocks
        block_size = seq_len // self.ring_size
        
        # Reshape into blocks
        q_blocks = q.view(batch_size, num_heads, self.ring_size, block_size, head_dim)
        k_blocks = k.view(batch_size, num_heads, self.ring_size, block_size, head_dim)
        v_blocks = v.view(batch_size, num_heads, self.ring_size, block_size, head_dim)
        
        # Initialize output and running statistics for each block
        output_blocks = torch.zeros_like(q_blocks)
        row_max = torch.full((batch_size, num_heads, self.ring_size, block_size), 
                            float('-inf'), device=device)
        row_sum = torch.zeros(batch_size, num_heads, self.ring_size, block_size, device=device)
        
        # Current KV blocks being passed around the ring
        current_k = k_blocks
        current_v = v_blocks
        
        # Ring attention: each block computes attention with all KV blocks
        for step in range(self.ring_size):
            # Source block index for KV (where the current KV came from)
            kv_source = (torch.arange(self.ring_size, device=device) - step) % self.ring_size
            
            # Compute attention between each Q block and current KV
            for q_idx in range(self.ring_size):
                k_idx = kv_source[q_idx].item()
                
                # Causal check
                if self.causal and k_idx > q_idx:
                    continue
                
                q_block = q_blocks[:, :, q_idx]  # (batch, heads, block, head_dim)
                k_block = current_k[:, :, q_idx]  # KV for this position
                v_block = current_v[:, :, q_idx]
                
                # Compute attention scores
                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale
                
                # Apply causal mask within block if same block
                if self.causal and k_idx == q_idx:
                    mask = torch.triu(torch.ones(block_size, block_size, device=device), diagonal=1)
                    scores = scores.masked_fill(mask.bool(), float('-inf'))
                
                # Online softmax update
                curr_max = row_max[:, :, q_idx]
                curr_sum = row_sum[:, :, q_idx]
                curr_out = output_blocks[:, :, q_idx]
                
                new_max = torch.maximum(curr_max, scores.max(dim=-1).values)
                
                # Rescale
                scale_old = torch.exp(curr_max - new_max).unsqueeze(-1)
                scale_new = torch.exp(scores - new_max.unsqueeze(-1))
                
                curr_out = curr_out * scale_old
                curr_sum = curr_sum * scale_old.squeeze(-1)
                
                curr_out = curr_out + torch.matmul(scale_new, v_block)
                curr_sum = curr_sum + scale_new.sum(dim=-1)
                
                output_blocks[:, :, q_idx] = curr_out
                row_max[:, :, q_idx] = new_max
                row_sum[:, :, q_idx] = curr_sum
            
            # Simulate ring pass: shift KV blocks
            current_k = torch.roll(current_k, shifts=1, dims=2)
            current_v = torch.roll(current_v, shifts=1, dims=2)
        
        # Normalize output
        output_blocks = output_blocks / row_sum.unsqueeze(-1).clamp(min=1e-9)
        
        # Reshape back
        output = output_blocks.view(batch_size, num_heads, seq_len, head_dim)
        
        return output


# Test parameters
batch_size = 4
num_heads = 16
seq_len = 4096  # Should be divisible by ring_size
head_dim = 64
ring_size = 4

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
    return [num_heads, head_dim, ring_size]

