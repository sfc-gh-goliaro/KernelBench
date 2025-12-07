import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    KV Cache Update Operations for autoregressive inference.
    
    Manages the key-value cache for efficient incremental decoding,
    supporting dynamic length sequences and batch operations.
    """
    def __init__(self, num_layers, num_heads, head_dim, max_batch_size, max_seq_len):
        """
        :param num_layers: Number of transformer layers
        :param num_heads: Number of attention heads
        :param head_dim: Dimension per head
        :param max_batch_size: Maximum batch size
        :param max_seq_len: Maximum sequence length
        """
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        
        # Pre-allocated KV cache for all layers
        # Shape: (num_layers, 2, max_batch, num_heads, max_seq, head_dim)
        # 2 is for K and V
        self.register_buffer('cache', 
            torch.zeros(num_layers, 2, max_batch_size, num_heads, max_seq_len, head_dim))
        
        # Current sequence lengths per batch item
        self.register_buffer('seq_lens', torch.zeros(max_batch_size, dtype=torch.long))
        
    def reset(self, batch_indices=None):
        """
        Reset cache for specified batch items or all.
        
        :param batch_indices: Optional batch indices to reset
        """
        if batch_indices is None:
            self.cache.zero_()
            self.seq_lens.zero_()
        else:
            self.cache[:, :, batch_indices].zero_()
            self.seq_lens[batch_indices] = 0
    
    def update(self, layer_idx, new_k, new_v, batch_indices=None):
        """
        Append new K, V to the cache.
        
        :param layer_idx: Which layer's cache to update
        :param new_k: New keys (batch, num_heads, seq_new, head_dim)
        :param new_v: New values (batch, num_heads, seq_new, head_dim)
        :param batch_indices: Optional batch indices (for continuous batching)
        :return: Updated (keys, values) including history
        """
        batch_size, _, seq_new, _ = new_k.shape
        
        if batch_indices is None:
            batch_indices = torch.arange(batch_size, device=new_k.device)
        
        # Get current lengths
        current_lens = self.seq_lens[batch_indices]
        
        # Update cache
        for i, b in enumerate(batch_indices):
            start = current_lens[i].item()
            end = start + seq_new
            
            if end > self.max_seq_len:
                # Shift cache if overflow (sliding window)
                shift = end - self.max_seq_len
                self.cache[layer_idx, 0, b, :, :-shift] = self.cache[layer_idx, 0, b, :, shift:]
                self.cache[layer_idx, 1, b, :, :-shift] = self.cache[layer_idx, 1, b, :, shift:]
                start = self.max_seq_len - seq_new
                end = self.max_seq_len
            
            self.cache[layer_idx, 0, b, :, start:end] = new_k[i]
            self.cache[layer_idx, 1, b, :, start:end] = new_v[i]
        
        # Update sequence lengths
        self.seq_lens[batch_indices] = torch.clamp(current_lens + seq_new, max=self.max_seq_len)
        
        # Return cached K, V up to current length
        max_len = self.seq_lens[batch_indices].max().item()
        cached_k = self.cache[layer_idx, 0, batch_indices, :, :max_len]
        cached_v = self.cache[layer_idx, 1, batch_indices, :, :max_len]
        
        return cached_k, cached_v
    
    def get_cache(self, layer_idx, batch_indices=None):
        """
        Get current cache for a layer.
        
        :param layer_idx: Layer index
        :param batch_indices: Optional batch indices
        :return: Tuple of (keys, values)
        """
        if batch_indices is None:
            batch_indices = torch.arange(self.max_batch_size, device=self.cache.device)
        
        max_len = self.seq_lens[batch_indices].max().item()
        if max_len == 0:
            return None, None
        
        return (self.cache[layer_idx, 0, batch_indices, :, :max_len],
                self.cache[layer_idx, 1, batch_indices, :, :max_len])
    
    def forward(self, layer_idx, new_k, new_v, batch_indices=None):
        """
        Forward pass: update cache and return full K, V.
        
        :param layer_idx: Layer index
        :param new_k: New keys
        :param new_v: New values
        :param batch_indices: Optional batch indices
        :return: Full (keys, values) including new entries
        """
        return self.update(layer_idx, new_k, new_v, batch_indices)


# Test parameters
num_layers = 32
num_heads = 32
head_dim = 128
max_batch_size = 32
max_seq_len = 2048
batch_size = 8
layer_idx = 0

def get_inputs():
    new_k = torch.randn(batch_size, num_heads, 1, head_dim)  # Single token
    new_v = torch.randn(batch_size, num_heads, 1, head_dim)
    return [layer_idx, new_k, new_v]

def get_init_inputs():
    return [num_layers, num_heads, head_dim, max_batch_size, max_seq_len]

