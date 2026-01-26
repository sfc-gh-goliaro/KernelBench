import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused Logits + Softmax + Top-K/Top-P Sampling
    
    Used by: vLLM, SGLang, TensorRT-LLM inference engines
    
    Fuses the entire sampling pipeline into a single optimized operation:
    1. Temperature scaling
    2. Top-K filtering
    3. Top-P (nucleus) filtering  
    4. Softmax normalization
    5. Multinomial sampling
    
    This avoids multiple kernel launches and intermediate tensor allocations
    compared to composing individual level1 operators.
    
    Shapes:
        Input: (batch_size, vocab_size) logits
        Output: (batch_size,) sampled token indices
    """
    
    def __init__(self, top_k: int = 50, top_p: float = 0.9, temperature: float = 1.0, min_p: float = 0.0):
        """
        Initialize fused sampling.
        
        Args:
            top_k: Number of top tokens to consider (0 = disabled)
            top_p: Cumulative probability threshold (1.0 = disabled)
            temperature: Sampling temperature
            min_p: Minimum probability threshold relative to top token
        """
        super(Model, self).__init__()
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.min_p = min_p
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Fused sampling from logits.
        
        Args:
            logits: Raw logits tensor (batch_size, vocab_size)
            
        Returns:
            Sampled token indices (batch_size,)
        """
        batch_size, vocab_size = logits.shape
        
        # Step 1: Temperature scaling
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Step 2: Top-K filtering
        if self.top_k > 0 and self.top_k < vocab_size:
            top_k_values, _ = torch.topk(logits, self.top_k, dim=-1)
            threshold = top_k_values[:, -1, None]
            logits = torch.where(logits < threshold, 
                               torch.full_like(logits, float('-inf')), logits)
        
        # Step 3: Min-P filtering (relative to top token)
        if self.min_p > 0.0:
            probs_for_minp = F.softmax(logits, dim=-1)
            max_prob = probs_for_minp.max(dim=-1, keepdim=True).values
            min_threshold = max_prob * self.min_p
            logits = torch.where(probs_for_minp < min_threshold,
                               torch.full_like(logits, float('-inf')), logits)
        
        # Step 4: Top-P (nucleus) filtering
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Find cutoff point
            sorted_mask = cumulative_probs > self.top_p
            sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
            sorted_mask[:, 0] = False
            
            # Scatter mask back
            mask = torch.zeros_like(sorted_mask)
            mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
            logits = logits.masked_fill(mask, float('-inf'))
        
        # Step 5: Softmax + Sampling
        probs = F.softmax(logits, dim=-1)
        
        # Handle edge case where all probs are 0 (all filtered)
        probs = torch.where(probs.sum(dim=-1, keepdim=True) == 0,
                           torch.ones_like(probs) / vocab_size, probs)
        
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        return sampled_tokens


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 64
vocab_size = 128000  # Llama-3 vocab size
top_k = 50
top_p = 0.9
temperature = 1.0

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    logits = torch.randn(batch_size, vocab_size, device='cuda')
    return [logits]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [top_k, top_p, temperature]

