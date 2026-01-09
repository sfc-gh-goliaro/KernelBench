import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class Model(nn.Module):
    """
    Fused Vocabulary-Parallel Cross-Entropy Loss
    
    Used by: Megatron-LM, vLLM, DeepSpeed for tensor-parallel LLM training
    
    Computes cross-entropy loss when logits are split across tensor parallel ranks.
    This fuses: (1) max reduction across ranks, (2) sum(exp) reduction,
    (3) loss computation without materializing full logits.
    
    Inspired by Megatron-LM's fused_vocab_parallel_cross_entropy.
    
    Shapes:
        Input logits: (batch_size * seq_len, vocab_size // tp_size) - local partition
        Input labels: (batch_size * seq_len,) - global labels
        Output: scalar loss
    """
    
    def __init__(self, vocab_size: int, tp_size: int = 1):
        """
        Initialize vocab-parallel cross-entropy.
        
        Args:
            vocab_size: Total vocabulary size
            tp_size: Tensor parallel size (for simulation)
        """
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.tp_size = tp_size
        self.vocab_per_partition = vocab_size // tp_size
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, tp_rank: int = 0) -> torch.Tensor:
        """
        Compute vocab-parallel cross-entropy loss.
        
        Args:
            logits: Local logits partition (batch * seq, vocab_size // tp_size)
            labels: Global target labels (batch * seq,)
            tp_rank: Tensor parallel rank (for determining label ownership)
            
        Returns:
            Scalar cross-entropy loss
        """
        # Determine vocab range for this partition
        vocab_start = tp_rank * self.vocab_per_partition
        vocab_end = vocab_start + self.vocab_per_partition
        
        # Step 1: Calculate local max for numerical stability
        local_max = logits.max(dim=-1, keepdim=True).values
        # In real TP: all_reduce max across ranks
        
        # Step 2: Subtract max and compute exp
        logits_stable = logits - local_max
        exp_logits = torch.exp(logits_stable)
        
        # Step 3: Sum of exp (would be all_reduced in real TP)
        sum_exp = exp_logits.sum(dim=-1, keepdim=True)
        
        # Step 4: Get predicted logits for target labels
        # Create mask for labels in this partition
        label_mask = (labels >= vocab_start) & (labels < vocab_end)
        local_labels = labels - vocab_start
        local_labels = local_labels.clamp(0, self.vocab_per_partition - 1)
        
        # Gather predicted logits for targets
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        predicted_logits = logits_stable[batch_indices, local_labels]
        
        # Zero out predictions for labels not in this partition
        predicted_logits = predicted_logits * label_mask.float()
        # In real TP: all_reduce sum predicted_logits
        
        # Step 5: Compute cross-entropy loss
        # loss = -predicted_logit + log(sum_exp)
        log_sum_exp = torch.log(sum_exp).squeeze(-1)
        loss = -predicted_logits + log_sum_exp
        
        return loss.mean()


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
vocab_size = 32000
tp_size = 1  # Simulated tensor parallel size

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    # Simulate local partition of logits
    vocab_per_rank = vocab_size // tp_size
    logits = torch.randn(batch_size * seq_length, vocab_per_rank, device='cuda')
    labels = torch.randint(0, vocab_size, (batch_size * seq_length,), device='cuda')
    return [logits, labels]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [vocab_size, tp_size]

