import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused Linear + Cross Entropy Loss (The "Unsloth" Layer).
    
    Combines the final LM head projection with cross-entropy loss computation
    to avoid materializing the full logits tensor (vocab_size can be 100K+).
    
    Memory savings: O(batch * seq * vocab) -> O(batch * seq)
    
    This is critical for training large vocabulary models and was
    popularized by the Unsloth library for efficient fine-tuning.
    
    Reference: Unsloth, Cut Cross Entropy (Apple)
    """
    def __init__(self, hidden_dim, vocab_size, ignore_index=-100):
        """
        :param hidden_dim: Model hidden dimension
        :param vocab_size: Vocabulary size
        :param ignore_index: Label index to ignore in loss
        """
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        
        # LM head weight
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
    
    def _chunked_cross_entropy(self, hidden_states, labels, chunk_size=4096):
        """
        Compute cross entropy in chunks to reduce memory.
        
        :param hidden_states: (batch * seq, hidden_dim)
        :param labels: (batch * seq,)
        :param chunk_size: Vocabulary chunk size
        :return: Cross entropy loss
        """
        total_tokens = hidden_states.shape[0]
        device = hidden_states.device
        
        # Compute loss without materializing full logits
        loss = torch.tensor(0.0, device=device)
        valid_tokens = 0
        
        for i in range(0, self.vocab_size, chunk_size):
            # Get chunk of LM head weights
            chunk_end = min(i + chunk_size, self.vocab_size)
            weight_chunk = self.lm_head.weight[i:chunk_end]
            
            # Compute logits for this vocabulary chunk
            logits_chunk = F.linear(hidden_states, weight_chunk)
            
            # Find labels in this chunk
            labels_in_chunk = (labels >= i) & (labels < chunk_end) & (labels != self.ignore_index)
            
            if labels_in_chunk.any():
                # Adjust labels for chunk
                local_labels = labels[labels_in_chunk] - i
                local_hidden = hidden_states[labels_in_chunk]
                local_logits = F.linear(local_hidden, weight_chunk)
                
                # Need full row for proper softmax normalization
                # This is a simplified version - full impl needs running max
                chunk_loss = F.cross_entropy(local_logits, local_labels, reduction='sum')
                loss = loss + chunk_loss
                valid_tokens += labels_in_chunk.sum()
        
        return loss / valid_tokens.clamp(min=1)
    
    def forward(self, hidden_states, labels=None):
        """
        Fused linear + cross entropy forward.
        
        :param hidden_states: (batch, seq, hidden_dim)
        :param labels: (batch, seq) target labels
        :return: loss if labels provided, else logits
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        if labels is None:
            # Inference: just compute logits
            return self.lm_head(hidden_states)
        
        # Training: fused linear + cross entropy
        # Flatten
        hidden_flat = hidden_states.view(-1, self.hidden_dim)
        labels_flat = labels.view(-1)
        
        # === FUSED KERNEL APPROACH ===
        # In a true fused kernel, we compute:
        # 1. One row of logits at a time
        # 2. Compute partial softmax (with online normalization)
        # 3. Accumulate cross entropy loss
        # 4. Never materialize full logits tensor
        
        # Reference implementation (for correctness checking)
        logits = self.lm_head(hidden_flat)
        loss = F.cross_entropy(logits, labels_flat, ignore_index=self.ignore_index)
        
        return loss


# Memory-efficient variant with gradient checkpointing
class FusedLinearCrossEntropyChunked(nn.Module):
    """
    Chunked Fused Linear + Cross Entropy for extreme memory efficiency.
    
    Processes vocabulary in chunks with online log-sum-exp computation.
    """
    def __init__(self, hidden_dim, vocab_size, chunk_size=8192, ignore_index=-100):
        super(FusedLinearCrossEntropyChunked, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index
        
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
    
    def forward(self, hidden_states, labels):
        """
        Memory-efficient fused forward with chunking.
        
        Uses online softmax to avoid materializing full logits.
        """
        batch_size, seq_len, _ = hidden_states.shape
        hidden_flat = hidden_states.view(-1, self.hidden_dim)
        labels_flat = labels.view(-1)
        
        total_tokens = hidden_flat.shape[0]
        device = hidden_flat.device
        
        # Online max and sum for softmax normalization
        running_max = torch.full((total_tokens,), float('-inf'), device=device)
        running_sum = torch.zeros(total_tokens, device=device)
        target_logits = torch.zeros(total_tokens, device=device)
        
        for chunk_start in range(0, self.vocab_size, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.vocab_size)
            weight_chunk = self.lm_head.weight[chunk_start:chunk_end]
            
            # Compute chunk logits
            logits_chunk = F.linear(hidden_flat, weight_chunk)  # (tokens, chunk_size)
            
            # Online softmax update
            chunk_max = logits_chunk.max(dim=-1).values
            new_max = torch.maximum(running_max, chunk_max)
            
            # Rescale running sum
            running_sum = running_sum * torch.exp(running_max - new_max)
            
            # Add chunk contribution
            exp_logits = torch.exp(logits_chunk - new_max.unsqueeze(-1))
            running_sum = running_sum + exp_logits.sum(dim=-1)
            
            # Track target logits
            in_chunk = (labels_flat >= chunk_start) & (labels_flat < chunk_end)
            if in_chunk.any():
                local_idx = labels_flat[in_chunk] - chunk_start
                target_logits[in_chunk] = logits_chunk[in_chunk, local_idx]
            
            running_max = new_max
        
        # Compute cross entropy: -log(exp(target) / sum(exp))
        # = -target + log(sum(exp))
        # = -target + max + log(running_sum)
        log_sum_exp = running_max + torch.log(running_sum)
        
        # Mask ignored tokens
        valid = labels_flat != self.ignore_index
        loss = (-target_logits[valid] + log_sum_exp[valid]).mean()
        
        return loss


# Test parameters
batch_size = 8
seq_len = 2048
hidden_dim = 4096
vocab_size = 128000  # Large vocabulary

def get_inputs():
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    return [hidden_states, labels]

def get_init_inputs():
    return [hidden_dim, vocab_size]

