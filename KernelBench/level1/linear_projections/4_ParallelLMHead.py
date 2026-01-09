import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Parallel Language Model Head
    
    Used by: vLLM, TensorRT-LLM, Megatron-LM
    
    Final projection from hidden states to vocabulary logits.
    Can be partitioned across vocabulary dimension for tensor parallelism.
    Often shares weights with input embeddings (tied embeddings).
    
    Shapes:
        Input: (batch_size, seq_length, hidden_size)
        Output: (batch_size, seq_length, vocab_size)
    """
    
    def __init__(self, hidden_size: int = 4096, vocab_size: int = 128256):
        """
        Initialize Parallel LM Head.
        
        Args:
            hidden_size: Input hidden dimension
            vocab_size: Vocabulary size (output dimension)
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # LM head projection (not tied to embeddings in this implementation)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to vocabulary logits.
        
        Args:
            x: Hidden states (batch_size, seq_length, hidden_size)
            
        Returns:
            Logits (batch_size, seq_length, vocab_size)
        """
        return self.lm_head(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
hidden_size = 4096
vocab_size = 128256  # Llama-3 vocabulary size

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, vocab_size]

