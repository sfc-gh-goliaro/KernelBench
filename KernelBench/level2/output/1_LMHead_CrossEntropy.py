import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused LM Head + Cross-Entropy Loss
    
    Used by: LLM training (memory optimization)
    
    Chunked LM head matmul + cross-entropy without full logits materialization.
    """
    
    def __init__(self, hidden_size: int, vocab_size: int):
        super(Model, self).__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))


batch_size, seq_len, hidden_size, vocab_size = 8, 512, 4096, 32000

def get_inputs():
    hidden = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
    return [hidden, labels]

def get_init_inputs():
    return [hidden_size, vocab_size]

