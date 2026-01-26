import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Reward Model Head
    
    The output head for reward models used in RLHF.
    Used by: Skywork-Reward, ArmoRM, Nemotron-RM, InternLM2-Reward
    
    Architecture:
        hidden_states (from LLM) -> Last Token Pooling -> Linear -> Scalar Score
    
    Key features:
    - Extracts last token representation
    - Projects to scalar reward score
    - No softmax (direct score output for preference learning)
    
    Typically attached to a decoder-only LLM backbone.
    """
    def __init__(self, hidden_size: int, intermediate_size: int = None, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        
        if intermediate_size is not None:
            # Two-layer MLP head
            self.head = nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_size, 1)
            )
        else:
            # Single linear projection
            self.head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute reward scores.
        
        Args:
            hidden_states: LLM output (batch, seq_len, hidden_size)
            attention_mask: Optional mask to find last real token (batch, seq_len)
        
        Returns:
            Reward scores (batch, 1)
        """
        if attention_mask is not None:
            # Find last non-padding token for each sequence
            # Sum attention mask to get sequence lengths, then index
            seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_token_hidden = hidden_states[batch_indices, seq_lengths.long()]
        else:
            # No mask, use actual last token
            last_token_hidden = hidden_states[:, -1]
        
        # Project to scalar score
        reward = self.head(last_token_hidden)
        
        return reward


# Benchmark configuration (Llama-3-8B dimensions)
batch_size = 8
seq_len = 2048
hidden_size = 4096

def get_inputs():
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    # Simulate variable length sequences
    for i in range(batch_size):
        attention_mask[i, seq_len - i * 100:] = 0
    return [hidden_states, attention_mask]

def get_init_inputs():
    return [hidden_size]

