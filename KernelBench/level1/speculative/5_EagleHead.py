import torch
import torch.nn as nn

class Model(nn.Module):
    """
    EAGLE Speculative Decoding Head
    
    Used by: EAGLE, EAGLE-2 speculative decoding
    
    Lightweight head that predicts multiple future tokens autoregressively.
    Uses hidden states from base model to draft next tokens efficiently.
    
    Shapes:
        Input: (batch_size, hidden_size) - current hidden state
        Output: (batch_size, num_draft, vocab_size) - draft token logits
    """
    
    def __init__(self, hidden_size: int = 4096, vocab_size: int = 128256,
                 num_draft: int = 5, inner_dim: int = 1024):
        """
        Initialize EAGLE Head.
        
        Args:
            hidden_size: Base model hidden dimension
            vocab_size: Vocabulary size
            num_draft: Number of draft tokens to generate
            inner_dim: Inner dimension for draft head
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_draft = num_draft
        
        # Project from base hidden to inner dimension
        self.fc_in = nn.Linear(hidden_size, inner_dim)
        
        # Small autoregressive layers for drafting
        self.draft_layers = nn.ModuleList([
            nn.Linear(inner_dim + hidden_size, inner_dim)
            for _ in range(num_draft)
        ])
        
        # Output heads for each draft position
        self.lm_heads = nn.ModuleList([
            nn.Linear(inner_dim, vocab_size, bias=False)
            for _ in range(num_draft)
        ])
        
        # Embedding for feeding back draft tokens
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
    
    def forward(self, hidden_states: torch.Tensor,
                base_hidden: torch.Tensor = None) -> torch.Tensor:
        """
        Generate draft token logits.
        
        Args:
            hidden_states: Current hidden state (batch_size, hidden_size)
            base_hidden: Optional base model hidden for context
            
        Returns:
            Draft logits (batch_size, num_draft, vocab_size)
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        
        if base_hidden is None:
            base_hidden = hidden_states
        
        # Initial projection
        h = torch.relu(self.fc_in(hidden_states))
        
        # Collect draft logits
        draft_logits = []
        
        for i in range(self.num_draft):
            # Combine current state with base context
            combined = torch.cat([h, base_hidden], dim=-1)
            h = torch.relu(self.draft_layers[i](combined))
            
            # Compute logits for this position
            logits = self.lm_heads[i](h)
            draft_logits.append(logits)
            
            # Sample token and embed for next iteration (greedy for simplicity)
            token = logits.argmax(dim=-1)
            token_emb = self.token_embed(token)
            
            # Update base_hidden with token embedding
            base_hidden = base_hidden + token_emb
        
        # Stack: (batch_size, num_draft, vocab_size)
        return torch.stack(draft_logits, dim=1)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 32
hidden_size = 4096
vocab_size = 128256
num_draft = 5
inner_dim = 1024

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    hidden_states = torch.randn(batch_size, hidden_size, device='cuda')
    return [hidden_states]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, vocab_size, num_draft, inner_dim]

