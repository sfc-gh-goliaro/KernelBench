import torch
import torch.nn as nn
import torch.nn.functional as F


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Medusa Heads for parallel draft token generation.
    
    Uses multiple prediction heads to generate multiple future tokens
    in a single forward pass, enabling efficient speculative decoding.
    
    Based on: "Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads"
    """
    def __init__(self, hidden_dim, vocab_size, num_heads, num_layers=1):
        """
        :param hidden_dim: Hidden dimension from the base model
        :param vocab_size: Vocabulary size
        :param num_heads: Number of Medusa heads (each predicts a future token)
        :param num_layers: Number of layers per Medusa head
        """
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Create Medusa heads
        # Each head predicts token at position i+1, i+2, ..., i+num_heads
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            layers = []
            for j in range(num_layers):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.SiLU())
            layers.append(nn.Linear(hidden_dim, vocab_size))
            self.heads.append(nn.Sequential(*layers))
        
        # Residual connections with learnable weights
        self.residual_weights = nn.Parameter(torch.ones(num_heads))
        
    def forward(self, hidden_states, top_k=10):
        """
        Generate draft tokens from multiple Medusa heads.
        
        :param hidden_states: Hidden states from base model (batch, seq_len, hidden_dim)
        :param top_k: Number of top candidates per head
        :return: Tuple of (draft_tokens, draft_probs)
            - draft_tokens: (batch, num_heads, top_k) top-k token candidates per head
            - draft_probs: (batch, num_heads, top_k) probabilities for candidates
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Use last position for prediction
        last_hidden = hidden_states[:, -1, :]  # (batch, hidden_dim)
        
        all_logits = []
        all_top_k_tokens = []
        all_top_k_probs = []
        
        for i, head in enumerate(self.heads):
            # Apply residual weight
            weighted_hidden = last_hidden * self.residual_weights[i]
            
            # Get logits for this head
            logits = head(weighted_hidden)  # (batch, vocab_size)
            all_logits.append(logits)
            
            # Get top-k candidates
            probs = F.softmax(logits, dim=-1)
            top_k_probs, top_k_tokens = torch.topk(probs, top_k, dim=-1)
            
            all_top_k_tokens.append(top_k_tokens)
            all_top_k_probs.append(top_k_probs)
        
        # Stack results
        draft_tokens = torch.stack(all_top_k_tokens, dim=1)  # (batch, num_heads, top_k)
        draft_probs = torch.stack(all_top_k_probs, dim=1)    # (batch, num_heads, top_k)
        
        return draft_tokens, draft_probs
    
    def build_tree_candidates(self, hidden_states, tree_indices, top_k=10):
        """
        Build tree-structured candidates using Medusa heads.
        
        :param hidden_states: Hidden states from base model (batch, seq_len, hidden_dim)
        :param tree_indices: Predefined tree structure for candidate generation
        :param top_k: Number of candidates per level
        :return: Tree candidates and their probabilities
        """
        batch_size = hidden_states.shape[0]
        
        # Get candidates from all heads
        draft_tokens, draft_probs = self.forward(hidden_states, top_k)
        
        # Build tree structure
        # tree_indices defines which candidates to combine
        # e.g., [[0], [0,1], [0,1,2], ...] for progressive expansion
        
        tree_candidates = []
        tree_probs = []
        
        for path in tree_indices:
            if len(path) > self.num_heads:
                continue
            
            # Build candidate sequence for this path
            candidates = []
            probs = []
            
            for head_idx, token_idx in enumerate(path):
                if head_idx < self.num_heads and token_idx < top_k:
                    candidates.append(draft_tokens[:, head_idx, token_idx])
                    probs.append(draft_probs[:, head_idx, token_idx])
            
            if candidates:
                tree_candidates.append(torch.stack(candidates, dim=1))
                tree_probs.append(torch.stack(probs, dim=1).prod(dim=1))
        
        return tree_candidates, tree_probs


# Test parameters
batch_size = 8
seq_len = 256
hidden_dim = 4096
vocab_size = 32000
num_heads = 5  # Predict 5 future tokens
top_k = 10

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    return [hidden_states, top_k]

def get_init_inputs():
    return [hidden_dim, vocab_size, num_heads]

