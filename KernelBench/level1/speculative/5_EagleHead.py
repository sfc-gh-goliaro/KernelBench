import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    EAGLE Speculative Decoding Head
    
    Used by: EAGLE, EAGLE-2 speculative decoding
    """
    
    def __init__(self, hidden_size: int = 4096, vocab_size: int = 128256,
                 num_draft: int = 5, inner_dim: int = 1024):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_draft = num_draft
        
        self.fc_in = nn.Linear(hidden_size, inner_dim)
        
        self.draft_layers = nn.ModuleList([
            nn.Linear(inner_dim + hidden_size, inner_dim)
            for _ in range(num_draft)
        ])
        
        self.lm_heads = nn.ModuleList([
            nn.Linear(inner_dim, vocab_size, bias=False)
            for _ in range(num_draft)
        ])
        
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
    
    def forward(self, hidden_states: torch.Tensor,
                base_hidden: torch.Tensor = None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        
        if base_hidden is None:
            base_hidden = hidden_states
        
        h = torch.relu(self.fc_in(hidden_states))
        
        draft_logits = []
        
        for i in range(self.num_draft):
            combined = torch.cat([h, base_hidden], dim=-1)
            h = torch.relu(self.draft_layers[i](combined))
            
            logits = self.lm_heads[i](h)
            draft_logits.append(logits)
            
            token = logits.argmax(dim=-1)
            token_emb = self.token_embed(token)
            base_hidden = base_hidden + token_emb
        
        return torch.stack(draft_logits, dim=1)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "hidden_size": 4096, "vocab_size": 128256, "num_draft": 5, "inner_dim": 1024},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("speculative", "5_EagleHead")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    hidden_states = DISTRIBUTIONS[dist_name]((p["batch_size"], p["hidden_size"]), dtype=dtype, device=device)
    return [hidden_states]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["vocab_size"], p["num_draft"], p["inner_dim"]]
