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
    EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) Drafting.
    
    Uses a lightweight autoregressive head to generate draft tokens
    by predicting feature vectors rather than tokens directly.
    
    Based on: "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"
    """
    def __init__(self, hidden_dim, num_layers=1, num_heads=8):
        """
        :param hidden_dim: Hidden dimension of the model
        :param num_layers: Number of transformer layers in draft model
        :param num_heads: Number of attention heads
        """
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Feature regression head (predicts next hidden state)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Lightweight transformer layers for auto-regression
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=0.0,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Embedding for token input (optional, for hybrid approach)
        self.token_embed_dim = hidden_dim
        
    def forward(self, hidden_states, input_embeds, num_draft=4):
        """
        Generate draft feature vectors autoregressively.
        
        :param hidden_states: Hidden states from the target model 
                             (batch, seq_len, hidden_dim)
        :param input_embeds: Token embeddings (batch, seq_len, hidden_dim)
        :param num_draft: Number of draft tokens to generate
        :return: Draft hidden states (batch, num_draft, hidden_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Concatenate hidden states and embeddings (EAGLE's key insight)
        # This provides richer context for prediction
        combined = torch.cat([hidden_states, input_embeds], dim=-1)
        
        # Project to hidden dimension
        features = self.fc(combined)  # (batch, seq_len, hidden_dim)
        
        # Get last position as starting point
        current_feature = features[:, -1:, :]  # (batch, 1, hidden_dim)
        
        draft_features = []
        
        for i in range(num_draft):
            # Create causal context window
            if i == 0:
                context = current_feature
            else:
                context = torch.cat([current_feature] + draft_features, dim=1)
            
            # Apply transformer layers
            x = context
            for layer in self.layers:
                x = layer(x)
            
            # Take last position as next feature
            next_feature = self.norm(x[:, -1:, :])
            draft_features.append(next_feature)
            
            # Update for next iteration
            current_feature = next_feature
        
        # Concatenate all draft features
        draft_hidden = torch.cat(draft_features, dim=1)  # (batch, num_draft, hidden_dim)
        
        return draft_hidden
    
    def compute_draft_logits(self, draft_hidden, lm_head_weight):
        """
        Compute logits from draft hidden states using LM head.
        
        :param draft_hidden: Draft hidden states (batch, num_draft, hidden_dim)
        :param lm_head_weight: LM head weight matrix (vocab_size, hidden_dim)
        :return: Draft logits (batch, num_draft, vocab_size)
        """
        # Apply LM head
        draft_logits = F.linear(draft_hidden, lm_head_weight)
        return draft_logits


# Test parameters
batch_size = 8
seq_len = 128
hidden_dim = 2048
num_layers = 1
num_heads = 8
num_draft = 6

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    input_embeds = torch.randn(batch_size, seq_len, hidden_dim)
    return [hidden_states, input_embeds, num_draft]

def get_init_inputs():
    return [hidden_dim, num_layers, num_heads]

