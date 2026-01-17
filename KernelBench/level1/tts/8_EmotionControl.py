import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Emotion Control (TTS/Speech Synthesis)

    Used by: EmotiVoice, EmoSpeech, Expressive TTS

    Controls emotional expression in synthesized speech. Can use discrete
    emotion labels or continuous emotion embeddings (valence, arousal, dominance).

    Shapes:
        hidden_states: (batch, seq_len, hidden_size) decoder hidden states
        emotion_label: (batch,) discrete emotion ID, or
        emotion_embed: (batch, emotion_dim) continuous emotion embedding
        Output: (batch, seq_len, hidden_size) emotion-conditioned hidden states
    """

    def __init__(self, hidden_size: int = 512, num_emotions: int = 8,
                 emotion_dim: int = 64, use_continuous: bool = False):
        """
        Initialize emotion control.

        Args:
            hidden_size: Hidden dimension
            num_emotions: Number of discrete emotion categories
            emotion_dim: Emotion embedding dimension
            use_continuous: Whether to use continuous emotion representation
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.emotion_dim = emotion_dim
        self.use_continuous = use_continuous

        if not use_continuous:
            # Discrete emotion embeddings
            self.emotion_embed = nn.Embedding(num_emotions, emotion_dim)
        else:
            # Continuous emotion projection (e.g., from VAD values)
            self.emotion_proj = nn.Linear(3, emotion_dim)  # Valence, Arousal, Dominance

        # Emotion conditioning layers (FiLM-style)
        self.scale_net = nn.Sequential(
            nn.Linear(emotion_dim, hidden_size),
            nn.Tanh()
        )

        self.shift_net = nn.Sequential(
            nn.Linear(emotion_dim, hidden_size),
            nn.Tanh()
        )

        # Optional style token attention
        self.style_attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.style_tokens = nn.Parameter(torch.randn(num_emotions, hidden_size))

    def forward(self, hidden_states: torch.Tensor,
                emotion_input: torch.Tensor) -> torch.Tensor:
        """
        Apply emotion conditioning.

        Args:
            hidden_states: Decoder hidden states (batch, seq_len, hidden_size)
            emotion_input: Emotion label (batch,) if discrete, or
                          continuous embedding (batch, 3) for VAD

        Returns:
            Emotion-conditioned hidden states (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Get emotion embedding
        if self.use_continuous:
            emotion_emb = self.emotion_proj(emotion_input)  # (batch, emotion_dim)
        else:
            emotion_emb = self.emotion_embed(emotion_input)  # (batch, emotion_dim)

        # Compute FiLM parameters
        scale = self.scale_net(emotion_emb)  # (batch, hidden_size)
        shift = self.shift_net(emotion_emb)  # (batch, hidden_size)

        # Apply affine transformation
        scale = scale.unsqueeze(1)  # (batch, 1, hidden_size)
        shift = shift.unsqueeze(1)

        conditioned = hidden_states * (1 + scale) + shift

        # Optional: add style token attention
        style_tokens = self.style_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        style_out, _ = self.style_attention(conditioned, style_tokens, style_tokens)

        # Residual connection
        output = conditioned + 0.1 * style_out

        return output


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 16
seq_len = 200
hidden_size = 512
num_emotions = 8

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    emotion_label = torch.randint(0, num_emotions, (batch_size,), device='cuda')
    return [hidden_states, emotion_label]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, num_emotions]
