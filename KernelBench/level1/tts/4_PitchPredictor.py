import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Pitch Predictor (TTS/Speech Synthesis)

    Used by: FastPitch, PortaSpeech, VITS

    Predicts fundamental frequency (F0) contour for expressive speech.
    Can predict continuous pitch or pitch embeddings.

    Shapes:
        hidden_states: (batch, seq_len, hidden_size) encoder outputs
        Output: (batch, seq_len) predicted pitch values (Hz or normalized)
    """

    def __init__(self, hidden_size: int = 256, kernel_size: int = 3,
                 num_layers: int = 2, dropout: float = 0.1,
                 pitch_embedding_dim: int = 256):
        """
        Initialize pitch predictor.

        Args:
            hidden_size: Input hidden dimension
            kernel_size: Convolution kernel size
            num_layers: Number of conv layers
            dropout: Dropout probability
            pitch_embedding_dim: Dimension for pitch embeddings
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.pitch_embedding_dim = pitch_embedding_dim

        # Convolutional layers
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Conv1d(hidden_size, hidden_size, kernel_size,
                         padding=kernel_size // 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout)
            ])
        self.conv_layers = nn.ModuleList(layers)

        # Pitch value prediction
        self.pitch_proj = nn.Linear(hidden_size, 1)

        # Pitch embedding (continuous)
        self.pitch_embed = nn.Sequential(
            nn.Linear(1, pitch_embedding_dim),
            nn.Tanh()
        )

    def forward(self, hidden_states: torch.Tensor,
                padding_mask: torch.Tensor = None) -> tuple:
        """
        Predict pitch contour.

        Args:
            hidden_states: Encoder hidden states (batch, seq_len, hidden_size)
            padding_mask: Optional padding mask (batch, seq_len)

        Returns:
            Tuple of:
                - pitch_values: Predicted pitch (batch, seq_len)
                - pitch_embeddings: Pitch embeddings (batch, seq_len, pitch_embed_dim)
        """
        # Transpose for conv1d
        x = hidden_states.transpose(1, 2)

        # Apply conv layers
        for i in range(0, len(self.conv_layers), 4):
            conv = self.conv_layers[i]
            relu = self.conv_layers[i + 1]
            norm = self.conv_layers[i + 2]
            dropout = self.conv_layers[i + 3]

            x = conv(x)
            x = relu(x)
            x = x.transpose(1, 2)
            x = norm(x)
            x = dropout(x)
            x = x.transpose(1, 2)

        # Transpose back and project
        x = x.transpose(1, 2)  # (batch, seq, hidden)
        pitch_values = self.pitch_proj(x).squeeze(-1)  # (batch, seq)

        # Get pitch embeddings
        pitch_embeddings = self.pitch_embed(pitch_values.unsqueeze(-1))

        # Apply padding mask if provided
        if padding_mask is not None:
            pitch_values = pitch_values.masked_fill(padding_mask, 0.0)
            pitch_embeddings = pitch_embeddings.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return pitch_values, pitch_embeddings


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 16
seq_len = 128
hidden_size = 256
pitch_embedding_dim = 256

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    return [hidden_states]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size]
