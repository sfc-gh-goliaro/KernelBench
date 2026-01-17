import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Duration Predictor (TTS/Speech Synthesis)

    Used by: FastSpeech, VITS, Grad-TTS

    Predicts phoneme/token durations for non-autoregressive TTS.
    Determines how many mel frames each input token spans.

    Shapes:
        hidden_states: (batch, seq_len, hidden_size) encoder outputs
        Output: (batch, seq_len) predicted durations (positive values)
    """

    def __init__(self, hidden_size: int = 256, kernel_size: int = 3,
                 num_layers: int = 2, dropout: float = 0.1):
        """
        Initialize duration predictor.

        Args:
            hidden_size: Input hidden dimension
            kernel_size: Convolution kernel size
            num_layers: Number of conv layers
            dropout: Dropout probability
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size

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

        # Output projection
        self.output_proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Predict durations.

        Args:
            hidden_states: Encoder hidden states (batch, seq_len, hidden_size)
            padding_mask: Optional padding mask (batch, seq_len)

        Returns:
            Predicted durations (batch, seq_len), positive values
        """
        # Transpose for conv1d: (batch, hidden, seq)
        x = hidden_states.transpose(1, 2)

        # Apply conv layers
        for i in range(0, len(self.conv_layers), 4):
            conv = self.conv_layers[i]
            relu = self.conv_layers[i + 1]
            norm = self.conv_layers[i + 2]
            dropout = self.conv_layers[i + 3]

            x = conv(x)
            x = relu(x)
            x = x.transpose(1, 2)  # (batch, seq, hidden) for LayerNorm
            x = norm(x)
            x = dropout(x)
            x = x.transpose(1, 2)  # Back to (batch, hidden, seq)

        # Transpose back and project
        x = x.transpose(1, 2)  # (batch, seq, hidden)
        durations = self.output_proj(x).squeeze(-1)  # (batch, seq)

        # Apply softplus to ensure positive durations
        durations = F.softplus(durations)

        # Apply padding mask if provided
        if padding_mask is not None:
            durations = durations.masked_fill(padding_mask, 0.0)

        return durations


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 16
seq_len = 128  # Number of phonemes/tokens
hidden_size = 256

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    return [hidden_states]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size]
