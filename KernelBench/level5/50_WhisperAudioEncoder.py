import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Whisper-style Audio Encoder.
    
    Convolutional frontend followed by transformer encoder
    for processing mel spectrograms in speech recognition.
    
    Based on: "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper)
    """
    def __init__(self, n_mels=80, n_ctx=1500, n_state=512, n_head=8, n_layer=6):
        """
        :param n_mels: Number of mel channels
        :param n_ctx: Maximum context length (audio frames)
        :param n_state: Model dimension
        :param n_head: Number of attention heads
        :param n_layer: Number of transformer layers
        """
        super(Model, self).__init__()
        self.n_mels = n_mels
        self.n_ctx = n_ctx
        self.n_state = n_state
        
        # Convolutional frontend (2 conv layers)
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        
        # Sinusoidal positional embedding
        self.register_buffer('positional_embedding', 
            self._sinusoidal_embedding(n_ctx, n_state))
        
        # Transformer encoder layers
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_state,
                nhead=n_head,
                dim_feedforward=n_state * 4,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(n_layer)
        ])
        
        # Final layer norm
        self.ln_post = nn.LayerNorm(n_state)
    
    def _sinusoidal_embedding(self, length, dim):
        """Create sinusoidal positional embeddings."""
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)
        )
        
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, mel):
        """
        Encode mel spectrogram to hidden representations.
        
        :param mel: Mel spectrogram (batch, n_mels, time_frames)
        :return: Encoded features (batch, time_frames/2, n_state)
        """
        # Convolutional frontend
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        
        # Transpose to (batch, time, channels)
        x = x.transpose(1, 2)
        
        # Add positional embedding
        seq_len = x.shape[1]
        x = x + self.positional_embedding[:seq_len]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_post(x)
        
        return x


# Test parameters
batch_size = 4
n_mels = 80
time_frames = 3000  # ~30 seconds of audio
n_ctx = 1500
n_state = 512
n_head = 8
n_layer = 6

def get_inputs():
    mel = torch.randn(batch_size, n_mels, time_frames)
    return [mel]

def get_init_inputs():
    return [n_mels, n_ctx, n_state, n_head, n_layer]

