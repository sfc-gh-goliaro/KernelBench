import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Dual Autoregressive Decoder (TTS/Speech Synthesis)

    Used by: CosyVoice, VALL-E 2

    Two-stage autoregressive decoder: first AR generates coarse tokens,
    second AR (or NAR) generates fine tokens for high-quality synthesis.

    Shapes:
        text_embeds: (batch, text_len, hidden_size) text embeddings
        coarse_tokens: (batch, audio_len) coarse audio tokens (for fine stage)
        Output: (batch, audio_len, vocab_size) logits for next token
    """

    def __init__(self, hidden_size: int = 512, num_heads: int = 8,
                 num_layers: int = 4, vocab_size: int = 1024):
        """
        Initialize dual AR decoder.

        Args:
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            vocab_size: Audio token vocabulary size
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Token embeddings
        self.audio_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(8192, hidden_size)

        # Coarse decoder layers
        self.coarse_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # Fine decoder layers
        self.fine_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers // 2)
        ])

        # Output heads
        self.coarse_head = nn.Linear(hidden_size, vocab_size)
        self.fine_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, text_embeds: torch.Tensor,
                coarse_tokens: torch.Tensor = None,
                stage: str = 'coarse') -> torch.Tensor:
        """
        Dual AR decoder forward.

        Args:
            text_embeds: Text encoder outputs (batch, text_len, hidden_size)
            coarse_tokens: Coarse audio tokens for fine stage (batch, audio_len)
            stage: 'coarse' or 'fine'

        Returns:
            Logits (batch, audio_len, vocab_size)
        """
        batch_size = text_embeds.shape[0]
        device = text_embeds.device

        if stage == 'coarse':
            # Coarse stage: generate from text only
            # Use text as memory, start with BOS token
            audio_len = 1 if coarse_tokens is None else coarse_tokens.shape[1]

            if coarse_tokens is None:
                # Start token
                tgt = torch.zeros(batch_size, 1, self.hidden_size, device=device)
            else:
                tgt = self.audio_embed(coarse_tokens)

            # Add positional embeddings
            positions = torch.arange(tgt.shape[1], device=device)
            tgt = tgt + self.pos_embed(positions)

            # Causal mask
            tgt_len = tgt.shape[1]
            causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()

            # Apply coarse decoder
            for layer in self.coarse_layers:
                tgt = layer(tgt, text_embeds, tgt_mask=causal_mask)

            logits = self.coarse_head(tgt)

        else:  # fine stage
            # Fine stage: condition on coarse tokens
            tgt = self.audio_embed(coarse_tokens)
            positions = torch.arange(tgt.shape[1], device=device)
            tgt = tgt + self.pos_embed(positions)

            # Fine decoder can be non-autoregressive (no causal mask)
            for layer in self.fine_layers:
                tgt = layer(tgt, text_embeds)

            logits = self.fine_head(tgt)

        return logits


# ============================================================================
# Benchmark Configuration
# ============================================================================
