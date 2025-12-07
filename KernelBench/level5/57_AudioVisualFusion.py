import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Audio-Visual Fusion for Omni-modal Models.
    
    Fuses audio and visual features for multimodal understanding.
    Used in Qwen-Omni and similar models.
    
    Based on: Multimodal fusion architectures
    """
    def __init__(self, audio_dim, visual_dim, hidden_dim, num_heads, num_layers=2):
        """
        :param audio_dim: Audio feature dimension
        :param visual_dim: Visual feature dimension
        :param hidden_dim: Hidden/output dimension
        :param num_heads: Number of attention heads
        :param num_layers: Number of fusion layers
        """
        super(Model, self).__init__()
        self.audio_dim = audio_dim
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Modality projections
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        
        # Modality embeddings
        self.audio_type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.visual_type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Cross-modal fusion layers
        self.fusion_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.fusion_layers.append(nn.ModuleDict({
                # Audio attends to visual
                'audio_cross_attn': nn.MultiheadAttention(
                    hidden_dim, num_heads, batch_first=True
                ),
                'audio_norm1': nn.LayerNorm(hidden_dim),
                'audio_norm2': nn.LayerNorm(hidden_dim),
                'audio_ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                ),
                
                # Visual attends to audio
                'visual_cross_attn': nn.MultiheadAttention(
                    hidden_dim, num_heads, batch_first=True
                ),
                'visual_norm1': nn.LayerNorm(hidden_dim),
                'visual_norm2': nn.LayerNorm(hidden_dim),
                'visual_ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                ),
            }))
        
        # Temporal alignment (optional)
        self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, audio_features, visual_features, 
                audio_mask=None, visual_mask=None):
        """
        Fuse audio and visual features.
        
        :param audio_features: Audio features (batch, audio_len, audio_dim)
        :param visual_features: Visual features (batch, visual_len, visual_dim)
        :param audio_mask: Optional audio attention mask
        :param visual_mask: Optional visual attention mask
        :return: Tuple of (fused_audio, fused_visual)
        """
        # Project to hidden dimension
        audio = self.audio_proj(audio_features)
        visual = self.visual_proj(visual_features)
        
        # Add modality embeddings
        audio = audio + self.audio_type_embed
        visual = visual + self.visual_type_embed
        
        # Cross-modal fusion
        for layer in self.fusion_layers:
            # Audio attends to visual
            audio_norm = layer['audio_norm1'](audio)
            audio_cross, _ = layer['audio_cross_attn'](
                audio_norm, visual, visual,
                key_padding_mask=visual_mask
            )
            audio = audio + audio_cross
            audio = audio + layer['audio_ffn'](layer['audio_norm2'](audio))
            
            # Visual attends to audio
            visual_norm = layer['visual_norm1'](visual)
            visual_cross, _ = layer['visual_cross_attn'](
                visual_norm, audio, audio,
                key_padding_mask=audio_mask
            )
            visual = visual + visual_cross
            visual = visual + layer['visual_ffn'](layer['visual_norm2'](visual))
        
        # Temporal alignment for audio
        audio_temporal = self.temporal_conv(audio.transpose(1, 2)).transpose(1, 2)
        audio = audio + audio_temporal
        
        # Final normalization
        audio = self.output_norm(audio)
        visual = self.output_norm(visual)
        
        return audio, visual


# Test parameters
batch_size = 4
audio_len = 1500   # ~30 seconds of audio
visual_len = 256   # Image patches
audio_dim = 512    # Whisper encoder dimension
visual_dim = 1024  # Vision encoder dimension
hidden_dim = 768
num_heads = 12

def get_inputs():
    audio_features = torch.randn(batch_size, audio_len, audio_dim)
    visual_features = torch.randn(batch_size, visual_len, visual_dim)
    return [audio_features, visual_features]

def get_init_inputs():
    return [audio_dim, visual_dim, hidden_dim, num_heads]

