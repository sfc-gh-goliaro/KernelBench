import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    NaViT-style Patch Packing for Variable Resolution Images.
    
    Packs patches from multiple images into a single sequence
    with factorized position encoding. Used in PaddleOCR-VL.
    
    Based on: "Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution"
    """
    def __init__(self, patch_size, embed_dim, max_patches=4096):
        """
        :param patch_size: Size of image patches
        :param embed_dim: Embedding dimension
        :param max_patches: Maximum total patches in packed sequence
        """
        super(Model, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.max_patches = max_patches
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Factorized position embeddings (not absolute)
        # Learned embeddings for relative positions
        self.max_hw = 128  # Max height/width in patches
        self.row_embed = nn.Parameter(torch.randn(1, self.max_hw, embed_dim) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(1, self.max_hw, embed_dim) * 0.02)
        
        # Mask token for attention masking between images
        self.register_buffer('mask_token', torch.zeros(1, 1, embed_dim))
    
    def create_packed_attention_mask(self, image_ids, seq_len):
        """
        Create attention mask that prevents cross-image attention.
        
        :param image_ids: Image ID for each patch (batch, seq_len)
        :param seq_len: Sequence length
        :return: Attention mask (batch, 1, seq_len, seq_len)
        """
        # Patches can only attend to patches from the same image
        mask = image_ids.unsqueeze(-1) == image_ids.unsqueeze(-2)
        return mask.unsqueeze(1).float()
    
    def pack_images(self, images_list):
        """
        Pack multiple variable-size images into a single sequence.
        
        :param images_list: List of image tensors, each (3, H_i, W_i)
        :return: Tuple of (packed_patches, position_embeddings, image_ids)
        """
        all_patches = []
        all_pos_embeds = []
        all_image_ids = []
        
        for img_idx, image in enumerate(images_list):
            _, h, w = image.shape
            
            # Get patches
            patches = self.patch_embed(image.unsqueeze(0))  # (1, embed_dim, h_p, w_p)
            h_patches, w_patches = patches.shape[2], patches.shape[3]
            
            # Flatten patches
            patches = patches.flatten(2).transpose(1, 2)  # (1, num_patches, embed_dim)
            
            # Get position embeddings (factorized)
            row_pos = self.row_embed[:, :h_patches].unsqueeze(2)  # (1, h, 1, dim)
            col_pos = self.col_embed[:, :w_patches].unsqueeze(1)  # (1, 1, w, dim)
            pos_embed = (row_pos + col_pos).view(1, -1, self.embed_dim)  # (1, h*w, dim)
            
            # Image IDs
            image_ids = torch.full((patches.shape[1],), img_idx, dtype=torch.long)
            
            all_patches.append(patches.squeeze(0))
            all_pos_embeds.append(pos_embed.squeeze(0))
            all_image_ids.append(image_ids)
        
        # Concatenate all
        packed_patches = torch.cat(all_patches, dim=0)  # (total_patches, embed_dim)
        packed_pos = torch.cat(all_pos_embeds, dim=0)
        image_ids = torch.cat(all_image_ids, dim=0)
        
        return packed_patches, packed_pos, image_ids
    
    def forward(self, packed_patches, packed_pos, image_ids):
        """
        Process packed sequence with position embeddings.
        
        :param packed_patches: Packed patch embeddings (batch, seq_len, embed_dim)
        :param packed_pos: Position embeddings (batch, seq_len, embed_dim)
        :param image_ids: Image IDs for each patch (batch, seq_len)
        :return: Tuple of (embedded_patches, attention_mask)
        """
        batch_size, seq_len, _ = packed_patches.shape
        
        # Add position embeddings
        x = packed_patches + packed_pos
        
        # Create attention mask
        attn_mask = self.create_packed_attention_mask(image_ids, seq_len)
        
        # Convert to additive mask for transformer
        attn_mask = (1.0 - attn_mask) * -10000.0
        
        return x, attn_mask


# Test parameters
batch_size = 2
embed_dim = 1024
patch_size = 14
# Packed sequence from multiple images of different sizes
seq_len = 512  # Total patches from all images

def get_inputs():
    packed_patches = torch.randn(batch_size, seq_len, embed_dim)
    packed_pos = torch.randn(batch_size, seq_len, embed_dim)
    # Image IDs: first 256 patches from image 0, rest from image 1
    image_ids = torch.cat([
        torch.zeros(256, dtype=torch.long),
        torch.ones(256, dtype=torch.long)
    ]).unsqueeze(0).expand(batch_size, -1)
    return [packed_patches, packed_pos, image_ids]

def get_init_inputs():
    return [patch_size, embed_dim]

