import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused Patch Embed + Position Embed
    
    Used by: ViT, CLIP, SigLIP
    
    Conv2d patch embedding + learned position embedding addition.
    """
    
    def __init__(self, img_size: int, patch_size: int, embed_dim: int):
        super(Model, self).__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        return x + self.pos_embed


batch_size, img_size, patch_size, embed_dim = 32, 224, 16, 768

def get_inputs():
    return [torch.randn(batch_size, 3, img_size, img_size, device='cuda')]

def get_init_inputs():
    return [img_size, patch_size, embed_dim]

