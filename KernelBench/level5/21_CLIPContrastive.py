import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    CLIP-style Contrastive Learning Module.
    
    Computes contrastive loss between image and text embeddings
    using learned temperature parameter and normalized features.
    
    Based on: "Learning Transferable Visual Models From Natural Language Supervision"
    """
    def __init__(self, image_dim, text_dim, embed_dim, init_temperature=0.07):
        """
        :param image_dim: Dimension of image features
        :param text_dim: Dimension of text features
        :param embed_dim: Joint embedding dimension
        :param init_temperature: Initial temperature for softmax
        """
        super(Model, self).__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.embed_dim = embed_dim
        
        # Projection layers
        self.image_proj = nn.Linear(image_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        
        # Learnable temperature (log scale for positivity)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / init_temperature)))
        
        # Maximum temperature for stability
        self.max_logit_scale = math.log(100)
        
    def forward(self, image_features, text_features, return_loss=True):
        """
        Compute CLIP contrastive embeddings and optionally loss.
        
        :param image_features: Image features (batch, image_dim)
        :param text_features: Text features (batch, text_dim)
        :param return_loss: Whether to compute and return loss
        :return: If return_loss: (loss, image_embeds, text_embeds, logits_per_image, logits_per_text)
                 Else: (image_embeds, text_embeds, logits_per_image, logits_per_text)
        """
        batch_size = image_features.shape[0]
        
        # Project to joint embedding space
        image_embeds = self.image_proj(image_features)
        text_embeds = self.text_proj(text_features)
        
        # L2 normalize embeddings
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # Clamp logit scale for stability
        logit_scale = self.logit_scale.clamp(max=self.max_logit_scale).exp()
        
        # Compute similarity logits
        logits_per_image = logit_scale * torch.matmul(image_embeds, text_embeds.t())
        logits_per_text = logits_per_image.t()
        
        if return_loss:
            # Contrastive loss: cross-entropy both directions
            labels = torch.arange(batch_size, device=image_features.device)
            
            loss_i2t = F.cross_entropy(logits_per_image, labels)
            loss_t2i = F.cross_entropy(logits_per_text, labels)
            loss = (loss_i2t + loss_t2i) / 2
            
            return loss, image_embeds, text_embeds, logits_per_image, logits_per_text
        
        return image_embeds, text_embeds, logits_per_image, logits_per_text
    
    def encode_image(self, image_features):
        """Encode image features to joint embedding space."""
        embeds = self.image_proj(image_features)
        return F.normalize(embeds, dim=-1)
    
    def encode_text(self, text_features):
        """Encode text features to joint embedding space."""
        embeds = self.text_proj(text_features)
        return F.normalize(embeds, dim=-1)


# Test parameters
batch_size = 64
image_dim = 2048  # e.g., from ResNet
text_dim = 768    # e.g., from BERT
embed_dim = 512

def get_inputs():
    image_features = torch.randn(batch_size, image_dim)
    text_features = torch.randn(batch_size, text_dim)
    return [image_features, text_features]

def get_init_inputs():
    return [image_dim, text_dim, embed_dim]

