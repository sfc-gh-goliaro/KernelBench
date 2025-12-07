import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Text Detection Network (DBNet-style).
    
    Differentiable Binarization for scene text detection.
    Used in HunyuanOCR, PaddleOCR, and text spotting systems.
    
    Based on: "Real-time Scene Text Detection with Differentiable Binarization"
    """
    def __init__(self, in_channels=256, inner_channels=256):
        """
        :param in_channels: Input feature channels
        :param inner_channels: Internal channel dimension
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        
        # Feature Pyramid Network (FPN) style upsampling
        self.in5 = nn.Conv2d(in_channels, inner_channels, 1)
        self.in4 = nn.Conv2d(in_channels, inner_channels, 1)
        self.in3 = nn.Conv2d(in_channels, inner_channels, 1)
        self.in2 = nn.Conv2d(in_channels, inner_channels, 1)
        
        # Smooth layers
        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1),
            nn.Upsample(scale_factor=8, mode='nearest')
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1),
            nn.Upsample(scale_factor=4, mode='nearest')
        )
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.out2 = nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1)
        
        # Fusion and output heads
        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2),
            nn.Sigmoid()
        )
        
        # Threshold map (for differentiable binarization)
        self.thresh = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2),
            nn.Sigmoid()
        )
    
    def _upsample_add(self, x, y):
        """Upsample x and add to y."""
        _, _, h, w = y.shape
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False) + y
    
    def forward(self, features):
        """
        Detect text regions in image features.
        
        :param features: List of feature maps [c2, c3, c4, c5] from backbone
                        Each of shape (batch, in_channels, H_i, W_i)
        :return: Dict with 'probability_map', 'threshold_map', 'binary_map'
        """
        c2, c3, c4, c5 = features
        
        # Build FPN
        in5 = self.in5(c5)
        in4 = self._upsample_add(in5, self.in4(c4))
        in3 = self._upsample_add(in4, self.in3(c3))
        in2 = self._upsample_add(in3, self.in2(c2))
        
        # Multi-scale features
        out5 = self.out5(in5)
        out4 = self.out4(in4)
        out3 = self.out3(in3)
        out2 = self.out2(in2)
        
        # Concatenate
        fuse = torch.cat([out5, out4, out3, out2], dim=1)
        
        # Probability map (text regions)
        prob_map = self.binarize(fuse)
        
        # Threshold map (adaptive binarization threshold)
        thresh_map = self.thresh(fuse)
        
        # Differentiable binarization
        # DB = 1 / (1 + exp(-k * (P - T)))
        k = 50
        binary_map = 1 / (1 + torch.exp(-k * (prob_map - thresh_map)))
        
        return {
            'probability_map': prob_map,
            'threshold_map': thresh_map,
            'binary_map': binary_map
        }


# Test parameters
batch_size = 4
in_channels = 256
# Feature pyramid from different backbone stages
c2_size = (128, 128)
c3_size = (64, 64)
c4_size = (32, 32)
c5_size = (16, 16)

def get_inputs():
    c2 = torch.randn(batch_size, in_channels, *c2_size)
    c3 = torch.randn(batch_size, in_channels, *c3_size)
    c4 = torch.randn(batch_size, in_channels, *c4_size)
    c5 = torch.randn(batch_size, in_channels, *c5_size)
    return [[c2, c3, c4, c5]]

def get_init_inputs():
    return [in_channels]

