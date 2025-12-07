import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Feature Pyramid Fusion for Multi-Scale Multimodal Processing.
    
    Fuses features from different scales/levels using top-down
    and bottom-up pathways with lateral connections.
    
    Based on: "Feature Pyramid Networks" and multimodal variants
    """
    def __init__(self, in_channels_list, out_channels, num_levels=4):
        """
        :param in_channels_list: List of input channels for each level
        :param out_channels: Output channel dimension (uniform across levels)
        :param num_levels: Number of pyramid levels
        """
        super(Model, self).__init__()
        self.num_levels = num_levels
        self.out_channels = out_channels
        
        # Lateral connections (1x1 convs to match channel dims)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        
        # Top-down pathway (for upsampling and combining)
        self.topdown_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(num_levels - 1)
        ])
        
        # Bottom-up pathway (optional, for bidirectional fusion)
        self.bottomup_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_levels - 1)
        ])
        
        # Attention-based fusion weights
        self.fusion_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2))  # For combining top-down and lateral
            for _ in range(num_levels - 1)
        ])
        
        # Output projection
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(num_levels)
        ])
    
    def forward(self, features):
        """
        Forward pass for feature pyramid fusion.
        
        :param features: List of feature maps at different scales
                        [(batch, C1, H1, W1), ..., (batch, Cn, Hn, Wn)]
                        Ordered from finest (largest spatial) to coarsest
        :return: List of fused feature maps at each level
        """
        assert len(features) == self.num_levels
        
        # Apply lateral connections
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # Top-down pathway
        topdown = [laterals[-1]]  # Start with coarsest level
        
        for i in range(self.num_levels - 2, -1, -1):
            # Upsample previous level
            upsampled = F.interpolate(topdown[0], size=laterals[i].shape[-2:], 
                                      mode='bilinear', align_corners=False)
            
            # Weighted fusion
            weights = F.softmax(self.fusion_weights[i], dim=0)
            fused = weights[0] * laterals[i] + weights[1] * upsampled
            
            # Apply convolution
            fused = self.topdown_convs[self.num_levels - 2 - i](fused)
            
            topdown.insert(0, fused)
        
        # Bottom-up pathway (refine with top-down results)
        outputs = [topdown[0]]
        
        for i in range(1, self.num_levels):
            # Downsample previous output
            downsampled = self.bottomup_convs[i - 1](outputs[-1])
            
            # Add with top-down result
            combined = downsampled + topdown[i]
            outputs.append(combined)
        
        # Apply output projections
        outputs = [conv(out) for conv, out in zip(self.output_convs, outputs)]
        
        return outputs


# Test parameters
batch_size = 4
in_channels_list = [256, 512, 1024, 2048]  # Typical ResNet channels
out_channels = 256
num_levels = 4
# Feature sizes (decreasing by factor of 2)
sizes = [(56, 56), (28, 28), (14, 14), (7, 7)]

def get_inputs():
    features = [
        torch.randn(batch_size, in_ch, h, w)
        for in_ch, (h, w) in zip(in_channels_list, sizes)
    ]
    return [features]

def get_init_inputs():
    return [in_channels_list, out_channels, num_levels]

