import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Squeeze-Excitation Block
    
    Used by: EfficientNet, SE-ResNet
    
    AdaptiveAvgPool + FC + ReLU + FC + Sigmoid + Scale.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super(Model, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        se = self.avg_pool(x).view(b, c)
        se = torch.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).view(b, c, 1, 1)
        return x * se


batch_size, channels, height, width = 32, 256, 14, 14

def get_inputs():
    return [torch.randn(batch_size, channels, height, width, device='cuda')]

def get_init_inputs():
    return [channels]

