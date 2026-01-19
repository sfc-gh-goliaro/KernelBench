import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    MBConv (Mobile Inverted Bottleneck Convolution)
    
    Used by: EfficientNet, EfficientNet-Lite, MobileNetV2/V3
    
    Complete MBConv block: Expand -> Depthwise -> (SE) -> Project
    With skip connection when input/output dimensions match.
    
    Shapes:
        Input: (batch_size, in_channels, height, width)
        Output: (batch_size, out_channels, height', width')
    """
    
    def __init__(self, in_channels: int = 32, out_channels: int = 16,
                 expand_ratio: int = 6, kernel_size: int = 3,
                 stride: int = 1, use_se: bool = True, se_ratio: float = 0.25):
        """
        Initialize MBConv block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            expand_ratio: Expansion ratio for intermediate channels
            kernel_size: Depthwise convolution kernel size
            stride: Stride for depthwise convolution
            use_se: Whether to use squeeze-excitation
            se_ratio: Squeeze-excitation reduction ratio
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self.expand = expand_ratio != 1
        
        if self.expand:
            self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
            self.expand_bn = nn.BatchNorm2d(expanded_channels)
        else:
            expanded_channels = in_channels
        
        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv2d(
            expanded_channels, expanded_channels, kernel_size,
            stride=stride, padding=padding, groups=expanded_channels, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        
        # Squeeze-and-Excitation
        self.use_se = use_se
        if use_se:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se_reduce = nn.Conv2d(expanded_channels, se_channels, 1)
            self.se_expand = nn.Conv2d(se_channels, expanded_channels, 1)
        
        # Projection phase
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        # Activation
        self.activation = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply MBConv block.
        
        Args:
            x: Input tensor (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor (batch_size, out_channels, height', width')
        """
        identity = x
        
        # Expansion
        if self.expand:
            out = self.expand_conv(x)
            out = self.expand_bn(out)
            out = self.activation(out)
        else:
            out = x
        
        # Depthwise convolution
        out = self.depthwise_conv(out)
        out = self.depthwise_bn(out)
        out = self.activation(out)
        
        # Squeeze-and-Excitation
        if self.use_se:
            se = out.mean(dim=[-2, -1], keepdim=True)
            se = self.se_reduce(se)
            se = self.activation(se)
            se = self.se_expand(se)
            se = torch.sigmoid(se)
            out = out * se
        
        # Projection
        out = self.project_conv(out)
        out = self.project_bn(out)
        
        # Skip connection
        if self.use_residual:
            out = out + identity
        
        return out


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 32, "in_channels": 32, "out_channels": 32, "expand_ratio": 6, "kernel_size": 3, "stride": 1, "use_se": True, "se_ratio": 0.25, "height": 56, "width": 56},
    # EfficientNet-B0: stage 2 (32->16 channels, 112x112)
    {"batch_size": 32, "in_channels": 32, "out_channels": 16, "expand_ratio": 1, "kernel_size": 3, "stride": 1, "use_se": True, "se_ratio": 0.25, "height": 112, "width": 112},
    # EfficientNet-B0: stage 3 (16->24 channels, 112x112->56x56)
    {"batch_size": 32, "in_channels": 16, "out_channels": 24, "expand_ratio": 6, "kernel_size": 3, "stride": 2, "use_se": True, "se_ratio": 0.25, "height": 112, "width": 112},
    # EfficientNet-B4: stage 4 (48->24 channels, 95x95)
    {"batch_size": 16, "in_channels": 48, "out_channels": 24, "expand_ratio": 6, "kernel_size": 5, "stride": 1, "use_se": True, "se_ratio": 0.25, "height": 95, "width": 95},
    # EfficientNet-B7: stage 5 (80->48 channels, 75x75)
    {"batch_size": 8, "in_channels": 80, "out_channels": 48, "expand_ratio": 6, "kernel_size": 5, "stride": 1, "use_se": True, "se_ratio": 0.25, "height": 75, "width": 75},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("mobile", "5_MBConv")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["in_channels"], p["height"], p["width"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_channels"], p["out_channels"], p["expand_ratio"], p["kernel_size"], p["stride"], p["use_se"], p["se_ratio"]]
