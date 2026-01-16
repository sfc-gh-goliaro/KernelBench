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
    """
    
    def __init__(self, in_channels: int = 32, out_channels: int = 16,
                 expand_ratio: int = 6, kernel_size: int = 3,
                 stride: int = 1, use_se: bool = True, se_ratio: float = 0.25):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        expanded_channels = in_channels * expand_ratio
        self.expand = expand_ratio != 1
        
        if self.expand:
            self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
            self.expand_bn = nn.BatchNorm2d(expanded_channels)
        else:
            expanded_channels = in_channels
        
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv2d(
            expanded_channels, expanded_channels, kernel_size,
            stride=stride, padding=padding, groups=expanded_channels, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        
        self.use_se = use_se
        if use_se:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se_reduce = nn.Conv2d(expanded_channels, se_channels, 1)
            self.se_expand = nn.Conv2d(se_channels, expanded_channels, 1)
        
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        self.activation = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        if self.expand:
            out = self.expand_conv(x)
            out = self.expand_bn(out)
            out = self.activation(out)
        else:
            out = x
        
        out = self.depthwise_conv(out)
        out = self.depthwise_bn(out)
        out = self.activation(out)
        
        if self.use_se:
            se = out.mean(dim=[-2, -1], keepdim=True)
            se = self.se_reduce(se)
            se = self.activation(se)
            se = self.se_expand(se)
            se = torch.sigmoid(se)
            out = out * se
        
        out = self.project_conv(out)
        out = self.project_bn(out)
        
        if self.use_residual:
            out = out + identity
        
        return out


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "in_channels": 32, "out_channels": 32, "height": 56, "width": 56,
     "expand_ratio": 6, "kernel_size": 3, "stride": 1, "use_se": True, "se_ratio": 0.25},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("mobile", "5_MBConv")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["in_channels"], p["height"], p["width"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_channels"], p["out_channels"], p["expand_ratio"], p["kernel_size"], 
            p["stride"], p["use_se"], p["se_ratio"]]
