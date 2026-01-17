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

batch_size = 32
in_channels = 32
out_channels = 32
expand_ratio = 6
kernel_size = 3
stride = 1
use_se = True
se_ratio = 0.25
height = 56
width = 56

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, in_channels, height, width, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [in_channels, out_channels, expand_ratio, kernel_size, stride, use_se, se_ratio]

