import torch
import torch.nn as nn
import torch.nn.functional as F


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    W4A16 (4-bit Weights, 16-bit Activations) Matrix Multiplication.
    
    Implements quantized linear layer where weights are stored in 4-bit
    and dequantized to FP16/BF16 for computation.
    
    Based on: GPTQ, AWQ, and similar quantization schemes
    """
    def __init__(self, in_features, out_features, group_size=128, bias=True):
        """
        :param in_features: Size of input features
        :param out_features: Size of output features
        :param group_size: Size of quantization groups
        :param bias: Whether to include bias
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_groups = in_features // group_size
        
        # Quantized weights (packed 4-bit into int32)
        # Each int32 holds 8 4-bit values
        num_packed = (in_features * out_features) // 8
        self.register_buffer('qweight', torch.zeros(num_packed, dtype=torch.int32))
        
        # Scales per group (FP16)
        self.register_buffer('scales', torch.ones(self.num_groups, out_features))
        
        # Zero points per group (4-bit, packed)
        num_zp_packed = (self.num_groups * out_features) // 8
        self.register_buffer('zeros', torch.zeros(num_zp_packed, dtype=torch.int32))
        
        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize with fake quantization for testing
        self._init_fake_quant()
        
    def _init_fake_quant(self):
        """Initialize with fake quantized weights for testing."""
        # Create random FP32 weights and quantize
        weight = torch.randn(self.out_features, self.in_features) * 0.02
        
        # Quantize per group
        weight_reshaped = weight.view(self.out_features, self.num_groups, self.group_size)
        
        # Compute scales and zero points
        w_min = weight_reshaped.min(dim=-1).values
        w_max = weight_reshaped.max(dim=-1).values
        
        scales = (w_max - w_min) / 15  # 4-bit range is 0-15
        scales = scales.clamp(min=1e-8)
        zeros = (-w_min / scales).round().clamp(0, 15).to(torch.int32)
        
        # Store scales
        self.scales.copy_(scales.t())  # (num_groups, out_features)
        
        # Quantize weights
        weight_int = ((weight_reshaped - w_min.unsqueeze(-1)) / scales.unsqueeze(-1))
        weight_int = weight_int.round().clamp(0, 15).to(torch.int32)
        
        # Pack into int32 (8 values per int32)
        weight_flat = weight_int.view(-1)
        packed = self._pack_int4(weight_flat)
        self.qweight.copy_(packed)
        
        # Pack zeros
        zeros_flat = zeros.view(-1)
        zeros_packed = self._pack_int4(zeros_flat)
        if zeros_packed.numel() <= self.zeros.numel():
            self.zeros[:zeros_packed.numel()].copy_(zeros_packed)
    
    def _pack_int4(self, values):
        """Pack int4 values into int32."""
        # Pad to multiple of 8
        if values.numel() % 8 != 0:
            pad_size = 8 - (values.numel() % 8)
            values = F.pad(values.float(), (0, pad_size)).to(torch.int32)
        
        values = values.view(-1, 8)
        packed = torch.zeros(values.shape[0], dtype=torch.int32, device=values.device)
        
        for i in range(8):
            packed |= (values[:, i] & 0xF) << (i * 4)
        
        return packed
    
    def _unpack_int4(self, packed):
        """Unpack int32 to int4 values."""
        result = []
        for i in range(8):
            result.append((packed >> (i * 4)) & 0xF)
        return torch.stack(result, dim=-1).view(-1)
    
    def dequantize(self):
        """Dequantize weights to FP16."""
        # Unpack weights
        weight_int = self._unpack_int4(self.qweight)
        weight_int = weight_int[:self.in_features * self.out_features]
        weight_int = weight_int.view(self.out_features, self.num_groups, self.group_size)
        
        # Unpack zeros
        zeros = self._unpack_int4(self.zeros)
        zeros = zeros[:self.num_groups * self.out_features]
        zeros = zeros.view(self.num_groups, self.out_features).t().unsqueeze(-1)
        
        # Dequantize
        scales = self.scales.t().unsqueeze(-1)  # (out_features, num_groups, 1)
        weight = (weight_int.float() - zeros.float()) * scales
        
        return weight.view(self.out_features, self.in_features)
    
    def forward(self, x):
        """
        Forward pass with dequantized weights.
        
        :param x: Input tensor (batch, *, in_features)
        :return: Output tensor (batch, *, out_features)
        """
        # Dequantize weights
        weight = self.dequantize()
        
        # Matrix multiplication
        output = F.linear(x, weight, self.bias)
        
        return output


# Test parameters
batch_size = 32
seq_len = 256
in_features = 4096
out_features = 4096

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    return [torch.randn(batch_size, seq_len, in_features)]

def get_init_inputs():
    return [in_features, out_features]

