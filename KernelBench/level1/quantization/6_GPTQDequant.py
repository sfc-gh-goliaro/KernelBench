import torch
import torch.nn as nn

class Model(nn.Module):
    """
    GPTQ Dequantization
    
    Used by: Many open-source quantized models (TheBloke, etc.)
    
    Dequantizes GPTQ-quantized weights for inference.
    GPTQ uses optimal brain quantization with Hessian-based error correction.
    
    Shapes:
        Input: (batch_size, seq_length, in_features)
        Output: (batch_size, seq_length, out_features)
    """
    
    def __init__(self, in_features: int = 4096, out_features: int = 4096,
                 bits: int = 4, group_size: int = 128):
        """
        Initialize GPTQ Quantized Linear.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bits: Quantization bits (typically 4 or 8)
            group_size: Quantization group size
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        
        assert in_features % group_size == 0
        self.num_groups = in_features // group_size
        
        # Quantized weights (packed)
        elements_per_int32 = 32 // bits
        self.qweight = nn.Parameter(
            torch.randint(0, 2**bits, (out_features, in_features // elements_per_int32),
                         dtype=torch.int32),
            requires_grad=False
        )
        
        # Scales and zeros for each group
        self.scales = nn.Parameter(torch.ones(out_features, self.num_groups))
        self.qzeros = nn.Parameter(
            torch.zeros(out_features, self.num_groups // elements_per_int32, dtype=torch.int32),
            requires_grad=False
        )
        
        # Permutation for optimal access pattern (optional)
        self.g_idx = nn.Parameter(
            torch.arange(in_features, dtype=torch.int32),
            requires_grad=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GPTQ quantized linear with dequantization.
        
        Args:
            x: Input tensor (batch_size, seq_length, in_features)
            
        Returns:
            Output tensor (batch_size, seq_length, out_features)
        """
        # Dequantize weights
        weight_fp = self._dequantize()
        
        # Linear operation
        return torch.nn.functional.linear(x, weight_fp)
    
    def _dequantize(self) -> torch.Tensor:
        """Dequantize GPTQ weights."""
        elements_per_int32 = 32 // self.bits
        mask = (1 << self.bits) - 1
        
        # Unpack quantized weights
        weight_int = torch.zeros(self.out_features, self.in_features,
                                 dtype=torch.float32, device=self.qweight.device)
        
        for i in range(elements_per_int32):
            shift = i * self.bits
            weight_int[:, i::elements_per_int32] = ((self.qweight >> shift) & mask).float()
        
        # Unpack zeros
        zeros = torch.zeros(self.out_features, self.num_groups,
                           dtype=torch.float32, device=self.qzeros.device)
        for i in range(elements_per_int32):
            shift = i * self.bits
            zeros[:, i::elements_per_int32] = ((self.qzeros >> shift) & mask).float()
        
        # Dequantize: w = (q - zero) * scale
        weight_fp = torch.zeros_like(weight_int)
        for g in range(self.num_groups):
            start = g * self.group_size
            end = start + self.group_size
            weight_fp[:, start:end] = (
                (weight_int[:, start:end] - zeros[:, g:g+1]) *
                self.scales[:, g:g+1]
            )
        
        return weight_fp


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
in_features = 4096
out_features = 4096
bits = 4
group_size = 128

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, in_features, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [in_features, out_features, bits, group_size]

