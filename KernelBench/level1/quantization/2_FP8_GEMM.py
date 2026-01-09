import torch
import torch.nn as nn

class Model(nn.Module):
    """
    FP8 GEMM
    
    Used by: FP8 training and inference
    
    FP8 (E4M3/E5M2) matrix multiplication with per-tensor or
    per-channel scaling. Simulates FP8 computation on FP16 hardware.
    
    Shapes:
        Input: (batch, seq_len, in_features)
        Output: (batch, seq_len, out_features)
    """
    
    def __init__(self, in_features: int, out_features: int, use_bias: bool = False):
        """
        Initialize FP8 GEMM layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            use_bias: Whether to use bias
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # FP16 weights (simulating FP8 storage)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        
        # Per-tensor scales for FP8 quantization
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('input_scale', torch.tensor(1.0))
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
    
    def _quantize_fp8(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Simulate FP8 quantization by clamping to FP8 range."""
        # FP8 E4M3 range: approximately [-448, 448]
        fp8_max = 448.0
        
        # Scale and clamp
        x_scaled = x / scale
        x_clamped = torch.clamp(x_scaled, -fp8_max, fp8_max)
        
        # Simulate quantization noise (rounding)
        # In real FP8, this would be actual format conversion
        x_quantized = x_clamped.to(torch.float16)
        
        return x_quantized * scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FP8 GEMM forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, in_features)
            
        Returns:
            Output tensor (batch, seq_len, out_features)
        """
        # Update input scale based on tensor statistics (dynamic quantization)
        with torch.no_grad():
            input_amax = x.abs().max()
            self.input_scale = input_amax / 448.0
        
        # Quantize inputs (simulated)
        x_fp8 = self._quantize_fp8(x, self.input_scale)
        
        # Quantize weights (simulated)
        w_fp8 = self._quantize_fp8(self.weight, self.weight_scale)
        
        # Matrix multiplication
        output = torch.nn.functional.linear(x_fp8, w_fp8, self.bias)
        
        return output


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
in_features = 4096
out_features = 4096

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, in_features, device='cuda', dtype=torch.float16)
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [in_features, out_features]

