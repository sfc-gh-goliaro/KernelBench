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
    Fused INT8 GEMM with W8A8 Quantization.
    
    Both weights AND activations are INT8 quantized:
    - Weights: statically quantized to INT8 with per-channel or per-tensor scale
    - Activations: dynamically quantized to INT8 before GEMM
    
    The fusion:
    1. Quantize input activations to INT8 (dynamic)
    2. Perform INT8 GEMM (using INT8 tensor cores)
    3. Dequantize output (or keep quantized for next layer)
    
    This provides 2-4x speedup over FP16 on modern GPUs (A100, H100).
    
    Reference: SGLang W8A8, LLM.int8(), SmoothQuant
    """
    def __init__(self, in_features, out_features, per_channel=True, 
                 symmetric=True):
        """
        :param in_features: Input dimension
        :param out_features: Output dimension
        :param per_channel: Use per-channel weight quantization
        :param symmetric: Use symmetric quantization (no zero-point)
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.per_channel = per_channel
        self.symmetric = symmetric
        
        # Quantized weights (stored as INT8)
        self.weight_int8 = nn.Parameter(
            torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8),
            requires_grad=False
        )
        
        # Weight scales
        if per_channel:
            self.weight_scale = nn.Parameter(torch.ones(out_features))
        else:
            self.weight_scale = nn.Parameter(torch.tensor(1.0))
        
        # Optional zero-point for asymmetric
        if not symmetric:
            if per_channel:
                self.weight_zp = nn.Parameter(
                    torch.zeros(out_features, dtype=torch.int8),
                    requires_grad=False
                )
            else:
                self.weight_zp = nn.Parameter(
                    torch.tensor(0, dtype=torch.int8),
                    requires_grad=False
                )
        else:
            self.register_buffer('weight_zp', None)
    
    def _quantize_activation(self, x):
        """
        Dynamic INT8 quantization of activations.
        
        :param x: FP16/FP32 activations
        :return: (x_int8, scale)
        """
        # Per-token quantization for better accuracy
        absmax = x.abs().max(dim=-1, keepdim=True).values
        scale = 127.0 / absmax.clamp(min=1e-12)
        
        x_int8 = (x * scale).round().clamp(-128, 127).to(torch.int8)
        
        return x_int8, 1.0 / scale
    
    def _int8_gemm(self, a_int8, b_int8, a_scale, b_scale):
        """
        INT8 GEMM with scale correction.
        
        output = (a_int8 @ b_int8) * a_scale * b_scale
        """
        # In practice, this uses INT8 tensor cores
        # Here we simulate with float conversion
        a_float = a_int8.float()
        b_float = b_int8.float()
        
        output = F.linear(a_float, b_float)
        
        # Apply scales
        if self.per_channel:
            # b_scale is per-output-channel
            output = output * a_scale * b_scale
        else:
            output = output * a_scale * b_scale
        
        return output
    
    def forward(self, x, return_int8=False):
        """
        Fused W8A8 GEMM forward.
        
        :param x: Input activation (*, in_features)
        :param return_int8: Return INT8 output (for next W8A8 layer)
        :return: Output (or (output_int8, scale) if return_int8)
        """
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        
        # === FUSED KERNEL START ===
        # Step 1: Quantize activations
        x_int8, x_scale = self._quantize_activation(x)
        
        # Step 2: INT8 GEMM
        output = self._int8_gemm(
            x_int8, 
            self.weight_int8,
            x_scale,
            self.weight_scale
        )
        # === FUSED KERNEL END ===
        
        output = output.view(*original_shape[:-1], self.out_features)
        
        if return_int8:
            # Quantize output for next layer
            out_int8, out_scale = self._quantize_activation(output)
            return out_int8, out_scale
        
        return output
    
    @classmethod
    def from_float(cls, float_linear, per_channel=True):
        """
        Create W8A8 layer from float Linear.
        
        :param float_linear: nn.Linear with float weights
        :param per_channel: Use per-channel quantization
        :return: FusedW8A8GEMM instance
        """
        in_features = float_linear.in_features
        out_features = float_linear.out_features
        
        module = cls(in_features, out_features, per_channel)
        
        # Quantize weights
        weight = float_linear.weight.data
        
        if per_channel:
            absmax = weight.abs().max(dim=1).values
            scale = 127.0 / absmax.clamp(min=1e-12)
            weight_int8 = (weight * scale.unsqueeze(1)).round().clamp(-128, 127)
            module.weight_int8.data = weight_int8.to(torch.int8)
            module.weight_scale.data = 1.0 / scale
        else:
            absmax = weight.abs().max()
            scale = 127.0 / absmax.clamp(min=1e-12)
            weight_int8 = (weight * scale).round().clamp(-128, 127)
            module.weight_int8.data = weight_int8.to(torch.int8)
            module.weight_scale.data = torch.tensor(1.0 / scale)
        
        return module


# SmoothQuant-style W8A8
class FusedW8A8SmoothQuant(nn.Module):
    """
    W8A8 GEMM with SmoothQuant migration.
    
    Applies smooth factor to balance weight/activation quantization difficulty.
    """
    def __init__(self, in_features, out_features, smooth_factor=None):
        super(FusedW8A8SmoothQuant, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Smooth factor (per input channel)
        if smooth_factor is not None:
            self.register_buffer('smooth_factor', smooth_factor)
        else:
            self.register_buffer('smooth_factor', torch.ones(in_features))
        
        # Quantized weights (after smooth adjustment)
        self.weight_int8 = nn.Parameter(
            torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8),
            requires_grad=False
        )
        self.weight_scale = nn.Parameter(torch.ones(out_features))
    
    def forward(self, x):
        """SmoothQuant W8A8 forward."""
        # Apply smooth factor to activation
        x_smooth = x / self.smooth_factor
        
        # Quantize smoothed activation
        absmax = x_smooth.abs().max(dim=-1, keepdim=True).values
        x_scale = 127.0 / absmax.clamp(min=1e-12)
        x_int8 = (x_smooth * x_scale).round().clamp(-128, 127)
        
        # INT8 GEMM
        output = F.linear(x_int8.float(), self.weight_int8.float())
        output = output * (1.0 / x_scale) * self.weight_scale
        
        return output


# Test parameters
batch_size = 32
seq_len = 512
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

