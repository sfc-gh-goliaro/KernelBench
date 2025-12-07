import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused BitNet (1.58-bit) GEMM.
    
    Implements BitNet b1.58 where weights are constrained to {-1, 0, +1}.
    This enables extremely efficient computation using only additions
    and subtractions instead of multiplications.
    
    The fused kernel:
    1. Loads ternary weights (2 bits per weight, packed)
    2. Performs add/subtract operations based on weight values
    3. Applies scaling factor
    
    Reference: BitNet, BitNet b1.58
    """
    def __init__(self, in_features, out_features):
        """
        :param in_features: Input dimension
        :param out_features: Output dimension
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Ternary weights packed (4 weights per byte using 2 bits each)
        # Values: 00 = 0, 01 = +1, 10 = -1
        packed_size = (in_features * out_features + 3) // 4
        self.qweight = nn.Parameter(
            torch.randint(0, 256, (packed_size,), dtype=torch.uint8),
            requires_grad=False
        )
        
        # Per-row scaling factor (to recover magnitude)
        self.scale = nn.Parameter(torch.ones(out_features))
        
        # Input normalization (RMSNorm before BitLinear)
        self.input_norm = nn.Parameter(torch.ones(in_features))
        self.eps = 1e-6
    
    def _unpack_ternary(self):
        """Unpack ternary weights from packed format."""
        # Extract 2-bit values
        w0 = (self.qweight & 0x03)
        w1 = ((self.qweight >> 2) & 0x03)
        w2 = ((self.qweight >> 4) & 0x03)
        w3 = ((self.qweight >> 6) & 0x03)
        
        # Concatenate and reshape
        unpacked = torch.stack([w0, w1, w2, w3], dim=-1).view(-1)
        unpacked = unpacked[:self.in_features * self.out_features]
        unpacked = unpacked.view(self.out_features, self.in_features)
        
        # Convert to ternary: 0->0, 1->+1, 2->-1
        ternary = torch.zeros_like(unpacked, dtype=torch.float32)
        ternary[unpacked == 1] = 1.0
        ternary[unpacked == 2] = -1.0
        
        return ternary
    
    def _absmax_quantize_input(self, x):
        """Quantize input to INT8 using absmax."""
        absmax = x.abs().max(dim=-1, keepdim=True).values.clamp(min=self.eps)
        scale = 127.0 / absmax
        x_quant = (x * scale).round().clamp(-128, 127)
        return x_quant, scale
    
    def forward(self, x):
        """
        Fused BitNet GEMM forward.
        
        :param x: Input (*, in_features)
        :return: Output (*, out_features)
        """
        batch_shape = x.shape[:-1]
        x = x.view(-1, self.in_features)
        
        # === FUSED KERNEL START ===
        # Step 1: RMSNorm input
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps) * self.input_norm
        
        # Step 2: Quantize input to INT8
        x_quant, x_scale = self._absmax_quantize_input(x_norm)
        
        # Step 3: Ternary matmul (in fused kernel, this is add/subtract)
        weights = self._unpack_ternary()
        
        # In a fused kernel, we would:
        # - For +1: add the activation
        # - For -1: subtract the activation  
        # - For 0: skip
        # This avoids multiplications entirely
        
        # Reference implementation using standard matmul
        output = F.linear(x_quant.float(), weights)
        
        # Step 4: Apply scales
        output = output / x_scale  # Undo input quantization scale
        output = output * self.scale  # Apply weight scale
        # === FUSED KERNEL END ===
        
        return output.view(*batch_shape, self.out_features)


# Efficient ternary operation implementation
class BitNetTernaryGEMM(nn.Module):
    """
    Ternary GEMM using efficient add/subtract operations.
    
    Stores positive and negative masks separately for efficient computation.
    """
    def __init__(self, in_features, out_features):
        super(BitNetTernaryGEMM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store as separate masks for +1 and -1
        # This enables: output = x @ pos_mask - x @ neg_mask
        self.pos_mask = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.bool),
            requires_grad=False
        )
        self.neg_mask = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.bool),
            requires_grad=False
        )
        self.scale = nn.Parameter(torch.ones(out_features))
    
    def init_from_float_weights(self, float_weights):
        """Initialize ternary weights from float weights."""
        # Quantize to ternary using threshold
        threshold = float_weights.abs().mean() * 0.5
        
        self.pos_mask.data = (float_weights > threshold)
        self.neg_mask.data = (float_weights < -threshold)
        self.scale.data = float_weights.abs().mean(dim=-1)
    
    def forward(self, x):
        """
        Efficient ternary matmul using masks.
        
        output[i] = sum(x[j] for j where pos[i,j]) - sum(x[j] for j where neg[i,j])
        """
        batch_shape = x.shape[:-1]
        x = x.view(-1, self.in_features)
        
        # === FUSED KERNEL ===
        # This can be implemented as:
        # 1. Masked sum for positive weights
        # 2. Masked sum for negative weights
        # 3. Subtract and scale
        
        # Using einsum with bool masks (converted to float)
        pos_contribution = x @ self.pos_mask.float().t()
        neg_contribution = x @ self.neg_mask.float().t()
        
        output = (pos_contribution - neg_contribution) * self.scale
        
        return output.view(*batch_shape, self.out_features)


# BitNet with absmax input quantization fused
class FusedBitNetWithQuantization(nn.Module):
    """
    Full BitNet layer with fused input quantization.
    """
    def __init__(self, in_features, out_features, eps=1e-6):
        super(FusedBitNetWithQuantization, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        
        # Ternary weights (using float for reference)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.scale = nn.Parameter(torch.ones(out_features))
        self.input_norm = nn.Parameter(torch.ones(in_features))
    
    def _ternary_weight(self):
        """Get ternary quantized weights."""
        absmax = self.weight.abs().mean()
        ternary = torch.sign(self.weight) * (self.weight.abs() > absmax * 0.5).float()
        return ternary
    
    def forward(self, x):
        batch_shape = x.shape[:-1]
        x = x.view(-1, self.in_features)
        
        # Fused: RMSNorm + Quantize + Ternary Matmul
        # RMSNorm
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps) * self.input_norm
        
        # INT8 quantization
        absmax = x.abs().max(dim=-1, keepdim=True).values.clamp(min=self.eps)
        x_quant = (x / absmax * 127).round().clamp(-128, 127)
        
        # Ternary matmul
        w_ternary = self._ternary_weight()
        output = F.linear(x_quant, w_ternary)
        
        # Rescale
        output = output * absmax / 127 * self.scale
        
        return output.view(*batch_shape, self.out_features)


# Test parameters
batch_size = 32
seq_len = 512
in_features = 4096
out_features = 4096

def get_inputs():
    return [torch.randn(batch_size, seq_len, in_features)]

def get_init_inputs():
    return [in_features, out_features]

