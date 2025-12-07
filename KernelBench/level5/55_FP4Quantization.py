import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    FP4 (4-bit Floating Point) Quantization.
    
    Implements NF4 (Normal Float 4) and FP4 quantization
    used in QLoRA and efficient inference.
    
    Based on: "QLoRA: Efficient Finetuning of Quantized LLMs"
    """
    def __init__(self, in_features, out_features, use_nf4=True, block_size=64):
        """
        :param in_features: Input dimension
        :param out_features: Output dimension
        :param use_nf4: Use NF4 (normal float) vs standard FP4
        :param block_size: Block size for quantization
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_nf4 = use_nf4
        self.block_size = block_size
        
        # Number of blocks
        self.num_blocks = (in_features * out_features + block_size - 1) // block_size
        
        # NF4 quantization levels (optimized for normal distribution)
        if use_nf4:
            # 16 levels optimized for normally distributed weights
            nf4_levels = torch.tensor([
                -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
            ])
        else:
            # Uniform 4-bit levels
            nf4_levels = torch.linspace(-1, 1, 16)
        
        self.register_buffer('quant_levels', nf4_levels)
        
        # Full precision weight (for quantization)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        
        # Scales per block
        self.register_buffer('scales', torch.ones(self.num_blocks))
        
        # Quantized weight (4-bit packed into int8)
        # Each byte holds 2 4-bit values
        num_bytes = (out_features * in_features + 1) // 2
        self.register_buffer('quant_weight', torch.zeros(num_bytes, dtype=torch.uint8))
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def _quantize_block(self, block, scale):
        """Quantize a block of weights to 4-bit."""
        # Normalize by scale
        normalized = block / scale
        
        # Clamp to valid range
        normalized = normalized.clamp(-1, 1)
        
        # Find nearest quantization level
        distances = (normalized.unsqueeze(-1) - self.quant_levels.unsqueeze(0)).abs()
        indices = distances.argmin(dim=-1)
        
        return indices.byte()
    
    def _dequantize_block(self, indices, scale):
        """Dequantize a block of weights from 4-bit."""
        # Look up quantization levels
        values = self.quant_levels[indices.long()]
        
        # Apply scale
        return values * scale
    
    def quantize_weights(self):
        """Quantize the full weight matrix."""
        with torch.no_grad():
            weight_flat = self.weight.view(-1)
            
            # Process in blocks
            quant_indices = []
            scales = []
            
            for i in range(0, weight_flat.numel(), self.block_size):
                block = weight_flat[i:i + self.block_size]
                
                # Compute block scale (absmax)
                scale = block.abs().max().clamp(min=1e-8)
                scales.append(scale)
                
                # Quantize block
                indices = self._quantize_block(block, scale)
                quant_indices.append(indices)
            
            # Store scales
            self.scales = torch.stack(scales)
            
            # Pack 4-bit indices into bytes
            all_indices = torch.cat(quant_indices)
            # Pad to even length
            if all_indices.numel() % 2 == 1:
                all_indices = F.pad(all_indices.float(), (0, 1)).byte()
            
            # Pack pairs of 4-bit values into bytes
            packed = (all_indices[0::2] & 0x0F) | ((all_indices[1::2] & 0x0F) << 4)
            self.quant_weight = packed
    
    def dequantize_weights(self):
        """Dequantize weights to full precision."""
        # Unpack bytes to 4-bit indices
        low = self.quant_weight & 0x0F
        high = (self.quant_weight >> 4) & 0x0F
        
        indices = torch.stack([low, high], dim=1).view(-1)
        indices = indices[:self.in_features * self.out_features]
        
        # Dequantize blocks
        weight_flat = torch.zeros(self.in_features * self.out_features, 
                                  device=self.weight.device)
        
        for i, block_start in enumerate(range(0, weight_flat.numel(), self.block_size)):
            block_end = min(block_start + self.block_size, weight_flat.numel())
            block_indices = indices[block_start:block_end]
            weight_flat[block_start:block_end] = self._dequantize_block(
                block_indices, self.scales[i]
            )
        
        return weight_flat.view(self.out_features, self.in_features)
    
    def forward(self, x):
        """
        Forward pass with dequantized weights.
        
        :param x: Input tensor (..., in_features)
        :return: Output tensor (..., out_features)
        """
        # Dequantize weights
        weight = self.dequantize_weights()
        
        # Linear transformation
        return F.linear(x, weight, self.bias)


# Test parameters
batch_size = 16
seq_len = 256
in_features = 4096
out_features = 4096

def get_inputs():
    return [torch.randn(batch_size, seq_len, in_features)]

def get_init_inputs():
    return [in_features, out_features, True]  # use_nf4=True

