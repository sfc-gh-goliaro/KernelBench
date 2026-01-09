import torch
import torch.nn as nn

class Model(nn.Module):
    """
    QLoRA Linear Layer
    
    Used by: QLoRA finetuning
    
    QLoRA: 4-bit quantized base weights + FP16 LoRA adapter path.
    Enables finetuning of large models on consumer GPUs.
    
    Shapes:
        Input: (batch, seq_len, in_features)
        Output: (batch, seq_len, out_features)
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16,
                 alpha: float = 16.0, group_size: int = 128):
        """
        Initialize QLoRA linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: LoRA rank
            alpha: LoRA alpha for scaling
            group_size: Quantization group size for base weights
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.group_size = group_size
        
        # 4-bit quantized base weights (simulated)
        num_groups = in_features // group_size
        self.register_buffer('base_qweight', 
            torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer('base_scales',
            torch.ones(num_groups, out_features))
        self.register_buffer('base_zeros',
            torch.zeros(num_groups, out_features, dtype=torch.int8))
        
        # FP16 LoRA adapters
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Initialize LoRA
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
    
    def _dequantize_base(self) -> torch.Tensor:
        """Dequantize 4-bit base weights."""
        # Unpack 4-bit values
        w_low = self.base_qweight & 0x0F
        w_high = self.base_qweight >> 4
        weights = torch.stack([w_low, w_high], dim=-1).view(self.out_features, self.in_features)
        weights = weights.float()
        
        # Dequantize per group
        for g in range(self.in_features // self.group_size):
            start = g * self.group_size
            end = start + self.group_size
            weights[:, start:end] = (weights[:, start:end] - self.base_zeros[g].unsqueeze(1).float()) * self.base_scales[g].unsqueeze(1)
        
        return weights.T
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        QLoRA forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, in_features)
            
        Returns:
            Output tensor (batch, seq_len, out_features)
        """
        # Dequantize and apply base weights
        base_weight = self._dequantize_base()
        base_output = torch.matmul(x, base_weight)
        
        # FP16 LoRA path
        lora_output = self.lora_B(self.lora_A(x))
        
        return base_output + self.scaling * lora_output


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
in_features = 4096
out_features = 4096
rank = 16

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, in_features, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [in_features, out_features, rank]

