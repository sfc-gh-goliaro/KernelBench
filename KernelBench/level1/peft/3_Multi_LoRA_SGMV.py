import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Multi-LoRA SGMV (Segmented Grouped Matrix-Vector multiplication)
    
    Used by: Multi-tenant LoRA serving
    
    SGMV kernel for batched inference with multiple LoRA adapters
    where different sequences in the batch use different adapters.
    
    Shapes:
        Input: (batch, seq_len, in_features)
        adapter_indices: (batch,) which adapter each sequence uses
        Output: (batch, seq_len, out_features)
    """
    
    def __init__(self, in_features: int, out_features: int, num_adapters: int = 8,
                 rank: int = 16, alpha: float = 16.0):
        """
        Initialize Multi-LoRA layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            num_adapters: Number of different LoRA adapters
            rank: LoRA rank
            alpha: LoRA alpha for scaling
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_adapters = num_adapters
        self.rank = rank
        self.scaling = alpha / rank
        
        # Base linear (shared across all adapters)
        self.base_linear = nn.Linear(in_features, out_features, bias=False)
        
        # Multiple LoRA adapters: (num_adapters, rank, in_features) and (num_adapters, out_features, rank)
        self.lora_A = nn.Parameter(torch.randn(num_adapters, rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(num_adapters, out_features, rank))
    
    def forward(self, x: torch.Tensor, adapter_indices: torch.Tensor) -> torch.Tensor:
        """
        Multi-LoRA forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, in_features)
            adapter_indices: Adapter index per sequence (batch,)
            
        Returns:
            Output tensor (batch, seq_len, out_features)
        """
        batch_size, seq_len, _ = x.shape
        
        # Base model output
        base_output = self.base_linear(x)
        
        # Apply different LoRA adapters per sequence
        lora_output = torch.zeros_like(base_output)
        
        for i in range(self.num_adapters):
            # Find sequences using this adapter
            mask = adapter_indices == i
            if not mask.any():
                continue
            
            # Get sequences for this adapter
            x_adapter = x[mask]  # (num_seqs, seq_len, in_features)
            
            # Apply this adapter's LoRA: x @ A^T @ B^T
            # A: (rank, in_features), B: (out_features, rank)
            intermediate = torch.matmul(x_adapter, self.lora_A[i].T)  # (num_seqs, seq_len, rank)
            adapter_out = torch.matmul(intermediate, self.lora_B[i].T)  # (num_seqs, seq_len, out_features)
            
            lora_output[mask] = adapter_out
        
        return base_output + self.scaling * lora_output


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 32
seq_length = 512
in_features = 4096
out_features = 4096
num_adapters = 8
rank = 16

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, in_features, device='cuda')
    adapter_indices = torch.randint(0, num_adapters, (batch_size,), device='cuda')
    return [x, adapter_indices]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [in_features, out_features, num_adapters, rank]

