import torch
from typing import List

# Distribution functions: {name: sampler_fn}
DISTRIBUTIONS = {
    # D1: N(0, 1)
    "normal": lambda size, dtype, device: torch.randn(size, dtype=dtype, device=device),
    # D2: U(-1, 1)
    "uniform_sym": lambda size, dtype, device: torch.empty(size, dtype=dtype, device=device).uniform_(-1, 1),
    # D3: U(0, 1)
    "uniform_pos": lambda size, dtype, device: torch.rand(size, dtype=dtype, device=device),
    # D4: randint [0, index_max)
    "indices": lambda size, index_max, dtype, device: torch.randint(0, index_max, size, dtype=dtype, device=device),
    # D5: N(0, 10) for softmax stability
    "normal_scaled": lambda size, dtype, device: torch.randn(size, dtype=dtype, device=device) * 10,
    # D6: N(0,1) with 0.1% at ±10σ
    "normal_outliers": lambda size, dtype, device: (lambda t, m: torch.where(m, torch.sign(t) * 10, t))(torch.randn(size, dtype=dtype, device=device), torch.rand(size, device=device) < 0.001),
    # D7: U(-0.5, 0.5) for cumprod safety
    "uniform_bounded": lambda size, dtype, device: torch.empty(size, dtype=dtype, device=device).uniform_(-0.5, 0.5),
}


def get_supported_distributions(category: str, op_name: str) -> List[str]:
    """Return list of distribution names for a given operator category and name."""
    
    SPECIAL_CASES = {
        # Softmax stability
        "5_Softmax": ["normal", "normal_scaled"],
        "6_LogSoftmax": ["normal", "normal_scaled"],
        
        # Integer indices
        "6_KVCache_Operations": ["indices"],
        "7_PagedAttention": ["normal", "indices"],
        
        # Quantization with outliers
        "3_KVCache_Quantize": ["normal", "normal_outliers"],
        "5_AWQQuantize": ["normal", "normal_outliers"],
        "6_GPTQDequant": ["normal", "normal_outliers"],
        
        # Bounded for overflow safety
        "1_Sum": ["uniform_sym", "uniform_bounded"],
        "7_Cumsum": ["uniform_bounded"],
        "8_Cumprod": ["uniform_bounded"],
        
        # Router/NMS probabilities
        "1_TopK_Router": ["uniform_pos"],
        "8_NMS": ["uniform_pos"],
    }
    
    if op_name in SPECIAL_CASES:
        return SPECIAL_CASES[op_name]
    
    CATEGORY_DEFAULTS = {
        "activations":        ["normal", "uniform_sym"],
        "attention":          ["normal", "uniform_sym"],
        "audio":              ["normal", "uniform_sym"],
        "communication":      ["normal"],
        "convolutions":       ["normal", "uniform_sym"],
        "detection":          ["normal", "uniform_pos"],
        "diffusion":          ["normal"],
        "embeddings":         ["indices"],
        "linear_projections": ["normal"],
        "loss":               ["normal", "uniform_pos"],
        "matmul":             ["normal", "uniform_sym"],
        "mobile":             ["normal", "uniform_sym"],
        "model_merging":      ["normal"],
        "moe":                ["normal", "uniform_pos"],
        "normalization":      ["normal", "uniform_sym"],
        "optimizers":         ["normal"],
        "peft":               ["normal"],
        "pooling":            ["normal", "uniform_sym"],
        "quantization":       ["normal", "uniform_sym"],
        "recommendation":     ["indices", "normal"],
        "reductions":         ["normal", "uniform_sym"],
        "sampling":           ["normal", "uniform_pos"],
        "speculative":        ["normal", "uniform_pos"],
        "ssm":                ["normal"],
        "upsampling":         ["normal", "uniform_pos"],
        "vae":                ["normal", "uniform_pos"],
        "vision":             ["normal", "uniform_pos"],
    }
    
    return CATEGORY_DEFAULTS.get(category, ["normal"])
