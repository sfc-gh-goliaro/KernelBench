"""
Base Infrastructure for Level4 Models

This module provides base classes and utilities for all level4 model implementations:
- BaseLevel4Model: Abstract base class with common interface
- ModelConfig: Dataclass for model configuration
- Operator composition helpers for importing level1/2/3 modules
"""

import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum


@dataclass
class ModelConfig:
    """Base configuration for level4 models."""
    # Common parameters
    hidden_size: int = 4096
    num_layers: int = 32
    vocab_size: int = 32000
    max_seq_len: int = 8192
    
    # Attention parameters
    num_heads: int = 32
    num_kv_heads: Optional[int] = None  # For GQA; None means MHA
    head_dim: Optional[int] = None  # Computed if not provided
    
    # FFN parameters
    intermediate_size: int = 11008
    
    # Normalization
    rms_norm_eps: float = 1e-6
    
    # Position encoding
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # HuggingFace model ID for config loading
    hf_model_id: Optional[str] = None
    
    # Additional model-specific parameters
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        # Separate known fields from extra
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        known = {k: v for k, v in config_dict.items() if k in known_fields}
        extra = {k: v for k, v in config_dict.items() if k not in known_fields}
        if extra:
            known['extra_config'] = extra
        return cls(**known)


class OperatorLevel(Enum):
    """Enum for operator implementation levels."""
    LEVEL1 = 1  # Individual operators
    LEVEL2 = 2  # Fused operators
    LEVEL3 = 3  # Entire blocks


class BaseLevel4Model(nn.Module, ABC):
    """
    Abstract base class for all level4 models.
    
    Provides common interface for:
    - Model configuration and variant support
    - HuggingFace config loading
    - Validation against reference implementations
    - Benchmarking utilities
    """
    
    # Class-level variant configurations (override in subclasses)
    VARIANTS: Dict[str, Dict[str, Any]] = {}
    
    # Default operator level to use
    DEFAULT_OPERATOR_LEVEL: OperatorLevel = OperatorLevel.LEVEL3
    
    def __init__(self, config: ModelConfig, operator_level: Optional[OperatorLevel] = None):
        super().__init__()
        self.config = config
        self.operator_level = operator_level or self.DEFAULT_OPERATOR_LEVEL
    
    @classmethod
    def from_variant(
        cls,
        variant: str,
        operator_level: Optional[OperatorLevel] = None,
        **override_kwargs
    ) -> "BaseLevel4Model":
        """
        Create model from a predefined variant.
        
        Args:
            variant: Variant name (e.g., "8B", "70B")
            operator_level: Which level of operators to use
            **override_kwargs: Override specific config parameters
            
        Returns:
            Initialized model instance
        """
        if variant not in cls.VARIANTS:
            available = list(cls.VARIANTS.keys())
            raise ValueError(f"Unknown variant '{variant}'. Available: {available}")
        
        config_dict = cls.VARIANTS[variant].copy()
        config_dict.update(override_kwargs)
        config = ModelConfig.from_dict(config_dict)
        
        return cls(config, operator_level)
    
    @classmethod
    def from_pretrained_config(
        cls,
        model_name_or_path: str,
        operator_level: Optional[OperatorLevel] = None,
        **override_kwargs
    ) -> "BaseLevel4Model":
        """
        Create model with config loaded from HuggingFace.
        
        Args:
            model_name_or_path: HuggingFace model ID or path
            operator_level: Which level of operators to use
            **override_kwargs: Override specific config parameters
            
        Returns:
            Initialized model instance
        """
        from .config_loader import load_hf_config
        
        config_dict = load_hf_config(model_name_or_path)
        config_dict.update(override_kwargs)
        config = ModelConfig.from_dict(config_dict)
        
        return cls(config, operator_level)
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses."""
        pass
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Get total number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_footprint(self) -> int:
        """Get approximate memory footprint in bytes."""
        return sum(p.numel() * p.element_size() for p in self.parameters())


class ValidationResult:
    """Container for validation results."""
    
    def __init__(
        self,
        is_valid: bool,
        max_diff: float,
        mean_diff: float,
        rtol: float,
        atol: float,
        details: Optional[Dict[str, Any]] = None
    ):
        self.is_valid = is_valid
        self.max_diff = max_diff
        self.mean_diff = mean_diff
        self.rtol = rtol
        self.atol = atol
        self.details = details or {}
    
    def __repr__(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        return (
            f"ValidationResult({status}, max_diff={self.max_diff:.2e}, "
            f"mean_diff={self.mean_diff:.2e})"
        )


# ============================================================================
# Level1 Operator Import Utility
# ============================================================================

def import_level1_operator(category: str, filename: str):
    """
    Import a level1 operator module by category and filename.
    
    Args:
        category: The subfolder under level1 (e.g., "normalization", "activations")
        filename: The filename including .py (e.g., "4_RMSNorm.py")
        
    Returns:
        The Model class from the specified module
        
    Example:
        >>> RMSNorm = import_level1_operator("normalization", "_4_RMSNorm.py")
        >>> norm = RMSNorm(hidden_size=4096)
    """
    import importlib.util
    
    level1_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "level1", category, filename
    )
    
    if not os.path.exists(level1_path):
        raise FileNotFoundError(f"Level1 operator not found: {level1_path}")
    
    spec = importlib.util.spec_from_file_location(filename[:-3], level1_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


# Pre-defined level1 operator registry for easy access
LEVEL1_OPERATORS = {
    # Normalization
    "RMSNorm": ("normalization", "4_RMSNorm.py"),
    "LayerNorm": ("normalization", "6_LayerNorm.py"),
    "BatchNorm": ("normalization", "1_BatchNorm.py"),
    "GroupNorm": ("normalization", "3_GroupNorm.py"),
    
    # Activations
    "ReLU": ("activations", "1_ReLU.py"),
    "GELU": ("activations", "8_GELU.py"),
    "Swish": ("activations", "7_Swish.py"),
    "SiLU": ("activations", "7_Swish.py"),  # Alias
    "Softmax": ("activations", "5_Softmax.py"),
    "Sigmoid": ("activations", "3_Sigmoid.py"),
    "Tanh": ("activations", "4_Tanh.py"),
    "SiluAndMul": ("activations", "16_SiluAndMul.py"),
    "GeluAndMul": ("activations", "17_GeluAndMul.py"),
    
    # Embeddings
    "RotaryEmbedding": ("embeddings", "1_RotaryEmbedding.py"),
    "Embedding": ("embeddings", "2_Embedding.py"),
    "SinusoidalPosEmbed": ("embeddings", "3_SinusoidalPosEmbed.py"),
    "RelativePositionBias": ("embeddings", "4_RelativePositionBias.py"),
    
    # Attention
    "ScaledDotProductAttention": ("attention", "1_ScaledDotProductAttention.py"),
    "MultiHeadAttention": ("attention", "2_MultiHeadAttention_Causal.py"),
    "GroupedQueryAttention": ("attention", "3_GroupedQueryAttention.py"),
    "MultiQueryAttention": ("attention", "4_MultiQueryAttention.py"),
    "MultiHeadLatentAttention": ("attention", "5_MultiHeadLatentAttention.py"),
    "ALiBi": ("attention", "6_ALiBi.py"),
    "SlidingWindowAttention": ("attention", "7_SlidingWindowAttention.py"),
    
    # MatMul
    "MatMul": ("matmul", "1_MatMul.py"),
    "BatchedMatMul": ("matmul", "2_BatchedMatMul.py"),
    
    # Pooling
    "MaxPool2d": ("pooling", "2_MaxPool2d.py"),
    "AvgPool2d": ("pooling", "5_AvgPool2d.py"),
    "AdaptiveAvgPool2d": ("pooling", "7_AdaptiveAvgPool2d.py"),
    "GlobalAveragePooling": ("pooling", "8_GlobalAveragePooling.py"),
    "MeanPooling": ("pooling", "9_MeanPooling.py"),
    
    # Convolutions
    "Conv2d": ("convolutions", "1_Conv2d_Standard.py"),
    "DepthwiseConv2d": ("convolutions", "30_DepthwiseConv2d_Square.py"),
    "DepthwiseSeparable": ("convolutions", "34_DepthwiseSeparable.py"),
    
    # Mobile
    "SqueezeExcitation": ("mobile", "2_SqueezeExcitation.py"),
    "MBConv": ("mobile", "5_MBConv.py"),
    
    # MoE
    "TopKRouter": ("moe", "1_TopK_Router.py"),
    "ExpertDispatch": ("moe", "2_Expert_Dispatch.py"),
    "FusedMoE": ("moe", "3_FusedMoE.py"),
    
    # SSM
    "SelectiveScan": ("ssm", "1_SelectiveScan.py"),
    
    # Detection
    "CSPBlock": ("detection", "1_CSPBlock.py"),
    "SPPF": ("detection", "3_SPPF.py"),
    "NMS": ("detection", "8_NMS.py"),
    
    # Speculative
    "TreeAttention": ("speculative", "1_TreeAttention.py"),
    "VerificationSampling": ("speculative", "2_VerificationSampling.py"),
    "DraftHead": ("speculative", "3_DraftHead.py"),
    "JacobiIteration": ("speculative", "6_JacobiIteration.py"),
    "EarlyExit": ("speculative", "7_EarlyExit.py"),
    
    # Model Merging
    "TIES": ("model_merging", "3_TIES.py"),
    "TaskArithmetic": ("model_merging", "5_TaskArithmetic.py"),
    
    # Rendering
    "GaussianSplatting": ("rendering", "1_GaussianSplat.py"),
    "HashEncoding": ("rendering", "2_HashEncoding.py"),
}


def get_level1_operator(name: str):
    """
    Get a level1 operator by its registered name.
    
    Args:
        name: Operator name from LEVEL1_OPERATORS registry
        
    Returns:
        The Model class for the operator
        
    Example:
        >>> RMSNorm = get_level1_operator("RMSNorm")
        >>> norm = RMSNorm(hidden_size=4096)
    """
    if name not in LEVEL1_OPERATORS:
        raise ValueError(f"Unknown operator: {name}. Available: {list(LEVEL1_OPERATORS.keys())}")
    
    category, filename = LEVEL1_OPERATORS[name]
    return import_level1_operator(category, filename)


# ============================================================================
# Operator Composition Utilities (Legacy)
# ============================================================================

def get_operator_module(level: OperatorLevel, category: str, name: str):
    """
    Import an operator module from level1/2/3.
    
    Args:
        level: OperatorLevel.LEVEL1, LEVEL2, or LEVEL3
        category: Category subfolder (e.g., "llm", "normalization", "ffn")
        name: Module name (e.g., "3_LlamaDecoderLayer", "4_RMSNorm")
        
    Returns:
        The Model class from the specified module
        
    Example:
        >>> LlamaLayer = get_operator_module(OperatorLevel.LEVEL3, "llm", "3_LlamaDecoderLayer")
        >>> layer = LlamaLayer(hidden_size=4096, num_heads=32, ...)
    """
    import importlib
    import os
    
    level_map = {
        OperatorLevel.LEVEL1: "level1",
        OperatorLevel.LEVEL2: "level2",
        OperatorLevel.LEVEL3: "level3",
    }
    
    level_str = level_map.get(level)
    if level_str is None:
        raise ValueError(f"Invalid operator level: {level}")
    
    # Construct module path
    module_path = f"KernelBench.{level_str}.{category}.{name}"
    
    try:
        module = importlib.import_module(module_path)
        return module.Model
    except ImportError as e:
        # Try with relative import from workspace root
        try:
            import sys
            workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if workspace_root not in sys.path:
                sys.path.insert(0, workspace_root)
            module = importlib.import_module(module_path)
            return module.Model
        except ImportError:
            raise ImportError(f"Could not import operator: {module_path}. Error: {e}")


# Pre-defined operator mappings for common components
OPERATOR_MAPPINGS = {
    # Level 3 blocks
    "LlamaDecoderLayer": (OperatorLevel.LEVEL3, "llm", "3_LlamaDecoderLayer"),
    "MixtralMoEBlock": (OperatorLevel.LEVEL3, "moe", "1_MixtralMoEBlock"),
    "DeepSeekV2Block": (OperatorLevel.LEVEL3, "moe", "2_DeepSeekV2Block"),
    "Mamba1Block": (OperatorLevel.LEVEL3, "ssm", "3_Mamba1Block"),
    "T5Block": (OperatorLevel.LEVEL3, "encoder", "2_T5Block"),
    "SwinTransformerV2": (OperatorLevel.LEVEL3, "vision_transformer", "3_SwinTransformerV2"),
    
    # Level 2 fused ops
    "SwiGLU": (OperatorLevel.LEVEL2, "ffn", "1_SwiGLU"),
    "RMSNorm_QKVProj": (OperatorLevel.LEVEL2, "attention", "2_RMSNorm_QKVProj"),
    "QKV_Proj_RoPE": (OperatorLevel.LEVEL2, "attention", "1_QKV_Proj_RoPE"),
    
    # Level 1 primitives
    "RMSNorm": (OperatorLevel.LEVEL1, "normalization", "4_RMSNorm"),
    "LayerNorm": (OperatorLevel.LEVEL1, "normalization", "1_LayerNorm"),
    "GELU": (OperatorLevel.LEVEL1, "activations", "1_GELU"),
}


def get_operator(name: str, fallback_level: Optional[OperatorLevel] = None):
    """
    Get a pre-defined operator by name.
    
    Args:
        name: Operator name from OPERATOR_MAPPINGS
        fallback_level: If specified and the preferred level fails, try this level
        
    Returns:
        The Model class for the operator
    """
    if name not in OPERATOR_MAPPINGS:
        raise ValueError(f"Unknown operator: {name}. Available: {list(OPERATOR_MAPPINGS.keys())}")
    
    level, category, module_name = OPERATOR_MAPPINGS[name]
    
    try:
        return get_operator_module(level, category, module_name)
    except ImportError:
        if fallback_level is not None:
            # Try lower level fallback
            pass
        raise


# ============================================================================
# Benchmarking Interface Functions
# ============================================================================

def create_benchmark_config(
    model_class: type,
    variant: str = None,
    reduced: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create benchmark configuration for a model.
    
    Args:
        model_class: The model class to benchmark
        variant: Optional variant name
        reduced: Whether to use reduced config for faster benchmarking
        
    Returns:
        Tuple of (init_kwargs, input_kwargs)
    """
    if variant and hasattr(model_class, 'VARIANTS'):
        config = model_class.VARIANTS.get(variant, {}).copy()
    else:
        config = {}
    
    if reduced:
        # Apply reductions for benchmarking
        if 'num_layers' in config:
            config['num_layers'] = min(config['num_layers'], 8)
        if 'hidden_size' in config:
            config['hidden_size'] = min(config['hidden_size'], 4096)
    
    return config, {}
