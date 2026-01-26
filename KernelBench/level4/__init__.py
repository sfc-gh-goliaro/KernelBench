"""
Level 4: Complete Model Architectures

This package contains full model implementations for benchmarking,
organized by architecture family. Each Model class:
- Is a pure nn.Module subclass with no framework dependencies
- Uses level1 operators from KernelBench
- Accepts configuration via constructor parameters

HuggingFace Integration:
- Use adapters from hf_adapters.py for weight loading and validation
- Adapters are separate from Model classes for clean separation of concerns
- Example: LlamaAdapter, FalconAdapter, MistralAdapter, etc.

Architecture Categories:
- Dense Decoder: Llama-3.1, Falcon, Mistral
- MoE: DeepSeek-V2, Mixtral
- SSM: Mamba-2
- Linear Attention: RWKV-6, GLA, RetNet
- Encoder-Decoder: T5
- Vision Transformer: Swin-v2
- Vision-Language: Qwen2-VL
- Audio-Language: Whisper
- Diffusion: Stable Diffusion
- CNN: EfficientNet, ResNet
- TTS: CosyVoice
- Object Detection: YOLOv8, Deformable-DETR
- Recommendation: DCN-v2
- Embedding: BGE-M3
- Speculative Decoding: EAGLE-3, Lookahead, CLLM, LayerSkip
- Model Merging: TIES-Merging
- Neural Rendering: 3DGS, InstantNGP
"""

from .base import (
    BaseLevel4Model,
    ModelConfig,
    OperatorLevel,
    ValidationResult,
    create_benchmark_config,
    # Level1 operator imports
    import_level1_operator,
    get_level1_operator,
    LEVEL1_OPERATORS,
    # Legacy
    get_operator_module,
    get_operator,
    OPERATOR_MAPPINGS,
)

from .config_loader import (
    load_hf_config,
    get_available_configs,
    get_config_info,
    HARDCODED_CONFIGS,
)

from .validation import (
    compare_tensors,
    compare_model_outputs,
    compare_layer_by_layer,
    benchmark_model,
    validate_against_huggingface,
    run_validation_suite,
    PerformanceMetrics,
    # End-to-end validation with weight transfer
    validate_with_hf_weights,
    validate_level4_model,
    run_full_validation_suite,
)

# HuggingFace adapters (separated from model definitions)
from .hf_adapters import (
    # Infrastructure
    WeightMapper,
    load_hf_model,
    transfer_weights,
    # Base adapter class
    BaseHFAdapter,
    # Model-specific adapters
    LlamaAdapter,
    FalconAdapter,
    MistralAdapter,
    T5Adapter,
    WhisperAdapter,
    SwinV2Adapter,
    # Registry
    ADAPTER_REGISTRY,
    get_adapter,
    register_adapter,
    # Convenience functions
    create_model_from_pretrained,
    validate_model,
)

__all__ = [
    # Base classes
    "BaseLevel4Model",
    "ModelConfig",
    "OperatorLevel",
    "ValidationResult",
    "create_benchmark_config",
    # Level1 operator imports
    "import_level1_operator",
    "get_level1_operator",
    "LEVEL1_OPERATORS",
    # Legacy operator composition
    "get_operator_module",
    "get_operator",
    "OPERATOR_MAPPINGS",
    # Config loading
    "load_hf_config",
    "get_available_configs",
    "get_config_info",
    "HARDCODED_CONFIGS",
    # Validation
    "compare_tensors",
    "compare_model_outputs",
    "compare_layer_by_layer",
    "benchmark_model",
    "validate_against_huggingface",
    "run_validation_suite",
    "PerformanceMetrics",
    "validate_with_hf_weights",
    "validate_level4_model",
    "run_full_validation_suite",
    # HuggingFace adapters
    "WeightMapper",
    "load_hf_model",
    "transfer_weights",
    "BaseHFAdapter",
    "LlamaAdapter",
    "FalconAdapter",
    "MistralAdapter",
    "T5Adapter",
    "WhisperAdapter",
    "SwinV2Adapter",
    "ADAPTER_REGISTRY",
    "get_adapter",
    "register_adapter",
    "create_model_from_pretrained",
    "validate_model",
]
