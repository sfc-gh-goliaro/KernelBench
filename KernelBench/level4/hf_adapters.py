"""
HuggingFace Adapters for Level4 Models

This module provides adapters that handle:
- Config loading from HuggingFace
- Weight mapping between HuggingFace and our implementations
- Weight loading and transfer
- Model creation from pretrained weights
- Validation against HuggingFace implementations

Each adapter wraps a specific Model class and provides HF integration.
The Model classes themselves remain clean nn.Module definitions.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Type
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Weight Mapping Infrastructure
# ============================================================================

class WeightMapper:
    """
    Utility class for mapping weights between HuggingFace models and our implementations.
    """
    
    def __init__(self, hf_to_local: Optional[Dict[str, str]] = None):
        """
        Initialize weight mapper.
        
        Args:
            hf_to_local: Dictionary mapping HuggingFace parameter names to local names.
                         Supports wildcards with {layer_idx} placeholder.
        """
        self.hf_to_local = hf_to_local or {}
        self._compiled_patterns = None
    
    def _compile_patterns(self, num_layers: int) -> Dict[str, str]:
        """Compile patterns by expanding layer indices."""
        if self._compiled_patterns is not None:
            return self._compiled_patterns
        
        compiled = {}
        for hf_pattern, local_pattern in self.hf_to_local.items():
            if "{layer_idx}" in hf_pattern:
                for i in range(num_layers):
                    hf_name = hf_pattern.format(layer_idx=i)
                    local_name = local_pattern.format(layer_idx=i)
                    compiled[hf_name] = local_name
            else:
                compiled[hf_pattern] = local_pattern
        
        self._compiled_patterns = compiled
        return compiled
    
    def map_state_dict(
        self,
        hf_state_dict: Dict[str, torch.Tensor],
        num_layers: int,
        strict: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
        """
        Map HuggingFace state dict to local model format.
        
        Args:
            hf_state_dict: State dict from HuggingFace model
            num_layers: Number of layers in the model
            strict: If True, raise error on unmapped weights
            
        Returns:
            Tuple of (mapped_state_dict, unmapped_hf_keys, missing_local_keys)
        """
        patterns = self._compile_patterns(num_layers)
        
        mapped = {}
        unmapped_hf = []
        
        for hf_name, tensor in hf_state_dict.items():
            if hf_name in patterns:
                local_name = patterns[hf_name]
                mapped[local_name] = tensor
            else:
                unmapped_hf.append(hf_name)
        
        if strict and unmapped_hf:
            raise ValueError(f"Unmapped HuggingFace weights: {unmapped_hf[:10]}...")
        
        return mapped, unmapped_hf, []


def load_hf_model(
    hf_model_name: str,
    model_type: str = "causal_lm",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
    """
    Load a HuggingFace model.
    
    Args:
        hf_model_name: HuggingFace model name/path
        model_type: Type of model ("causal_lm", "seq2seq", "encoder", "vision", "speech")
        device: Device to load model on
        dtype: Data type for model weights
        
    Returns:
        Tuple of (hf_model, state_dict)
    """
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoModel,
            AutoModelForImageClassification,
            AutoModelForSpeechSeq2Seq,
        )
        
        model_class_map = {
            "causal_lm": AutoModelForCausalLM,
            "seq2seq": AutoModelForSeq2SeqLM,
            "encoder": AutoModel,
            "vision": AutoModelForImageClassification,
            "speech": AutoModelForSpeechSeq2Seq,
        }
        
        model_class = model_class_map.get(model_type, AutoModel)
        
        hf_model = model_class.from_pretrained(
            hf_model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
        )
        
        if device == "cpu":
            hf_model = hf_model.to(device)
        
        hf_model.eval()
        state_dict = hf_model.state_dict()
        
        return hf_model, state_dict
        
    except Exception as e:
        raise RuntimeError(f"Failed to load HuggingFace model '{hf_model_name}': {e}")


def transfer_weights(
    source_state_dict: Dict[str, torch.Tensor],
    target_model: nn.Module,
    weight_mapper: WeightMapper,
    num_layers: int,
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Transfer weights from source state dict to target model using weight mapper.
    
    Args:
        source_state_dict: Source state dict (e.g., from HuggingFace)
        target_model: Target model to load weights into
        weight_mapper: WeightMapper instance with mapping rules
        num_layers: Number of layers in the model
        strict: If True, raise error on missing/unexpected weights
        
    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    mapped_state_dict, unmapped_source, _ = weight_mapper.map_state_dict(
        source_state_dict, num_layers, strict=False
    )
    
    # Get target model's state dict keys
    target_keys = set(target_model.state_dict().keys())
    mapped_keys = set(mapped_state_dict.keys())
    
    missing_keys = list(target_keys - mapped_keys)
    unexpected_keys = list(mapped_keys - target_keys)
    
    # Load the mapped weights
    result = target_model.load_state_dict(mapped_state_dict, strict=False)
    
    if strict and (missing_keys or unexpected_keys):
        raise ValueError(
            f"Weight transfer failed. Missing: {missing_keys[:5]}..., "
            f"Unexpected: {unexpected_keys[:5]}..."
        )
    
    return missing_keys + list(result.missing_keys), unexpected_keys + list(result.unexpected_keys)


# ============================================================================
# Base HuggingFace Adapter
# ============================================================================

class BaseHFAdapter(ABC):
    """
    Base adapter class for integrating level4 models with HuggingFace.
    
    Each adapter wraps a specific Model class and provides:
    - Config loading from HuggingFace
    - Weight mapping definitions
    - Weight loading/transfer
    - Model creation from pretrained
    - Validation utilities
    """
    
    # Override in subclasses
    MODEL_CLASS: Type[nn.Module] = None
    HF_MODEL_TYPE: str = "causal_lm"
    VARIANTS: Dict[str, str] = {}  # variant_name -> hf_model_id
    
    def __init__(self, model: nn.Module, num_layers: int = 32):
        """
        Initialize adapter with a model instance.
        
        Args:
            model: The level4 model instance
            num_layers: Number of layers in the model
        """
        self.model = model
        self.num_layers = num_layers
    
    @classmethod
    def get_hf_model_name(cls, variant: str) -> str:
        """Get HuggingFace model name for a variant."""
        if variant not in cls.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Available: {list(cls.VARIANTS.keys())}")
        return cls.VARIANTS[variant]
    
    @classmethod
    def load_config(cls, variant: str) -> Dict[str, Any]:
        """Load config from HuggingFace for a variant."""
        from .config_loader import load_hf_config
        hf_model_name = cls.get_hf_model_name(variant)
        return load_hf_config(hf_model_name)
    
    @abstractmethod
    def get_weight_mapping(self) -> Dict[str, str]:
        """
        Return the weight mapping dictionary.
        
        Maps HuggingFace parameter names to local model parameter names.
        Use {layer_idx} placeholder for per-layer parameters.
        
        Returns:
            Dict mapping HF names to local names
        """
        pass
    
    def get_weight_mapper(self) -> WeightMapper:
        """Get a WeightMapper instance with this adapter's mappings."""
        return WeightMapper(self.get_weight_mapping())
    
    def load_weights(
        self,
        hf_model_name: Optional[str] = None,
        variant: Optional[str] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        strict: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """
        Load weights from HuggingFace into the wrapped model.
        
        Args:
            hf_model_name: HuggingFace model name. If None, use variant.
            variant: Variant name to get HF model name from VARIANTS.
            device: Device for loading
            dtype: Data type for weights
            strict: Raise error on missing/unexpected weights
            
        Returns:
            Tuple of (missing_keys, unexpected_keys)
        """
        if hf_model_name is None:
            if variant is None:
                raise ValueError("Must provide either hf_model_name or variant")
            hf_model_name = self.get_hf_model_name(variant)
        
        logger.info(f"Loading weights from: {hf_model_name}")
        
        # Load HuggingFace state dict
        _, hf_state_dict = load_hf_model(
            hf_model_name,
            model_type=self.HF_MODEL_TYPE,
            device=device,
            dtype=dtype,
        )
        
        # Transfer weights
        weight_mapper = self.get_weight_mapper()
        missing, unexpected = transfer_weights(
            hf_state_dict,
            self.model,
            weight_mapper,
            self.num_layers,
            strict=strict,
        )
        
        logger.info(f"Weight transfer complete. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        return missing, unexpected
    
    @classmethod
    def create_model(cls, variant: str, **kwargs) -> nn.Module:
        """
        Create a model instance with config from HuggingFace.
        
        Args:
            variant: Variant name
            **kwargs: Override config values
            
        Returns:
            Model instance (without weights loaded)
        """
        config = cls.load_config(variant)
        config.update(kwargs)
        return cls.MODEL_CLASS(**config)
    
    @classmethod
    def from_pretrained(
        cls,
        variant: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        strict: bool = False,
        **kwargs
    ) -> Tuple[nn.Module, "BaseHFAdapter"]:
        """
        Create model and load pretrained weights.
        
        This is the main entry point for creating a model with HF weights.
        
        Args:
            variant: Variant name
            device: Device to load model on
            dtype: Data type for model
            strict: Raise error on missing/unexpected weights
            **kwargs: Additional config overrides
            
        Returns:
            Tuple of (model, adapter)
        """
        # Create model
        config = cls.load_config(variant)
        config.update(kwargs)
        model = cls.MODEL_CLASS(**config)
        
        # Get num_layers from config
        num_layers = config.get('num_layers', 32)
        
        # Create adapter and load weights
        adapter = cls(model, num_layers=num_layers)
        missing, unexpected = adapter.load_weights(
            variant=variant,
            device=device,
            dtype=dtype,
            strict=strict,
        )
        
        # Move model to device
        model = model.to(device=device, dtype=dtype)
        
        logger.info(f"Created model from pretrained. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        return model, adapter
    
    def validate(
        self,
        variant: str,
        batch_size: int = 1,
        seq_len: int = 64,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ) -> "ValidationResult":
        """
        Validate the model against HuggingFace implementation.
        
        Args:
            variant: Variant name
            batch_size: Batch size for test
            seq_len: Sequence length for test
            device: Device to run on
            dtype: Data type
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            ValidationResult
        """
        from .validation import ValidationResult, compare_tensors
        
        hf_model_name = self.get_hf_model_name(variant)
        
        try:
            # Load HuggingFace model
            hf_model, hf_state_dict = load_hf_model(
                hf_model_name,
                model_type=self.HF_MODEL_TYPE,
                device=device,
                dtype=dtype,
            )
            
            # Transfer weights to our model
            weight_mapper = self.get_weight_mapper()
            transfer_weights(hf_state_dict, self.model, weight_mapper, self.num_layers)
            
            # Move our model to device
            self.model.to(device=device, dtype=dtype)
            self.model.eval()
            
            # Create test inputs
            vocab_size = getattr(self.model, 'vocab_size', 32000)
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'vocab_size'):
                vocab_size = self.model.config.vocab_size
            
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            
            # Run forward pass
            with torch.no_grad():
                our_output = self.model(input_ids)
                hf_output = hf_model(input_ids)
                hf_logits = hf_output.logits
            
            # Compare
            result = compare_tensors(our_output, hf_logits, rtol=rtol, atol=atol)
            return result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            from .validation import ValidationResult
            return ValidationResult(
                is_valid=False,
                max_diff=float('inf'),
                mean_diff=float('inf'),
                rtol=rtol,
                atol=atol,
                num_elements=0,
                details={"error": str(e)}
            )


# ============================================================================
# Validation Result (simplified, imports from validation.py at runtime)
# ============================================================================

class ValidationResult:
    """Container for validation results."""
    
    def __init__(
        self,
        is_valid: bool,
        max_diff: float,
        mean_diff: float,
        rtol: float,
        atol: float,
        num_elements: int = 0,
        details: Optional[Dict[str, Any]] = None
    ):
        self.is_valid = is_valid
        self.max_diff = max_diff
        self.mean_diff = mean_diff
        self.rtol = rtol
        self.atol = atol
        self.num_elements = num_elements
        self.details = details or {}
    
    def __repr__(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        return f"ValidationResult({status}, max_diff={self.max_diff:.2e}, mean_diff={self.mean_diff:.2e})"


# ============================================================================
# Model-Specific Adapters
# ============================================================================

class LlamaAdapter(BaseHFAdapter):
    """HuggingFace adapter for Llama-3.1 models."""
    
    HF_MODEL_TYPE = "causal_lm"
    VARIANTS = {
        "8B": "meta-llama/Llama-3.1-8B",
        "70B": "meta-llama/Llama-3.1-70B",
    }
    
    def get_weight_mapping(self) -> Dict[str, str]:
        return {
            # Embeddings
            "model.embed_tokens.weight": "embed_tokens.weight",
            
            # Final norm and LM head
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
            
            # Per-layer mappings
            "model.layers.{layer_idx}.input_layernorm.weight": "layers.{layer_idx}.input_layernorm.weight",
            "model.layers.{layer_idx}.post_attention_layernorm.weight": "layers.{layer_idx}.post_attention_layernorm.weight",
            
            # Attention projections
            "model.layers.{layer_idx}.self_attn.q_proj.weight": "layers.{layer_idx}.self_attn.q_proj.weight",
            "model.layers.{layer_idx}.self_attn.k_proj.weight": "layers.{layer_idx}.self_attn.k_proj.weight",
            "model.layers.{layer_idx}.self_attn.v_proj.weight": "layers.{layer_idx}.self_attn.v_proj.weight",
            "model.layers.{layer_idx}.self_attn.o_proj.weight": "layers.{layer_idx}.self_attn.o_proj.weight",
            
            # MLP projections
            "model.layers.{layer_idx}.mlp.gate_proj.weight": "layers.{layer_idx}.mlp.gate_proj.weight",
            "model.layers.{layer_idx}.mlp.up_proj.weight": "layers.{layer_idx}.mlp.up_proj.weight",
            "model.layers.{layer_idx}.mlp.down_proj.weight": "layers.{layer_idx}.mlp.down_proj.weight",
        }


class FalconAdapter(BaseHFAdapter):
    """HuggingFace adapter for Falcon models."""
    
    HF_MODEL_TYPE = "causal_lm"
    VARIANTS = {
        "7B": "tiiuae/falcon-7b",
        "40B": "tiiuae/falcon-40b",
    }
    
    def get_weight_mapping(self) -> Dict[str, str]:
        return {
            # Embeddings
            "transformer.word_embeddings.weight": "word_embeddings.weight",
            
            # Final norm and LM head
            "transformer.ln_f.weight": "ln_f.weight",
            "transformer.ln_f.bias": "ln_f.bias",
            "lm_head.weight": "lm_head.weight",
            
            # Per-layer mappings
            "transformer.h.{layer_idx}.input_layernorm.weight": "h.{layer_idx}.input_layernorm.weight",
            "transformer.h.{layer_idx}.input_layernorm.bias": "h.{layer_idx}.input_layernorm.bias",
            
            # Attention
            "transformer.h.{layer_idx}.self_attention.query_key_value.weight": "h.{layer_idx}.self_attention.query_key_value.weight",
            "transformer.h.{layer_idx}.self_attention.dense.weight": "h.{layer_idx}.self_attention.dense.weight",
            
            # MLP
            "transformer.h.{layer_idx}.mlp.dense_h_to_4h.weight": "h.{layer_idx}.mlp.dense_h_to_4h.weight",
            "transformer.h.{layer_idx}.mlp.dense_4h_to_h.weight": "h.{layer_idx}.mlp.dense_4h_to_h.weight",
        }


class MistralAdapter(BaseHFAdapter):
    """HuggingFace adapter for Mistral models."""
    
    HF_MODEL_TYPE = "causal_lm"
    VARIANTS = {
        "7B": "mistralai/Mistral-7B-v0.3",
        "Nemo-12B": "mistralai/Mistral-Nemo-12B",
    }
    
    def get_weight_mapping(self) -> Dict[str, str]:
        # Mistral uses same naming as Llama
        return {
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
            
            "model.layers.{layer_idx}.input_layernorm.weight": "layers.{layer_idx}.input_layernorm.weight",
            "model.layers.{layer_idx}.post_attention_layernorm.weight": "layers.{layer_idx}.post_attention_layernorm.weight",
            
            "model.layers.{layer_idx}.self_attn.q_proj.weight": "layers.{layer_idx}.self_attn.q_proj.weight",
            "model.layers.{layer_idx}.self_attn.k_proj.weight": "layers.{layer_idx}.self_attn.k_proj.weight",
            "model.layers.{layer_idx}.self_attn.v_proj.weight": "layers.{layer_idx}.self_attn.v_proj.weight",
            "model.layers.{layer_idx}.self_attn.o_proj.weight": "layers.{layer_idx}.self_attn.o_proj.weight",
            
            "model.layers.{layer_idx}.mlp.gate_proj.weight": "layers.{layer_idx}.mlp.gate_proj.weight",
            "model.layers.{layer_idx}.mlp.up_proj.weight": "layers.{layer_idx}.mlp.up_proj.weight",
            "model.layers.{layer_idx}.mlp.down_proj.weight": "layers.{layer_idx}.mlp.down_proj.weight",
        }


class T5Adapter(BaseHFAdapter):
    """HuggingFace adapter for T5 models."""
    
    HF_MODEL_TYPE = "seq2seq"
    VARIANTS = {
        "Base": "google-t5/t5-base",
        "Large": "google-t5/t5-large",
        "3B": "google-t5/t5-3b",
    }
    
    def get_weight_mapping(self) -> Dict[str, str]:
        # T5 has a more complex structure with encoder/decoder
        return {
            "shared.weight": "shared.weight",
            "lm_head.weight": "lm_head.weight",
            
            # Encoder layers
            "encoder.block.{layer_idx}.layer.0.SelfAttention.q.weight": "encoder.layers.{layer_idx}.self_attn.q_proj.weight",
            "encoder.block.{layer_idx}.layer.0.SelfAttention.k.weight": "encoder.layers.{layer_idx}.self_attn.k_proj.weight",
            "encoder.block.{layer_idx}.layer.0.SelfAttention.v.weight": "encoder.layers.{layer_idx}.self_attn.v_proj.weight",
            "encoder.block.{layer_idx}.layer.0.SelfAttention.o.weight": "encoder.layers.{layer_idx}.self_attn.o_proj.weight",
            "encoder.block.{layer_idx}.layer.0.layer_norm.weight": "encoder.layers.{layer_idx}.self_attn_norm.weight",
            "encoder.block.{layer_idx}.layer.1.DenseReluDense.wi.weight": "encoder.layers.{layer_idx}.mlp.wi.weight",
            "encoder.block.{layer_idx}.layer.1.DenseReluDense.wo.weight": "encoder.layers.{layer_idx}.mlp.wo.weight",
            "encoder.block.{layer_idx}.layer.1.layer_norm.weight": "encoder.layers.{layer_idx}.mlp_norm.weight",
            
            "encoder.final_layer_norm.weight": "encoder.final_norm.weight",
        }


class WhisperAdapter(BaseHFAdapter):
    """HuggingFace adapter for Whisper models."""
    
    HF_MODEL_TYPE = "speech"
    VARIANTS = {
        "Base": "openai/whisper-base",
        "Large-v3": "openai/whisper-large-v3",
    }
    
    def get_weight_mapping(self) -> Dict[str, str]:
        return {
            # Encoder
            "model.encoder.conv1.weight": "encoder.conv1.weight",
            "model.encoder.conv1.bias": "encoder.conv1.bias",
            "model.encoder.conv2.weight": "encoder.conv2.weight",
            "model.encoder.conv2.bias": "encoder.conv2.bias",
            "model.encoder.embed_positions.weight": "encoder.embed_positions.weight",
            
            # Decoder
            "model.decoder.embed_tokens.weight": "decoder.embed_tokens.weight",
            "model.decoder.embed_positions.weight": "decoder.embed_positions.weight",
            
            "proj_out.weight": "proj_out.weight",
        }


class SwinV2Adapter(BaseHFAdapter):
    """HuggingFace adapter for Swin Transformer V2 models."""
    
    HF_MODEL_TYPE = "vision"
    VARIANTS = {
        "T": "microsoft/swinv2-tiny-patch4-window8-256",
        "S": "microsoft/swinv2-small-patch4-window8-256",
        "B": "microsoft/swinv2-base-patch4-window12-192-22k",
        "L": "microsoft/swinv2-large-patch4-window12-192-22k",
    }
    
    def get_weight_mapping(self) -> Dict[str, str]:
        return {
            "swinv2.embeddings.patch_embeddings.projection.weight": "patch_embed.proj.weight",
            "swinv2.embeddings.patch_embeddings.projection.bias": "patch_embed.proj.bias",
            "swinv2.embeddings.norm.weight": "patch_embed.norm.weight",
            "swinv2.embeddings.norm.bias": "patch_embed.norm.bias",
            "classifier.weight": "head.weight",
            "classifier.bias": "head.bias",
        }


# ============================================================================
# Adapter Registry
# ============================================================================

ADAPTER_REGISTRY: Dict[str, Type[BaseHFAdapter]] = {
    "llama": LlamaAdapter,
    "llama31": LlamaAdapter,
    "falcon": FalconAdapter,
    "mistral": MistralAdapter,
    "t5": T5Adapter,
    "whisper": WhisperAdapter,
    "swinv2": SwinV2Adapter,
}


def get_adapter(model_name: str) -> Type[BaseHFAdapter]:
    """
    Get adapter class by model name.
    
    Args:
        model_name: Model name (e.g., "llama", "falcon", "mistral")
        
    Returns:
        Adapter class
    """
    name_lower = model_name.lower().replace("-", "").replace("_", "")
    
    if name_lower not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(ADAPTER_REGISTRY.keys())}")
    
    return ADAPTER_REGISTRY[name_lower]


def register_adapter(name: str, adapter_class: Type[BaseHFAdapter]):
    """Register a new adapter class."""
    ADAPTER_REGISTRY[name.lower()] = adapter_class


# ============================================================================
# Convenience Functions
# ============================================================================

def create_model_from_pretrained(
    model_class: Type[nn.Module],
    adapter_class: Type[BaseHFAdapter],
    variant: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    **kwargs
) -> Tuple[nn.Module, BaseHFAdapter]:
    """
    Create a model with pretrained weights using the specified adapter.
    
    Args:
        model_class: The Model class (nn.Module)
        adapter_class: The adapter class for weight mapping
        variant: Variant name
        device: Device to load on
        dtype: Data type
        **kwargs: Config overrides
        
    Returns:
        Tuple of (model, adapter)
    """
    # Set MODEL_CLASS on adapter
    adapter_class.MODEL_CLASS = model_class
    
    return adapter_class.from_pretrained(
        variant=variant,
        device=device,
        dtype=dtype,
        **kwargs
    )


def validate_model(
    model: nn.Module,
    adapter_class: Type[BaseHFAdapter],
    variant: str,
    num_layers: int = 32,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> ValidationResult:
    """
    Validate a model against HuggingFace.
    
    Args:
        model: Model instance
        adapter_class: Adapter class for weight mapping
        variant: Variant name
        num_layers: Number of layers
        device: Device
        dtype: Data type
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        ValidationResult
    """
    adapter = adapter_class(model, num_layers=num_layers)
    return adapter.validate(
        variant=variant,
        device=device,
        dtype=dtype,
        rtol=rtol,
        atol=atol,
    )
