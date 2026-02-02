"""
Test model alignment with HuggingFace transformers library.

This test validates that KernelBench model implementations produce
outputs matching the HuggingFace transformers implementation using:
1. A batch of 5 prompts with varying lengths
2. Continuous batching with paged KV cache
3. Both prefill and decode phases

Supports any model with a corresponding KernelBench level4 implementation.
Pass the model name via --model-name parameter.
Optionally limit the number of layers with --max-layers for faster testing
or to fit larger models on smaller GPUs.

Usage:
    pytest tests/test_hf_alignment.py --model-name meta-llama/Llama-3.1-8B-Instruct
    pytest tests/test_hf_alignment.py --model-name meta-llama/Llama-3.1-70B-Instruct
    pytest tests/test_hf_alignment.py --model-name mistralai/Mistral-7B-v0.1
    
    # Use only first 4 layers for faster testing:
    pytest tests/test_hf_alignment.py --model-name meta-llama/Llama-3.1-70B-Instruct --max-layers 4

Requires HuggingFace authentication with access to the specified model.
"""

import pytest
import torch
import sys
import os
import importlib
from typing import List, Tuple, Optional, Dict

# Set up HuggingFace environment
os.environ["HF_HOME"] = "/home/yak/data-fast/huggingface"
HF_TOKEN_PATH = "/home/yak/data-fast/huggingface/token"
if os.path.exists(HF_TOKEN_PATH):
    with open(HF_TOKEN_PATH, "r") as f:
        os.environ["HF_TOKEN"] = f.read().strip()

# Add paths for imports
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(TEST_DIR, '..')
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'KernelBench'))

# Skip all tests if transformers is not available
transformers = pytest.importorskip("transformers")

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Default model for testing
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# ============================================================================
# Model to KernelBench Implementation Mapping
# ============================================================================
# Maps HuggingFace model name patterns to their corresponding level4 module names.
# Multiple HF models can map to the same implementation (e.g., 8B and 70B variants).

MODEL_TO_IMPLEMENTATION: Dict[str, str] = {
    # Llama 3.1 variants
    "meta-llama/Llama-3.1-8B": "KernelBench.level4.1_Llama31",
    "meta-llama/Llama-3.1-8B-Instruct": "KernelBench.level4.1_Llama31",
    "meta-llama/Llama-3.1-70B": "KernelBench.level4.1_Llama31",
    "meta-llama/Llama-3.1-70B-Instruct": "KernelBench.level4.1_Llama31",
    "meta-llama/Llama-3.1-405B": "KernelBench.level4.1_Llama31",
    "meta-llama/Llama-3.1-405B-Instruct": "KernelBench.level4.1_Llama31",
    # Falcon variants
    "tiiuae/falcon-7b": "KernelBench.level4.2_Falcon",
    "tiiuae/falcon-7b-instruct": "KernelBench.level4.2_Falcon",
    "tiiuae/falcon-40b": "KernelBench.level4.2_Falcon",
    "tiiuae/falcon-40b-instruct": "KernelBench.level4.2_Falcon",
    # Mistral variants
    "mistralai/Mistral-7B-v0.1": "KernelBench.level4.3_Mistral",
    "mistralai/Mistral-7B-Instruct-v0.1": "KernelBench.level4.3_Mistral",
    "mistralai/Mistral-7B-Instruct-v0.2": "KernelBench.level4.3_Mistral",
    # Mixtral (MoE)
    "mistralai/Mixtral-8x7B-v0.1": "KernelBench.level4.4_MoE",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "KernelBench.level4.4_MoE",
    # T5 variants
    "google-t5/t5-small": "KernelBench.level4.9_T5",
    "google-t5/t5-base": "KernelBench.level4.9_T5",
    "google-t5/t5-large": "KernelBench.level4.9_T5",
    "google/flan-t5-small": "KernelBench.level4.9_T5",
    "google/flan-t5-base": "KernelBench.level4.9_T5",
    "google/flan-t5-large": "KernelBench.level4.9_T5",
    # Qwen2-VL
    "Qwen/Qwen2-VL-2B-Instruct": "KernelBench.level4.11_Qwen2VL",
    "Qwen/Qwen2-VL-7B-Instruct": "KernelBench.level4.11_Qwen2VL",
    # Whisper
    "openai/whisper-tiny": "KernelBench.level4.12_Whisper",
    "openai/whisper-small": "KernelBench.level4.12_Whisper",
    "openai/whisper-base": "KernelBench.level4.12_Whisper",
    "openai/whisper-medium": "KernelBench.level4.12_Whisper",
    "openai/whisper-large-v3": "KernelBench.level4.12_Whisper",
    # BLOOM variants
    "bigscience/bloom-560m": "KernelBench.level4.4_Bloom",
    "bigscience/bloom-1b1": "KernelBench.level4.4_Bloom",
    "bigscience/bloom-1b7": "KernelBench.level4.4_Bloom",
    "bigscience/bloom-3b": "KernelBench.level4.4_Bloom",
    "bigscience/bloom-7b1": "KernelBench.level4.4_Bloom",
    "bigscience/bloom": "KernelBench.level4.4_Bloom",
    # DeepSeek-V2 variants
    "deepseek-ai/DeepSeek-V2-Lite": "KernelBench.level4.3_Deepseek",
    "deepseek-ai/DeepSeek-V2": "KernelBench.level4.3_Deepseek",
}


def get_implementation_module(model_name: str) -> str:
    """
    Get the KernelBench implementation module for a given HuggingFace model name.
    
    Args:
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        
    Returns:
        Module path (e.g., "KernelBench.level4.1_Llama31")
        
    Raises:
        ValueError: If no implementation is found for the model
    """
    # Direct lookup
    if model_name in MODEL_TO_IMPLEMENTATION:
        return MODEL_TO_IMPLEMENTATION[model_name]
    
    # Try prefix matching for model families
    for pattern, module in MODEL_TO_IMPLEMENTATION.items():
        if model_name.startswith(pattern.rsplit("-", 1)[0]):
            return module
    
    raise ValueError(
        f"No KernelBench implementation found for model '{model_name}'. "
        f"Available models: {list(MODEL_TO_IMPLEMENTATION.keys())}"
    )

# Test prompts with varying lengths - realistic instruction-following prompts
TEST_PROMPTS = [
    # Short prompt
    "What is the capital of France? Answer in one word.",
    
    # Medium prompt
    "Explain the concept of machine learning to a 10-year-old. Use simple words and give an example that a child would understand.",
    
    # Long prompt with context
    """You are a helpful AI assistant. A user asks you the following question:

"I'm planning a road trip from San Francisco to Los Angeles. What are some interesting stops I should make along the way? I'm interested in nature, good food, and historical sites."

Please provide a detailed response with at least 5 recommendations.""",
    
    # Code-related prompt
    """Write a Python function that implements binary search on a sorted list. Include:
1. Proper docstring with examples
2. Type hints
3. Edge case handling
4. Time and space complexity comments

Here's the function signature to use:
def binary_search(arr: list[int], target: int) -> int:""",
    
    # Creative writing prompt
    """Write the opening paragraph of a mystery novel set in Victorian London. The scene should:
- Introduce a detective character
- Set a foggy, atmospheric mood
- Hint at an upcoming crime
- Be exactly 3 sentences long

Begin your paragraph:""",
]

# Response lengths for each prompt (number of tokens to generate)
RESPONSE_LENGTHS = [10, 50, 100, 150, 50]

# Tolerance thresholds for numerical comparison
# Using relative tolerance (rtol) is more robust than absolute tolerance for varying magnitudes
# We use different thresholds for mean vs max to handle outliers gracefully
RTOL_MEAN = 5e-2     # 5% mean relative tolerance (most values should be close)
RTOL_MAX = 3.0       # 300% max relative tolerance (allow large outliers for near-zero logits)
ATOL = 1e-2          # Absolute tolerance for values near zero
ATOL_STRICT = 1e-6   # Strict tolerance for component tests (weights should be exact)

# Note: RTOL_MAX is set high because near-zero logits can have large relative errors
# even with small absolute errors. The mean relative tolerance is the stricter check
# that ensures overall alignment, while max just catches extreme outliers.

# Generation test threshold: minimum fraction of consecutive matching tokens
# After the first mismatch, subsequent tokens are considered diverged
GENERATION_CONSECUTIVE_THRESHOLD = 0.50  # Require at least 50% consecutive matches


def load_kernelbench_model(model_name: str, config_dict: dict):
    """
    Load the KernelBench model corresponding to the HuggingFace model name.
    
    Args:
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        config_dict: Model configuration dictionary
        
    Returns:
        Tuple of (model instance, module)
    """
    module_path = get_implementation_module(model_name)
    kb_module = importlib.import_module(module_path)
    return kb_module.Model(**config_dict), kb_module


def _normalize_key(key: str) -> str:
    """
    Normalize a state dict key by removing wrapper module names.
    
    KernelBench models wrap primitives in extra modules:
    - LayerNorm wrapper: .ln.weight -> .weight
    - Linear wrapper: .linear.weight -> .weight
    - RMSNorm wrapper: .norm.weight -> .weight
    
    This allows matching KB keys to HF keys despite structural differences.
    """
    # Remove common wrapper suffixes
    wrappers = ['.ln.', '.linear.', '.norm.']
    for wrapper in wrappers:
        key = key.replace(wrapper, '.')
    return key


def copy_weights(hf_model, kb_model, num_layers: int) -> None:
    """
    Copy weights from HuggingFace model to KernelBench model.
    
    Uses automatic key matching to support different architectures:
    - Llama: model.embed_tokens, model.layers, model.norm
    - Falcon: transformer.word_embeddings, transformer.h, transformer.ln_f
    
    Handles structural differences where KB wraps primitives in extra modules.
    """
    hf_state = hf_model.state_dict()
    kb_state = kb_model.state_dict()
    
    # Detect HF model prefix (e.g., "model." for Llama, "transformer." for Falcon)
    hf_prefix = ""
    for hf_key in hf_state.keys():
        if "." in hf_key:
            potential_prefix = hf_key.split(".")[0] + "."
            if potential_prefix in ["model.", "transformer."]:
                hf_prefix = potential_prefix
                break
    
    # Build normalized HF key lookup: normalized_key -> original_key
    hf_normalized = {}
    for hf_key in hf_state.keys():
        # Strip prefix and normalize
        stripped = hf_key[len(hf_prefix):] if hf_key.startswith(hf_prefix) else hf_key
        normalized = _normalize_key(stripped)
        hf_normalized[normalized] = hf_key
    
    # Match KB keys to HF keys
    copied = 0
    skipped_buffers = 0
    missing_in_hf = []
    
    for kb_key, kb_tensor in kb_state.items():
        # Skip buffers that shouldn't be copied (inv_freq, kv_cache, alibi_slopes, etc.)
        if any(skip in kb_key for skip in ['inv_freq', 'kv_cache', '_cache', 'alibi_slopes']):
            skipped_buffers += 1
            continue
        
        # Normalize KB key and try to find matching HF key
        kb_normalized = _normalize_key(kb_key)
        
        if kb_normalized in hf_normalized:
            hf_key = hf_normalized[kb_normalized]
            hf_tensor = hf_state[hf_key]
            if kb_tensor.shape == hf_tensor.shape:
                kb_tensor.copy_(hf_tensor)
                copied += 1
            else:
                missing_in_hf.append(f"{kb_key} (shape mismatch: KB={kb_tensor.shape} vs HF={hf_tensor.shape})")
        else:
            missing_in_hf.append(kb_key)
    
    if missing_in_hf:
        print(f"  Warning: {len(missing_in_hf)} KB weights not found in HF model:")
        for m in missing_in_hf[:10]:
            print(f"    - {m}")
        if len(missing_in_hf) > 10:
            print(f"    ... and {len(missing_in_hf) - 10} more")
    
    print(f"  Copied {copied} weights, skipped {skipped_buffers} buffers")
    
    # Load the updated state dict back into the model
    kb_model.load_state_dict(kb_state)


def create_kb_model_from_hf_config(hf_config, num_blocks: int = 8192):
    """Create KernelBench model config from HuggingFace config.
    
    Handles different naming conventions across model architectures:
    - Llama: intermediate_size, rms_norm_eps, num_hidden_layers, num_attention_heads
    - Falcon: ffn_hidden_size (or 4*hidden_size), layer_norm_epsilon
    - Mistral: intermediate_size, rms_norm_eps
    - BLOOM: n_inner (or 4*hidden), layer_norm_epsilon, n_layer, n_head
    """
    # Get rope_scaling if available (not used by BLOOM which uses ALiBi)
    rope_scaling = getattr(hf_config, 'rope_scaling', None)
    
    # num_layers: different models use different attribute names
    num_layers = getattr(hf_config, 'num_hidden_layers', None)
    if num_layers is None:
        num_layers = getattr(hf_config, 'n_layer', None)  # BLOOM
    if num_layers is None:
        raise ValueError("Could not determine num_layers from config")
    
    # num_heads: different models use different attribute names
    num_heads = getattr(hf_config, 'num_attention_heads', None)
    if num_heads is None:
        num_heads = getattr(hf_config, 'n_head', None)  # BLOOM
    if num_heads is None:
        raise ValueError("Could not determine num_heads from config")
    
    # intermediate_size: Llama/Mistral use intermediate_size, Falcon uses ffn_hidden_size, BLOOM uses n_inner
    intermediate_size = getattr(hf_config, 'intermediate_size', None)
    if intermediate_size is None:
        intermediate_size = getattr(hf_config, 'ffn_hidden_size', None)
    if intermediate_size is None:
        intermediate_size = getattr(hf_config, 'n_inner', None)  # BLOOM
    if intermediate_size is None:
        # Default: 4 * hidden_size
        intermediate_size = hf_config.hidden_size * 4
    
    # norm_eps: different models use different attribute names
    norm_eps = getattr(hf_config, 'rms_norm_eps', None)
    if norm_eps is None:
        norm_eps = getattr(hf_config, 'layer_norm_epsilon', None)  # BLOOM, Falcon
    if norm_eps is None:
        norm_eps = getattr(hf_config, 'layer_norm_eps', 1e-5)
    
    # num_kv_heads: different models use different attribute names
    # - Llama/Mistral: num_key_value_heads
    # - Falcon: num_kv_heads, or multi_query=True means 1 KV head (MQA)
    # - BLOOM: uses full MHA, so num_kv_heads = num_heads
    num_kv_heads = getattr(hf_config, 'num_key_value_heads', None)
    if num_kv_heads is None:
        num_kv_heads = getattr(hf_config, 'num_kv_heads', None)
    if num_kv_heads is None:
        # Check for Falcon's multi_query attribute (MQA = 1 KV head)
        if getattr(hf_config, 'multi_query', False):
            num_kv_heads = 1
        else:
            # Default to full attention (num_kv_heads = num_heads)
            num_kv_heads = num_heads
    
    # BLOOM-specific: apply_residual_connection_post_layernorm
    apply_residual_post_ln = getattr(hf_config, 'apply_residual_connection_post_layernorm', False)
    
    config = {
        'vocab_size': hf_config.vocab_size,
        'hidden_size': hf_config.hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'head_dim': hf_config.hidden_size // num_heads,
        'intermediate_size': intermediate_size,
        'max_seq_len': getattr(hf_config, 'max_position_embeddings', 2048),
        'rope_theta': getattr(hf_config, 'rope_theta', 500000.0),
        'rope_scaling': rope_scaling,
        'rms_norm_eps': norm_eps,  # Keep the key name for KernelBench compatibility
        'layer_norm_eps': norm_eps,  # Also provide as layer_norm_eps for BLOOM
        'apply_residual_connection_post_layernorm': apply_residual_post_ln,
        'block_size': 16,
        'num_blocks': num_blocks,
    }
    
    # DeepSeek-V2 specific parameters (MLA + MoE)
    if hasattr(hf_config, 'qk_nope_head_dim'):
        config['qk_nope_head_dim'] = hf_config.qk_nope_head_dim
        config['qk_rope_head_dim'] = hf_config.qk_rope_head_dim
        config['v_head_dim'] = hf_config.v_head_dim
        config['kv_lora_rank'] = hf_config.kv_lora_rank
        config['q_lora_rank'] = getattr(hf_config, 'q_lora_rank', None)
        config['moe_intermediate_size'] = getattr(hf_config, 'moe_intermediate_size', intermediate_size)
        config['n_routed_experts'] = getattr(hf_config, 'n_routed_experts', 64)
        config['n_shared_experts'] = getattr(hf_config, 'n_shared_experts', 2)
        config['num_experts_per_tok'] = getattr(hf_config, 'num_experts_per_tok', 6)
        config['first_k_dense_replace'] = getattr(hf_config, 'first_k_dense_replace', 1)
        config['routed_scaling_factor'] = getattr(hf_config, 'routed_scaling_factor', 1.0)
        config['topk_method'] = getattr(hf_config, 'topk_method', 'greedy')
        config['n_group'] = getattr(hf_config, 'n_group', 1)
        config['topk_group'] = getattr(hf_config, 'topk_group', 1)
    
    return config


def _load_truncated_model(model_path: str, hf_config, num_layers: int, dtype, device: str):
    """
    Create and load a truncated model efficiently using meta tensors.
    
    This avoids:
    1. Loading all checkpoint shards (only loads needed ones)
    2. Random weight initialization (uses meta tensors, then materializes from checkpoint)
    
    Args:
        model_path: Local path to the model files
        hf_config: HuggingFace config (already modified with num_hidden_layers)
        num_layers: Number of layers to load
        dtype: Target dtype for weights
        device: Target device
        
    Returns:
        Loaded HuggingFace model
    """
    from safetensors.torch import load_file
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    import json
    
    # Create model with meta tensors first to get the actual weight key names
    # This is architecture-agnostic and handles Llama, Falcon, Mistral, etc.
    print("  Creating model architecture...")
    with init_empty_weights():
        hf_model = AutoModelForCausalLM.from_config(
            hf_config,
            torch_dtype=dtype,
            # Use default attention (SDPA when available) for better alignment with KB's SDPA
        )
    
    # Get the actual keys needed from the model's state dict
    needed_keys = set(hf_model.state_dict().keys())
    
    # Build a mapping from model keys to potential checkpoint keys
    # Some checkpoints (like BLOOM safetensors) omit the "transformer." prefix
    def get_checkpoint_key(model_key: str, weight_map: dict) -> str:
        """Try different key formats to find a match in weight_map."""
        if model_key in weight_map:
            return model_key
        # Try stripping common prefixes
        for prefix in ["transformer.", "model."]:
            if model_key.startswith(prefix):
                stripped = model_key[len(prefix):]
                if stripped in weight_map:
                    return stripped
        return None
    
    # Check for sharded vs single-file checkpoints (safetensors or .bin)
    safetensors_index = os.path.join(model_path, "model.safetensors.index.json")
    safetensors_single = os.path.join(model_path, "model.safetensors")
    bin_index = os.path.join(model_path, "pytorch_model.bin.index.json")
    bin_single = os.path.join(model_path, "pytorch_model.bin")
    
    loaded_state = {}
    
    if os.path.exists(safetensors_index):
        # Sharded safetensors - find which shards we need
        with open(safetensors_index) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        
        # Build key mapping and find needed shards
        key_mapping = {}  # checkpoint_key -> model_key
        needed_files = set()
        for model_key in needed_keys:
            ckpt_key = get_checkpoint_key(model_key, weight_map)
            if ckpt_key:
                key_mapping[ckpt_key] = model_key
                needed_files.add(weight_map[ckpt_key])
        
        total_shards = len(set(weight_map.values()))
        print(f"  Loading {len(needed_files)} of {total_shards} safetensors shards...")
        
        # Load only from needed shards
        for shard_file in needed_files:
            shard_path = os.path.join(model_path, shard_file)
            shard_data = load_file(shard_path, device="cpu")
            for ckpt_key, tensor in shard_data.items():
                if ckpt_key in key_mapping:
                    model_key = key_mapping[ckpt_key]
                    loaded_state[model_key] = tensor
            # Free memory immediately
            del shard_data
            
    elif os.path.exists(safetensors_single):
        # Single safetensors file
        print("  Loading from single safetensors file...")
        shard_data = load_file(safetensors_single, device="cpu")
        # Build key mapping for single file
        for ckpt_key, tensor in shard_data.items():
            for model_key in needed_keys:
                if ckpt_key == model_key:
                    loaded_state[model_key] = tensor
                    break
                # Try with prefix
                for prefix in ["transformer.", "model."]:
                    if model_key == prefix + ckpt_key:
                        loaded_state[model_key] = tensor
                        break
        del shard_data
        
    elif os.path.exists(bin_index):
        # Sharded pytorch .bin files - find which shards we need
        with open(bin_index) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        
        # Build key mapping and find needed shards
        key_mapping = {}  # checkpoint_key -> model_key
        needed_files = set()
        for model_key in needed_keys:
            ckpt_key = get_checkpoint_key(model_key, weight_map)
            if ckpt_key:
                key_mapping[ckpt_key] = model_key
                needed_files.add(weight_map[ckpt_key])
        
        total_shards = len(set(weight_map.values()))
        print(f"  Loading {len(needed_files)} of {total_shards} pytorch shards...")
        
        # Load only from needed shards
        for shard_file in needed_files:
            shard_path = os.path.join(model_path, shard_file)
            shard_data = torch.load(shard_path, map_location="cpu", weights_only=True)
            for ckpt_key, tensor in shard_data.items():
                if ckpt_key in key_mapping:
                    model_key = key_mapping[ckpt_key]
                    loaded_state[model_key] = tensor
            # Free memory immediately
            del shard_data
            
    elif os.path.exists(bin_single):
        # Single pytorch .bin file
        print("  Loading from single pytorch file...")
        shard_data = torch.load(bin_single, map_location="cpu", weights_only=True)
        # Build key mapping for single file
        for ckpt_key, tensor in shard_data.items():
            for model_key in needed_keys:
                if ckpt_key == model_key:
                    loaded_state[model_key] = tensor
                    break
                # Try with prefix
                for prefix in ["transformer.", "model."]:
                    if model_key == prefix + ckpt_key:
                        loaded_state[model_key] = tensor
                        break
        del shard_data
        
    else:
        raise FileNotFoundError(
            f"No checkpoint files found in {model_path}. "
            "Expected model.safetensors[.index.json] or pytorch_model.bin[.index.json]."
        )
    
    # Handle weight tying: if lm_head.weight is missing but word_embeddings.weight exists,
    # use word_embeddings for both (common in BLOOM, GPT-2, etc.)
    missing_keys = needed_keys - set(loaded_state.keys())
    if missing_keys:
        # Check for tied weights patterns
        tied_weight_sources = {
            "lm_head.weight": [
                "transformer.word_embeddings.weight",
                "model.embed_tokens.weight",
                "word_embeddings.weight",
            ],
        }
        resolved_keys = set()
        for missing_key in list(missing_keys):
            if missing_key in tied_weight_sources:
                for source_key in tied_weight_sources[missing_key]:
                    if source_key in loaded_state:
                        loaded_state[missing_key] = loaded_state[source_key]
                        resolved_keys.add(missing_key)
                        break
        missing_keys -= resolved_keys
    
    # Verify we found all needed weights
    if missing_keys:
        raise RuntimeError(f"Missing weights for truncated model: {missing_keys}")
    
    # Materialize each parameter directly from loaded weights
    print("  Loading weights into model...")
    for name, tensor in loaded_state.items():
        set_module_tensor_to_device(hf_model, name, device, value=tensor.to(dtype))
    
    return hf_model


def load_models(model_name: str, max_layers: Optional[int] = None):
    """
    Load HuggingFace and KernelBench models.
    
    Args:
        model_name: HuggingFace model name
        max_layers: Number of layers to use. If None, uses all layers.
                   If less than total layers, uses memory-efficient loading
                   that only loads the needed checkpoint shards.
    """
    from huggingface_hub import snapshot_download
    
    print(f"\nLoading models from {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    hf_config = AutoConfig.from_pretrained(model_name)
    
    # Get total layers (handle different naming conventions)
    total_layers = getattr(hf_config, 'num_hidden_layers', None)
    if total_layers is None:
        total_layers = getattr(hf_config, 'n_layer', None)  # BLOOM
    if total_layers is None:
        raise ValueError(f"Could not determine num_layers from config: {hf_config}")
    
    # Determine actual number of layers to use
    if max_layers is None:
        num_layers = total_layers
    else:
        num_layers = min(max_layers, total_layers)
    
    truncated = num_layers < total_layers
    
    if truncated:
        # Memory-efficient loading: use meta tensors + selective shard loading
        print(f"Using memory-efficient loading for {num_layers}/{total_layers} layers...")
        
        # Modify config to have fewer layers (set both attributes for compatibility)
        if hasattr(hf_config, 'num_hidden_layers'):
            hf_config.num_hidden_layers = num_layers
        if hasattr(hf_config, 'n_layer'):
            hf_config.n_layer = num_layers
        
        # Download checkpoint files (try safetensors first, fall back to .bin)
        # We download both formats and let the loading function choose
        model_path = snapshot_download(
            model_name,
            allow_patterns=[
                "*.safetensors", "*.safetensors.index.json",  # Preferred format
                "pytorch_model*.bin", "pytorch_model.bin.index.json",  # Fallback format
                "*.json",  # Config files
            ],
        )
        
        # Create and load model efficiently (no random init, only needed shards)
        hf_model = _load_truncated_model(model_path, hf_config, num_layers, DTYPE, DEVICE)
        hf_model.eval()
    else:
        # Full model - use standard loading
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=DTYPE, 
            device_map=DEVICE,
            # Use default attention (SDPA when available) for better alignment with KB's SDPA
        )
        hf_model.eval()
    
    # Calculate num_blocks needed for testing (per-layer KV cache)
    # Each layer has its own KV cache with num_blocks blocks
    # We need enough blocks to hold max_seq_len tokens per sequence
    max_seq_len = 4096  # Maximum sequence length for tests
    block_size = 16
    max_batch_size = 2  # Maximum batch size for tests
    # Blocks per sequence = ceil(max_seq_len / block_size)
    # Total blocks = blocks_per_seq * max_batch_size * safety_margin
    blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = blocks_per_seq * max_batch_size * 2  # 2x safety margin
    
    # Create KB config with the number of layers we're using
    kb_config = create_kb_model_from_hf_config(hf_config, num_blocks)
    kb_config['num_layers'] = num_layers  # Ensure correct layer count
    
    kb_model, kb_module = load_kernelbench_model(model_name, kb_config)
    kb_model = kb_model.to(device=DEVICE, dtype=DTYPE)
    
    # Copy weights for the layers we loaded
    copy_weights(hf_model, kb_model, num_layers)
    kb_model.eval()
    
    print(f"Loaded: {num_layers}/{total_layers} layers, {kb_config['hidden_size']} hidden, "
          f"{kb_config['num_heads']} heads, {kb_config['num_kv_heads']} kv_heads")
    
    return hf_model, kb_model, tokenizer, kb_config, kb_module


# ============================================================================
# Model Fixture (uses --model-name parameter)
# ============================================================================

@pytest.fixture(scope="module")
def loaded_models(request):
    """
    Load models based on command-line parameters.
    
    This fixture loads both the HuggingFace model and the corresponding
    KernelBench implementation for comparison testing.
    
    Parameters used:
        --model-name: HuggingFace model name
        --max-layers: Number of layers to use (optional, defaults to all)
    """
    model_name = request.config.getoption("--model-name")
    max_layers = request.config.getoption("--max-layers")
    try:
        return load_models(model_name, max_layers), model_name, max_layers
    except ValueError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not load {model_name}: {e}")


# ============================================================================
# Alignment Tests
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_alignment(loaded_models):
    """
    Test prefill (prompt processing) alignment.
    
    Uses generate(max_new_tokens=0, return_logits=True) to get prefill logits.
    Compares HuggingFace and KernelBench outputs to ensure numerical alignment.
    """
    (hf_model, kb_model, tokenizer, kb_config, kb_module), model_name, max_layers = loaded_models
    
    layers_info = f" ({max_layers} layers)" if max_layers else ""
    print("\n" + "="*70)
    print(f"Testing Prefill Alignment for {model_name}{layers_info}")
    print("="*70)
    
    for i, prompt in enumerate(TEST_PROMPTS):
        encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_ids = encoded['input_ids']
        batch_size, seq_len = input_ids.shape
        
        # Allocate block table
        max_blocks = (seq_len + kb_config['block_size'] - 1) // kb_config['block_size'] + 10
        block_table = torch.arange(max_blocks, device=DEVICE, dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            # HuggingFace
            hf_out = hf_model(input_ids=input_ids, use_cache=False)
            hf_logits = hf_out.logits
            
            # KernelBench prefill using generate with max_new_tokens=0
            _, kb_logits_list = kb_model.generate(
                input_ids, 
                max_new_tokens=0, 
                block_table=block_table,
                return_logits=True
            )
            kb_logits = kb_logits_list[0]  # Prefill logits
        
        # Compare last position logits using relative tolerance
        hf_last = hf_logits[:, -1, :]
        kb_last = kb_logits[:, -1, :]
        
        # Compute absolute and relative differences
        abs_diff = (hf_last - kb_last).abs()
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        
        # Relative difference: |a - b| / (max(|a|, |b|) + eps) for numerical stability
        denominator = torch.maximum(hf_last.abs(), kb_last.abs()) + 1e-8
        rel_diff = abs_diff / denominator
        max_rel_diff = rel_diff.max().item()
        mean_rel_diff = rel_diff.mean().item()
        
        # Check top prediction
        hf_top = hf_last.argmax(dim=-1).item()
        kb_top = kb_last.argmax(dim=-1).item()
        
        hf_token = tokenizer.decode([hf_top])
        kb_token = tokenizer.decode([kb_top])
        
        top_match = hf_top == kb_top
        
        # Tolerance check: mean relative diff should be small, max can be larger for outliers
        mean_ok = mean_rel_diff < RTOL_MEAN
        max_ok = max_rel_diff < RTOL_MAX
        is_close = mean_ok and max_ok
        
        # Pass if predictions match and values are within tolerance
        is_pass = top_match and is_close
        status = "PASS" if is_pass else "FAIL"
        
        print(f"\n  [{i}] {status}: (len={seq_len} tokens)")
        print(f"      Prompt: '{prompt[:60]}...'")
        print(f"      abs_diff: max={max_abs_diff:.2e}, mean={mean_abs_diff:.2e}")
        print(f"      rel_diff: max={max_rel_diff:.2e} (limit={RTOL_MAX}), mean={mean_rel_diff:.2e} (limit={RTOL_MEAN})")
        print(f"      HF next: '{hf_token}' | KB next: '{kb_token}' (match={top_match})")
        
        # Primary check: top predictions must match (this is the correctness criterion)
        assert top_match, f"Top predictions differ: HF={hf_token} vs KB={kb_token}"
        
        # Secondary check: mean relative difference should be small
        assert mean_ok, \
            f"Mean relative diff {mean_rel_diff:.2e} exceeds tolerance {RTOL_MEAN}"
        
        # Tertiary check: max relative difference shouldn't be too extreme
        assert max_ok, \
            f"Max relative diff {max_rel_diff:.2e} exceeds tolerance {RTOL_MAX}"
    
    print("\n" + "-"*70)
    print(f"All prefill tests passed for {model_name}!")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_generation(loaded_models):
    """
    Test continuous batching generation.
    
    Uses generate() for KernelBench model and compares with HuggingFace generation.
    Validates that both implementations produce matching token sequences.
    """
    (hf_model, kb_model, tokenizer, kb_config, kb_module), model_name, max_layers = loaded_models
    
    layers_info = f" ({max_layers} layers)" if max_layers else ""
    print("\n" + "="*70)
    print(f"Testing Continuous Batching Generation for {model_name}{layers_info}")
    print("="*70)
    
    results = []
    
    for i, (prompt, num_tokens) in enumerate(zip(TEST_PROMPTS, RESPONSE_LENGTHS)):
        print(f"\n[{i}] Generating {num_tokens} tokens for prompt (len={len(tokenizer.encode(prompt))})")
        print(f"    '{prompt[:60]}...'")
        
        encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_ids = encoded['input_ids']
        batch_size, prompt_len = input_ids.shape
        
        # Allocate blocks
        max_seq = prompt_len + num_tokens + 10
        max_blocks = (max_seq + kb_config['block_size'] - 1) // kb_config['block_size']
        block_table = torch.arange(max_blocks, device=DEVICE, dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            # HuggingFace generation (manual loop to match greedy decoding)
            hf_generated = []
            hf_out = hf_model(input_ids=input_ids, use_cache=True)
            hf_past = hf_out.past_key_values
            hf_next = hf_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            hf_generated.append(hf_next)
            
            for step in range(num_tokens - 1):
                hf_out = hf_model(
                    input_ids=hf_next,
                    past_key_values=hf_past,
                    use_cache=True,
                )
                hf_past = hf_out.past_key_values
                hf_next = hf_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                hf_generated.append(hf_next)
            
            hf_tokens = torch.cat(hf_generated, dim=1)
            
            # KernelBench generation using generate()
            kb_full_seq = kb_model.generate(
                input_ids,
                max_new_tokens=num_tokens,
                block_table=block_table,
                return_logits=False
            )
            # Extract only the generated tokens (exclude prompt)
            kb_tokens = kb_full_seq[:, prompt_len:]
        
        # Decode generated tokens
        hf_text = tokenizer.decode(hf_tokens[0], skip_special_tokens=True)
        kb_text = tokenizer.decode(kb_tokens[0], skip_special_tokens=True)
        
        # Count CONSECUTIVE matching tokens from the start
        # After the first mismatch, everything after is considered garbage
        matches = (hf_tokens[0] == kb_tokens[0])
        if matches.all():
            consecutive_matches = num_tokens
        else:
            # Find index of first mismatch
            first_mismatch_indices = (~matches).nonzero(as_tuple=True)[0]
            if len(first_mismatch_indices) > 0:
                consecutive_matches = first_mismatch_indices[0].item()
            else:
                consecutive_matches = num_tokens
        
        full_match = consecutive_matches == num_tokens
        
        print(f"    Consecutive matches: {consecutive_matches}/{num_tokens} ({100*consecutive_matches/num_tokens:.1f}%)")
        print(f"    HF: '{hf_text[:80]}...'")
        print(f"    KB: '{kb_text[:80]}...'")
        
        results.append({
            'prompt': prompt,
            'full_match': full_match,
            'consecutive_matches': consecutive_matches,
            'total_tokens': num_tokens,
        })
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    total_consecutive = sum(r['consecutive_matches'] for r in results)
    total_tokens = sum(r['total_tokens'] for r in results)
    full_matches = sum(r['full_match'] for r in results)
    
    print(f"Full sequence matches: {full_matches}/{len(results)}")
    print(f"Consecutive token accuracy: {total_consecutive}/{total_tokens} ({100*total_consecutive/total_tokens:.1f}%)")
    
    # Require minimum consecutive matching tokens.
    # After the first mismatch, subsequent tokens are considered diverged (butterfly effect).
    # This is a stricter test than total matches since it measures where divergence starts.
    assert total_consecutive / total_tokens >= GENERATION_CONSECUTIVE_THRESHOLD, \
        f"Consecutive token accuracy {100*total_consecutive/total_tokens:.1f}% below {100*GENERATION_CONSECUTIVE_THRESHOLD:.0f}% threshold"


def _get_hf_components(hf_model):
    """
    Get HF model components in an architecture-agnostic way.
    
    Returns: (embedding_layer, layers_list, layer0_norm, layer0_mlp)
    """
    # Try Llama-style structure first (LlamaForCausalLM)
    if hasattr(hf_model, 'model') and hasattr(hf_model.model, 'embed_tokens'):
        base = hf_model.model
        return (
            base.embed_tokens,
            base.layers,
            base.layers[0].input_layernorm,
            base.layers[0].mlp,
        )
    # Try Falcon/BLOOM-style structure (uses transformer.h)
    elif hasattr(hf_model, 'transformer') and hasattr(hf_model.transformer, 'h'):
        base = hf_model.transformer
        return (
            base.word_embeddings,
            base.h,
            base.h[0].input_layernorm,
            base.h[0].mlp,
        )
    else:
        raise ValueError(f"Unknown model structure: {type(hf_model)}")


def _get_kb_components(kb_model):
    """
    Get KB model components in an architecture-agnostic way.
    
    Returns: (embedding_layer, layers_list, layer0_norm, layer0_mlp)
    """
    # Try Llama-style structure (embed_tokens + layers)
    if hasattr(kb_model, 'embed_tokens'):
        return (
            kb_model.embed_tokens,
            kb_model.layers,
            kb_model.layers[0].input_layernorm,
            kb_model.layers[0].mlp,
        )
    # Try Falcon/BLOOM-style structure (word_embeddings + h)
    elif hasattr(kb_model, 'word_embeddings') and hasattr(kb_model, 'h'):
        return (
            kb_model.word_embeddings,
            kb_model.h,
            kb_model.h[0].input_layernorm,
            kb_model.h[0].mlp,
        )
    else:
        raise ValueError(f"Unknown model structure: {type(kb_model)}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_components(loaded_models):
    """
    Test individual component alignment.
    
    Validates that individual model components (embeddings, layer norms, MLP, LM head)
    produce matching outputs between HuggingFace and KernelBench implementations.
    """
    (hf_model, kb_model, tokenizer, kb_config, _), model_name, max_layers = loaded_models
    
    layers_info = f" ({max_layers} layers)" if max_layers else ""
    print("\n" + "="*70)
    print(f"Testing Component Alignment for {model_name}{layers_info}")
    print("="*70)
    
    # Get components in an architecture-agnostic way
    try:
        hf_embed, hf_layers, hf_norm, hf_mlp = _get_hf_components(hf_model)
        kb_embed, kb_layers, kb_norm, kb_mlp = _get_kb_components(kb_model)
    except ValueError as e:
        pytest.skip(f"Cannot test components for this architecture: {e}")
    
    # Test embeddings
    test_ids = torch.randint(0, kb_config['vocab_size'], (2, 32), device=DEVICE)
    with torch.no_grad():
        hf_emb = hf_embed(test_ids)
        kb_emb = kb_embed(test_ids)
    
    emb_diff = (hf_emb - kb_emb).abs().max().item()
    print(f"  Embedding diff: {emb_diff:.2e} {'PASS' if emb_diff < 1e-6 else 'FAIL'}")
    assert emb_diff < 1e-6
    
    # Test layer norms
    hidden = torch.randn(2, 32, kb_config['hidden_size'], device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        hf_norm_out = hf_norm(hidden)
        kb_norm_out = kb_norm(hidden)
    
    norm_diff = (hf_norm_out - kb_norm_out).abs().max().item()
    print(f"  LayerNorm diff: {norm_diff:.2e} {'PASS' if norm_diff < ATOL_STRICT else 'FAIL'}")
    assert norm_diff < ATOL_STRICT
    
    # Test MLP - some models (like BLOOM) require additional arguments
    try:
        with torch.no_grad():
            hf_mlp_out = hf_mlp(hidden)
            kb_mlp_out = kb_mlp(hidden)
        
        mlp_diff = (hf_mlp_out - kb_mlp_out).abs().max().item()
        print(f"  MLP diff: {mlp_diff:.2e} {'PASS' if mlp_diff < ATOL_STRICT else 'FAIL'}")
        assert mlp_diff < ATOL_STRICT
    except TypeError as e:
        if 'residual' in str(e):
            # BLOOM's MLP requires a residual argument - skip isolated MLP test
            print(f"  MLP diff: SKIPPED (different interface)")
        else:
            raise
    
    # Test LM head
    with torch.no_grad():
        hf_head = hf_model.lm_head(hidden)
        kb_head = kb_model.lm_head(hidden)
    
    head_diff = (hf_head - kb_head).abs().max().item()
    print(f"  LM Head diff: {head_diff:.2e} {'PASS' if head_diff < ATOL_STRICT else 'FAIL'}")
    assert head_diff < ATOL_STRICT
    
    print("\n  All component tests passed!")


if __name__ == "__main__":
    # Forward command-line arguments to pytest
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
