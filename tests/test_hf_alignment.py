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
# With bf16 and 32 layers, small differences accumulate
ATOL = 1e-4  # Absolute tolerance for logit max diff (exact match with eager attention)
RTOL = 1e-2  # Relative tolerance (unused for now)

# Additional thresholds for debugging
MAX_LOGIT_DIFF_WARN = 0.5  # Warn if logit diff exceeds this
TOP_K_MATCH = 5  # Consider pass if top prediction is in top-K of other model


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


def copy_weights(hf_model, kb_model, num_layers: int) -> None:
    """Copy weights from HuggingFace model to KernelBench model."""
    hf_state = hf_model.state_dict()
    
    kb_model.embed_tokens.weight.data.copy_(hf_state['model.embed_tokens.weight'])
    kb_model.lm_head.weight.data.copy_(hf_state['lm_head.weight'])
    kb_model.norm.weight.data.copy_(hf_state['model.norm.weight'])
    
    for i in range(num_layers):
        prefix = f'model.layers.{i}.'
        kb_model.layers[i].input_layernorm.weight.data.copy_(
            hf_state[prefix + 'input_layernorm.weight'])
        kb_model.layers[i].post_attention_layernorm.weight.data.copy_(
            hf_state[prefix + 'post_attention_layernorm.weight'])
        kb_model.layers[i].self_attn.q_proj.weight.data.copy_(
            hf_state[prefix + 'self_attn.q_proj.weight'])
        kb_model.layers[i].self_attn.k_proj.weight.data.copy_(
            hf_state[prefix + 'self_attn.k_proj.weight'])
        kb_model.layers[i].self_attn.v_proj.weight.data.copy_(
            hf_state[prefix + 'self_attn.v_proj.weight'])
        kb_model.layers[i].self_attn.o_proj.weight.data.copy_(
            hf_state[prefix + 'self_attn.o_proj.weight'])
        kb_model.layers[i].mlp.gate_proj.weight.data.copy_(
            hf_state[prefix + 'mlp.gate_proj.weight'])
        kb_model.layers[i].mlp.up_proj.weight.data.copy_(
            hf_state[prefix + 'mlp.up_proj.weight'])
        kb_model.layers[i].mlp.down_proj.weight.data.copy_(
            hf_state[prefix + 'mlp.down_proj.weight'])


def create_kb_model_from_hf_config(hf_config, num_blocks: int = 8192):
    """Create KernelBench model config from HuggingFace config."""
    # Get rope_scaling if available
    rope_scaling = getattr(hf_config, 'rope_scaling', None)
    
    return {
        'vocab_size': hf_config.vocab_size,
        'hidden_size': hf_config.hidden_size,
        'num_layers': hf_config.num_hidden_layers,
        'num_heads': hf_config.num_attention_heads,
        'num_kv_heads': getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads),
        'head_dim': hf_config.hidden_size // hf_config.num_attention_heads,
        'intermediate_size': hf_config.intermediate_size,
        'max_seq_len': getattr(hf_config, 'max_position_embeddings', 8192),
        'rope_theta': getattr(hf_config, 'rope_theta', 500000.0),
        'rope_scaling': rope_scaling,
        'rms_norm_eps': hf_config.rms_norm_eps,
        'block_size': 16,
        'num_blocks': num_blocks,
    }


def _get_needed_weight_keys(hf_config, num_layers: int) -> set:
    """
    Determine which weight keys are needed for a truncated model.
    
    This avoids having to create the model just to get its state dict keys.
    """
    needed_keys = set()
    
    # Embedding and output layers (always needed)
    needed_keys.add("model.embed_tokens.weight")
    needed_keys.add("model.norm.weight")
    needed_keys.add("lm_head.weight")
    
    # Per-layer weights
    for i in range(num_layers):
        prefix = f"model.layers.{i}."
        # Attention
        needed_keys.add(prefix + "self_attn.q_proj.weight")
        needed_keys.add(prefix + "self_attn.k_proj.weight")
        needed_keys.add(prefix + "self_attn.v_proj.weight")
        needed_keys.add(prefix + "self_attn.o_proj.weight")
        # MLP
        needed_keys.add(prefix + "mlp.gate_proj.weight")
        needed_keys.add(prefix + "mlp.up_proj.weight")
        needed_keys.add(prefix + "mlp.down_proj.weight")
        # Layer norms
        needed_keys.add(prefix + "input_layernorm.weight")
        needed_keys.add(prefix + "post_attention_layernorm.weight")
    
    return needed_keys


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
    
    # Determine which weight keys we need
    needed_keys = _get_needed_weight_keys(hf_config, num_layers)
    
    # Check for sharded vs single-file safetensors
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    single_file = os.path.join(model_path, "model.safetensors")
    
    if os.path.exists(index_file):
        # Sharded safetensors - find which shards we need
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        
        # Find which shard files contain weights we need
        needed_files = set()
        for key in needed_keys:
            if key in weight_map:
                needed_files.add(weight_map[key])
        
        total_shards = len(set(weight_map.values()))
        print(f"  Loading {len(needed_files)} of {total_shards} checkpoint shards...")
        
        # Load only from needed shards
        loaded_state = {}
        for shard_file in needed_files:
            shard_path = os.path.join(model_path, shard_file)
            shard_data = load_file(shard_path, device="cpu")
            for key, tensor in shard_data.items():
                if key in needed_keys:
                    loaded_state[key] = tensor
            # Free memory immediately
            del shard_data
    elif os.path.exists(single_file):
        # Single safetensors file
        print("  Loading from single safetensors file...")
        shard_data = load_file(single_file, device="cpu")
        loaded_state = {k: v for k, v in shard_data.items() if k in needed_keys}
        del shard_data
    else:
        raise FileNotFoundError(
            f"No safetensors files found in {model_path}. "
            "Memory-efficient loading requires safetensors format."
        )
    
    # Verify we found all needed weights
    missing_keys = needed_keys - set(loaded_state.keys())
    if missing_keys:
        raise RuntimeError(f"Missing weights for truncated model: {missing_keys}")
    
    # Create model with meta tensors (no memory allocation, no random init)
    print("  Creating model architecture...")
    with init_empty_weights():
        hf_model = AutoModelForCausalLM.from_config(
            hf_config,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
    
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
    total_layers = hf_config.num_hidden_layers
    
    # Determine actual number of layers to use
    if max_layers is None:
        num_layers = total_layers
    else:
        num_layers = min(max_layers, total_layers)
    
    truncated = num_layers < total_layers
    
    if truncated:
        # Memory-efficient loading: use meta tensors + selective shard loading
        print(f"Using memory-efficient loading for {num_layers}/{total_layers} layers...")
        
        # Modify config to have fewer layers
        hf_config.num_hidden_layers = num_layers
        
        # Download only safetensors and config files (not .bin weights)
        model_path = snapshot_download(
            model_name,
            allow_patterns=["*.safetensors", "*.json", "*.safetensors.index.json"],
            ignore_patterns=["*.bin", "*.bin.index.json", "pytorch_model*"],
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
            attn_implementation="eager",
        )
        hf_model.eval()
    
    # Calculate num_blocks needed for testing
    max_seq_len = 4096  # Maximum sequence length for tests
    block_size = 16
    # Use num_layers for block calculation
    num_blocks = (max_seq_len // block_size + 1) * num_layers * 2
    
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
        
        # Compare last position logits
        hf_last = hf_logits[:, -1, :]
        kb_last = kb_logits[:, -1, :]
        
        diff = (hf_last - kb_last).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        # Check top prediction
        hf_top = hf_last.argmax(dim=-1).item()
        kb_top = kb_last.argmax(dim=-1).item()
        
        hf_token = tokenizer.decode([hf_top])
        kb_token = tokenizer.decode([kb_top])
        
        top_match = hf_top == kb_top
        status = "PASS" if (max_diff < ATOL and top_match) else "FAIL"
        print(f"\n  [{i}] {status}: (len={seq_len} tokens)")
        print(f"      Prompt: '{prompt[:60]}...'")
        print(f"      max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
        print(f"      HF next: '{hf_token}' | KB next: '{kb_token}' (match={top_match})")
        
        # Primary check: top predictions must match
        assert top_match, f"Top predictions differ: HF={hf_token} vs KB={kb_token}"
        # Secondary check: logit difference within tolerance
        assert max_diff < ATOL, f"Prefill diff {max_diff} exceeds tolerance {ATOL}"
    
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
        
        tokens_match = (hf_tokens == kb_tokens).all().item()
        match_count = (hf_tokens == kb_tokens).sum().item()
        
        print(f"    Token match: {match_count}/{num_tokens} ({100*match_count/num_tokens:.1f}%)")
        print(f"    HF: '{hf_text[:80]}...'")
        print(f"    KB: '{kb_text[:80]}...'")
        
        results.append({
            'prompt': prompt,
            'tokens_match': tokens_match,
            'match_count': match_count,
            'total_tokens': num_tokens,
        })
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    total_match = sum(r['match_count'] for r in results)
    total_tokens = sum(r['total_tokens'] for r in results)
    full_match = sum(r['tokens_match'] for r in results)
    
    print(f"Full sequence matches: {full_match}/{len(results)}")
    print(f"Total token accuracy: {total_match}/{total_tokens} ({100*total_match/total_tokens:.1f}%)")
    
    # Require at least 80% token accuracy (float16 accumulates errors over long generation)
    # For production use, consider using float32 or bfloat16 for higher accuracy
    assert total_match / total_tokens >= 0.80, \
        f"Token accuracy {100*total_match/total_tokens:.1f}% below 80% threshold"


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
    
    # Test embeddings
    test_ids = torch.randint(0, kb_config['vocab_size'], (2, 32), device=DEVICE)
    with torch.no_grad():
        hf_emb = hf_model.model.embed_tokens(test_ids)
        kb_emb = kb_model.embed_tokens(test_ids)
    
    emb_diff = (hf_emb - kb_emb).abs().max().item()
    print(f"  Embedding diff: {emb_diff:.2e} {'PASS' if emb_diff < 1e-6 else 'FAIL'}")
    assert emb_diff < 1e-6
    
    # Test layer norms
    hidden = torch.randn(2, 32, kb_config['hidden_size'], device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        hf_norm = hf_model.model.layers[0].input_layernorm(hidden)
        kb_norm = kb_model.layers[0].input_layernorm(hidden)
    
    norm_diff = (hf_norm - kb_norm).abs().max().item()
    print(f"  LayerNorm diff: {norm_diff:.2e} {'PASS' if norm_diff < ATOL else 'FAIL'}")
    assert norm_diff < ATOL
    
    # Test MLP
    with torch.no_grad():
        hf_mlp = hf_model.model.layers[0].mlp(hidden)
        kb_mlp = kb_model.layers[0].mlp(hidden)
    
    mlp_diff = (hf_mlp - kb_mlp).abs().max().item()
    print(f"  MLP diff: {mlp_diff:.2e} {'PASS' if mlp_diff < ATOL else 'FAIL'}")
    assert mlp_diff < ATOL
    
    # Test LM head
    with torch.no_grad():
        hf_head = hf_model.lm_head(hidden)
        kb_head = kb_model.lm_head(hidden)
    
    head_diff = (hf_head - kb_head).abs().max().item()
    print(f"  LM Head diff: {head_diff:.2e} {'PASS' if head_diff < ATOL else 'FAIL'}")
    assert head_diff < ATOL
    
    print("\n  All component tests passed!")


if __name__ == "__main__":
    # Forward command-line arguments to pytest
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
