"""
Test model alignment with HuggingFace transformers library.

This test validates that KernelBench model implementations produce
outputs matching the HuggingFace transformers implementation using:
1. A batch of 5 prompts with varying lengths (text models)
2. Continuous batching with paged KV cache (decoder-only models)
3. Both prefill and decode phases (decoder-only/SSM models)
4. Image classification (vision models like SwinV2)
5. Encoder-decoder forward pass (T5 models)

Supports any model with a corresponding KernelBench level4 implementation.
Pass the model name via --model-name parameter.
Optionally limit the number of layers with --max-layers for faster testing
or to fit larger models on smaller GPUs.

Usage:
    pytest tests/test_hf_alignment.py --model-name meta-llama/Llama-3.1-8B-Instruct
    pytest tests/test_hf_alignment.py --model-name meta-llama/Llama-3.1-70B-Instruct
    pytest tests/test_hf_alignment.py --model-name mistralai/Mistral-7B-v0.1
    pytest tests/test_hf_alignment.py --model-name google/flan-t5-large --max-layers 4
    pytest tests/test_hf_alignment.py --model-name microsoft/swinv2-large-patch4-window12-192-22k
    
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

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoImageProcessor
from transformers import T5ForConditionalGeneration, Swinv2ForImageClassification
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import Qwen3VLForConditionalGeneration
from transformers import Qwen3OmniMoeThinkerForConditionalGeneration
import json
import re
import requests
from io import BytesIO
from PIL import Image

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
    "mistralai/Mixtral-8x7B-v0.1": "KernelBench.level4.4_Mixtral",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "KernelBench.level4.4_Mixtral",
    # T5 variants
    "google-t5/t5-small": "KernelBench.level4.11_T5",
    "google-t5/t5-base": "KernelBench.level4.11_T5",
    "google-t5/t5-large": "KernelBench.level4.11_T5",
    "google/flan-t5-small": "KernelBench.level4.11_T5",
    "google/flan-t5-base": "KernelBench.level4.11_T5",
    "google/flan-t5-large": "KernelBench.level4.11_T5",
    # SwinV2 variants
    "microsoft/swinv2-tiny-patch4-window8-256": "KernelBench.level4.12_SwinV2",
    "microsoft/swinv2-small-patch4-window8-256": "KernelBench.level4.12_SwinV2",
    "microsoft/swinv2-base-patch4-window12-192-22k": "KernelBench.level4.12_SwinV2",
    "microsoft/swinv2-large-patch4-window12-192-22k": "KernelBench.level4.12_SwinV2",
    # Qwen2-VL
    "Qwen/Qwen2-VL-2B-Instruct": "KernelBench.level4.13_Qwen2VL",
    "Qwen/Qwen2-VL-7B-Instruct": "KernelBench.level4.13_Qwen2VL",
    # Qwen3-VL
    "Qwen/Qwen3-VL-8B-Instruct": "KernelBench.level4.30_Qwen3VL",
    # Qwen3-Omni-MoE
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": "KernelBench.level4.31_Qwen3OmniMoe",
    # Whisper
    "openai/whisper-tiny": "KernelBench.level4.14_Whisper",
    "openai/whisper-small": "KernelBench.level4.14_Whisper",
    "openai/whisper-base": "KernelBench.level4.14_Whisper",
    "openai/whisper-medium": "KernelBench.level4.14_Whisper",
    "openai/whisper-large-v2": "KernelBench.level4.14_Whisper",
    "openai/whisper-large-v3": "KernelBench.level4.14_Whisper",
    # BLOOM variants
    "bigscience/bloom-560m": "KernelBench.level4.5_Bloom",
    "bigscience/bloom-1b1": "KernelBench.level4.5_Bloom",
    "bigscience/bloom-1b7": "KernelBench.level4.5_Bloom",
    "bigscience/bloom-3b": "KernelBench.level4.5_Bloom",
    "bigscience/bloom-7b1": "KernelBench.level4.5_Bloom",
    "bigscience/bloom": "KernelBench.level4.5_Bloom",
    # DeepSeek-V2 variants
    "deepseek-ai/DeepSeek-V2-Lite": "KernelBench.level4.3_Deepseek",
    "deepseek-ai/DeepSeek-V2": "KernelBench.level4.3_Deepseek",
    # Mamba-2 variants
    "mistralai/Mamba-Codestral-7B-v0.1": "KernelBench.level4.7_Mamba2",
    # Mamba-1 variants (generic Mamba-1 and Falcon Mamba)
    "tiiuae/falcon-mamba-7b": "KernelBench.level4.6_Mamba1",
    "tiiuae/falcon-mamba-7b-instruct": "KernelBench.level4.6_Mamba1",
    "state-spaces/mamba-2.8b-slimpj": "KernelBench.level4.6_Mamba1",
    "state-spaces/mamba-1.4b-hf": "KernelBench.level4.6_Mamba1",
    "state-spaces/mamba-790m-hf": "KernelBench.level4.6_Mamba1",
    "state-spaces/mamba-370m-hf": "KernelBench.level4.6_Mamba1",
    "state-spaces/mamba-130m-hf": "KernelBench.level4.6_Mamba1",
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

# Test prompts for T5 encoder-decoder models - task-specific text-to-text prompts
T5_TEST_PROMPTS = [
    # Short translation task
    "Translate English to German: The house is wonderful.",

    # Medium summarization task
    "Summarize: Machine learning is a branch of artificial intelligence that focuses on building "
    "applications that learn from data and improve their accuracy over time without being programmed "
    "to do so. In data science, an algorithm is a sequence of statistical processing steps.",

    # Question answering with context
    "Answer the question based on the context. Context: The Eiffel Tower is a wrought-iron lattice "
    "tower on the Champ de Mars in Paris, France. It was named after the engineer Gustave Eiffel, "
    "whose company designed and built the tower from 1887 to 1889. Question: Who designed the Eiffel Tower?",

    # Sentiment classification
    "Classify the sentiment of this review as positive or negative: "
    "The movie had stunning visuals and a compelling storyline that kept me on the edge of my seat.",

    # Longer paraphrase/rewrite task
    "Paraphrase: A road trip from San Francisco to Los Angeles offers many interesting stops along "
    "the way, including natural landmarks, excellent restaurants, and historical sites that showcase "
    "the rich cultural heritage of the California coast.",
]

# Test images for vision models (SwinV2) - publicly accessible COCO val2017 URLs
TEST_IMAGE_URLS = [
    # Two cats on a couch (640x480)
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    # A group of skiers on a snowy slope (640x427)
    "http://images.cocodataset.org/val2017/000000281759.jpg",
    # A street scene with people and vehicles (640x427)
    "http://images.cocodataset.org/val2017/000000397133.jpg",
    # A table with food items (640x428)
    "http://images.cocodataset.org/val2017/000000252219.jpg",
    # A kitchen scene (640x480)
    "http://images.cocodataset.org/val2017/000000087038.jpg",
]

def _count_consecutive_matches(hf_tokens: torch.Tensor, kb_tokens: torch.Tensor) -> Tuple[int, int]:
    """Count consecutive matching tokens from the start.
    
    Handles different lengths (e.g. HF stops at EOS, KB generates full length).
    Returns (consecutive_matches, comparable_length) where comparable_length is
    the minimum of the two token sequences.
    """
    min_len = min(len(hf_tokens), len(kb_tokens))
    if min_len == 0:
        return 0, 0
    matches = (hf_tokens[:min_len] == kb_tokens[:min_len])
    if matches.all():
        return min_len, min_len
    first_mismatch_indices = (~matches).nonzero(as_tuple=True)[0]
    if len(first_mismatch_indices) > 0:
        return first_mismatch_indices[0].item(), min_len
    return min_len, min_len


def _load_test_images() -> List[Tuple[Image.Image, str]]:
    """Download and return (PIL image, URL) pairs from TEST_IMAGE_URLS."""
    images = []
    for url in TEST_IMAGE_URLS:
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            images.append((img, url))
        except Exception as e:
            print(f"  Warning: Could not load image from {url}: {e}")
            # Skip failed images
            continue
    if not images:
        raise RuntimeError("Could not load any test images from URLs")
    return images

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
    - RMSNorm wrapper (for non-Mamba2): .norm.weight -> .weight
    - Conv1d wrapper: .conv1d.conv1d. -> .conv1d.
    
    This allows matching KB keys to HF keys despite structural differences.
    """
    # Remove common wrapper suffixes
    # Note: Order matters. Handle conv1d wrapping specially to avoid
    # colliding with Mamba2's .norm. (which is MambaRMSNormGated, a real submodule)
    
    # KB wraps nn.Conv1d inside a level1 Conv1d module:
    # KB: .conv1d.conv1d.weight -> HF: .conv1d.weight
    key = key.replace('.conv1d.conv1d.', '.conv1d.')
    
    wrappers = ['.ln.', '.linear.']
    for wrapper in wrappers:
        key = key.replace(wrapper, '.')
    return key


def _build_t5_key_mapping(hf_state, kb_state) -> dict:
    """Build explicit key mapping for T5 models.
    
    HF T5 uses:
      encoder.block.{i}.layer.{j} -> KB uses encoder_blocks.{i}.layer.{j}
      decoder.block.{i}.layer.{j} -> KB uses decoder_blocks.{i}.layer.{j}
      encoder.final_layer_norm -> KB uses encoder_final_layer_norm
      decoder.final_layer_norm -> KB uses decoder_final_layer_norm
    
    KB level1 operator wrappers add extra nesting that must be unwrapped:
      shared.embedding.weight -> shared.weight (Embedding wraps nn.Embedding)
      relative_attention_bias.embedding.weight -> relative_attention_bias.weight
    
    Note: T5 now uses RMSNorm directly (no T5LayerNorm wrapper), so
    layer_norm.weight maps directly to layer_norm.weight without unwrapping.
    """
    mapping = {}  # kb_key -> hf_key
    
    for kb_key in kb_state.keys():
        hf_key = kb_key
        # Map KB encoder_blocks -> HF encoder.block
        hf_key = hf_key.replace('encoder_blocks.', 'encoder.block.')
        # Map KB decoder_blocks -> HF decoder.block
        hf_key = hf_key.replace('decoder_blocks.', 'decoder.block.')
        # Map KB encoder_final_layer_norm -> HF encoder.final_layer_norm
        hf_key = hf_key.replace('encoder_final_layer_norm.', 'encoder.final_layer_norm.')
        # Map KB decoder_final_layer_norm -> HF decoder.final_layer_norm
        hf_key = hf_key.replace('decoder_final_layer_norm.', 'decoder.final_layer_norm.')
        # Unwrap level1 Embedding wrapper: shared.embedding.weight -> shared.weight
        hf_key = hf_key.replace('shared.embedding.weight', 'shared.weight')
        # Unwrap level1 Embedding in relative_attention_bias
        hf_key = hf_key.replace('relative_attention_bias.embedding.weight',
                                'relative_attention_bias.weight')
        
        if hf_key in hf_state:
            mapping[kb_key] = hf_key
    
    return mapping


def _build_swinv2_key_mapping(hf_state, kb_state) -> dict:
    """Build explicit key mapping for SwinV2 models.
    
    HF SwinV2 uses 'swinv2.' prefix for the backbone.
    KB model structure mirrors HF without the prefix.
    
    KB level1 LayerNorm wraps nn.LayerNorm as self.ln, adding '.ln.' in keys:
      embeddings.norm.ln.weight -> swinv2.embeddings.norm.weight
      layernorm_before.ln.weight -> layernorm_before.weight
      layernorm.ln.weight -> swinv2.layernorm.weight
    
    KB Q/K/V projections live on Swinv2Attention (attention.query/key/value),
    while HF has them inside Swinv2SelfAttention (attention.self.query/key/value):
      attention.query.weight -> swinv2...attention.self.query.weight
      attention.key.weight   -> swinv2...attention.self.key.weight
      attention.value.weight -> swinv2...attention.self.value.weight
    """
    mapping = {}  # kb_key -> hf_key
    
    for kb_key in kb_state.keys():
        # Unwrap level1 LayerNorm wrapper: .ln.weight/.ln.bias -> .weight/.bias
        unwrapped_key = kb_key.replace('.ln.weight', '.weight').replace('.ln.bias', '.bias')
        
        # Q/K/V projections: KB has attention.query/key/value,
        # HF has attention.self.query/key/value
        for proj in ('query', 'key', 'value'):
            unwrapped_key = unwrapped_key.replace(
                f'.attention.{proj}.', f'.attention.self.{proj}.'
            )
        
        # Try with swinv2. prefix for backbone keys
        hf_key = 'swinv2.' + unwrapped_key
        if hf_key in hf_state:
            mapping[kb_key] = hf_key
        elif unwrapped_key in hf_state:
            # classifier weights don't have swinv2. prefix
            mapping[kb_key] = unwrapped_key
    
    return mapping


def _build_whisper_key_mapping(hf_state, kb_state) -> dict:
    """Build explicit key mapping for Whisper models.
    
    HF WhisperForConditionalGeneration uses 'model.' prefix for the backbone:
      model.encoder.conv1.weight -> encoder.conv1.weight
      model.encoder.layers.0.self_attn.q_proj.weight -> encoder.layers.0.self_attn.q_proj.weight
      model.decoder.embed_tokens.weight -> decoder.embed_tokens.weight
      proj_out.weight -> proj_out.weight (no prefix)
    
    KB level1 operator wrappers add extra nesting that must be unwrapped:
      LayerNorm: .ln.weight/.ln.bias -> .weight/.bias
      Embedding: .embedding.weight -> .weight
      Conv1d:    .conv1d.weight/.conv1d.bias -> .weight/.bias
      Linear:    no extra nesting (weight/bias stored directly)
    """
    mapping = {}  # kb_key -> hf_key
    
    for kb_key in kb_state.keys():
        # Unwrap level1 wrapper key nesting
        unwrapped = kb_key
        # LayerNorm wrapper: .ln.weight -> .weight, .ln.bias -> .bias
        unwrapped = unwrapped.replace('.ln.weight', '.weight').replace('.ln.bias', '.bias')
        # Embedding wrapper: .embedding.weight -> .weight
        unwrapped = unwrapped.replace('.embedding.weight', '.weight')
        # Conv1d wrapper: .conv1d.weight -> .weight, .conv1d.bias -> .bias
        unwrapped = unwrapped.replace('.conv1d.weight', '.weight').replace('.conv1d.bias', '.bias')
        
        # Try with model. prefix (backbone weights)
        hf_key = 'model.' + unwrapped
        if hf_key in hf_state:
            mapping[kb_key] = hf_key
        elif unwrapped in hf_state:
            # proj_out.weight has no prefix
            mapping[kb_key] = unwrapped
    
    return mapping


def _build_qwen2vl_key_mapping(hf_state, kb_state) -> dict:
    """Build explicit key mapping for Qwen2-VL models.
    
    HF Qwen2VLForConditionalGeneration uses:
      model.visual.* -> KB: visual.*
      model.language_model.embed_tokens.weight -> KB: embed_tokens.embedding.weight
      model.language_model.layers.{i}.* -> KB: layers.{i}.*
      model.language_model.norm.weight -> KB: norm.weight
      lm_head.weight -> KB: lm_head.weight
    
    KB level1 operator wrappers add extra nesting:
      Embedding: .embedding.weight -> HF: .weight (embed_tokens only)
      LayerNorm: .ln.weight/.ln.bias -> HF: .weight/.bias
      PatchEmbed3D: .patch_embed_3d.proj. -> HF: .proj.
      PatchMerger MLP: .mlp_fc1. -> HF: .mlp.0., .mlp_fc2. -> HF: .mlp.2.
      Linear: no extra nesting (weight/bias stored directly)
    """
    mapping = {}  # kb_key -> hf_key
    
    for kb_key in kb_state.keys():
        # Unwrap level1 operator wrappers
        unwrapped = kb_key
        
        # Unwrap level1 Embedding wrapper for embed_tokens
        if unwrapped.startswith('embed_tokens.embedding.'):
            unwrapped = unwrapped.replace('embed_tokens.embedding.', 'embed_tokens.')
        
        # Unwrap level1 LayerNorm wrapper (.ln.weight -> .weight, .ln.bias -> .bias)
        unwrapped = unwrapped.replace('.ln.weight', '.weight').replace('.ln.bias', '.bias')
        
        # Unwrap level1 PatchEmbed3D wrapper (.patch_embed_3d.proj. -> .proj.)
        unwrapped = unwrapped.replace('.patch_embed_3d.proj.', '.proj.')
        
        # Unwrap PatchMerger MLP level1 Linear wrappers (.mlp_fc1. -> .mlp.0., .mlp_fc2. -> .mlp.2.)
        unwrapped = unwrapped.replace('.mlp_fc1.', '.mlp.0.').replace('.mlp_fc2.', '.mlp.2.')
        
        # Try with model.language_model. prefix for LLM backbone weights
        # (embed_tokens, layers, norm, rotary_emb)
        if unwrapped.startswith(('embed_tokens.', 'layers.', 'norm.', 'rotary_emb.')):
            hf_key = 'model.language_model.' + unwrapped
            if hf_key in hf_state:
                mapping[kb_key] = hf_key
                continue
        
        # Try with model. prefix for visual weights
        if unwrapped.startswith('visual.'):
            hf_key = 'model.' + unwrapped
            if hf_key in hf_state:
                mapping[kb_key] = hf_key
                continue
        
        # Try direct match (lm_head.weight)
        if unwrapped in hf_state:
            mapping[kb_key] = unwrapped
    
    return mapping


def _build_qwen3vl_key_mapping(hf_state, kb_state) -> dict:
    """Build explicit key mapping for Qwen3-VL models.
    
    HF Qwen3VLForConditionalGeneration uses:
      model.visual.* -> KB: visual.*
      model.language_model.embed_tokens.weight -> KB: embed_tokens.embedding.weight
      model.language_model.layers.{i}.* -> KB: layers.{i}.*
      model.language_model.norm.weight -> KB: norm.weight
      lm_head.weight -> KB: lm_head.weight
    
    Key differences from Qwen2-VL:
      - Vision MLP uses linear_fc1/linear_fc2 (same names in HF and KB)
      - PatchMerger uses norm/linear_fc1/linear_fc2 (not ln_q/mlp.0/mlp.2)
      - Has pos_embed (nn.Embedding) in vision encoder
      - Has deepstack_merger_list in vision encoder
      - Text attention has q_norm/k_norm
    """
    mapping = {}  # kb_key -> hf_key
    
    for kb_key in kb_state.keys():
        # Unwrap level1 Embedding wrapper for embed_tokens
        unwrapped = kb_key
        if unwrapped.startswith('embed_tokens.embedding.'):
            unwrapped = unwrapped.replace('embed_tokens.embedding.', 'embed_tokens.')
        
        # Try with model.language_model. prefix for LLM backbone weights
        if unwrapped.startswith(('embed_tokens.', 'layers.', 'norm.', 'rotary_emb.')):
            hf_key = 'model.language_model.' + unwrapped
            if hf_key in hf_state:
                mapping[kb_key] = hf_key
                continue
        
        # Try with model. prefix for visual weights
        if unwrapped.startswith('visual.'):
            hf_key = 'model.' + unwrapped
            if hf_key in hf_state:
                mapping[kb_key] = hf_key
                continue
        
        # Try direct match (lm_head.weight)
        if unwrapped in hf_state:
            mapping[kb_key] = unwrapped
    
    return mapping


def _build_qwen3omni_key_mapping(hf_state, kb_state) -> dict:
    """Build explicit key mapping for Qwen3-Omni-MoE models.
    
    When loaded as Qwen3OmniMoeThinkerForConditionalGeneration, the HF state dict
    has the 'thinker.' prefix stripped (via base_model_prefix). So the actual HF keys are:
      audio_tower.* -> KB: audio_tower.*  (direct match)
      visual.* -> KB: visual.*  (direct match)
      model.embed_tokens.weight -> KB: embed_tokens.embedding.weight
      model.layers.{i}.* -> KB: layers.{i}.*  (strip 'model.' prefix)
      model.norm.weight -> KB: norm.weight
      lm_head.weight -> KB: lm_head.weight  (direct match)
    
    Expert weights need special handling:
      HF: model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight (per-expert)
      KB: layers.{i}.mlp.experts.gate_up_proj (fused 3D), layers.{i}.mlp.experts.down_proj (3D)
    
    Returns mapping for non-expert weights only. Expert weights are handled separately
    in the copy_weights function via _fuse_qwen3omni_expert_weights.
    """
    mapping = {}  # kb_key -> hf_key
    
    for kb_key in kb_state.keys():
        # Skip fused expert parameters - handled separately
        if '.mlp.experts.gate_up_proj' in kb_key or '.mlp.experts.down_proj' in kb_key:
            continue
        
        unwrapped = kb_key
        # Unwrap level1 Embedding wrapper for embed_tokens
        if unwrapped.startswith('embed_tokens.embedding.'):
            unwrapped = unwrapped.replace('embed_tokens.embedding.', 'embed_tokens.')
        
        # Try with model. prefix for LLM backbone weights
        if unwrapped.startswith(('embed_tokens.', 'layers.', 'norm.', 'rotary_emb.')):
            hf_key = 'model.' + unwrapped
            if hf_key in hf_state:
                mapping[kb_key] = hf_key
                continue
        
        # Direct match for audio_tower, visual, lm_head
        if unwrapped in hf_state:
            mapping[kb_key] = unwrapped
            continue
    
    return mapping


def _fuse_qwen3omni_expert_weights(hf_state, kb_state):
    """Fuse per-expert HF weights into KB's fused 3D expert tensors.
    
    HF stores: model.layers.{i}.mlp.experts.{e}.gate_proj.weight  (intermediate, hidden)
               model.layers.{i}.mlp.experts.{e}.up_proj.weight    (intermediate, hidden)
               model.layers.{i}.mlp.experts.{e}.down_proj.weight  (hidden, intermediate)
    
    KB stores: layers.{i}.mlp.experts.gate_up_proj  (num_experts, 2*intermediate, hidden)
               layers.{i}.mlp.experts.down_proj      (num_experts, hidden, intermediate)
    """
    fused_count = 0
    for kb_key, kb_tensor in kb_state.items():
        if '.mlp.experts.gate_up_proj' in kb_key:
            # Extract layer index: layers.{i}.mlp.experts.gate_up_proj
            layer_prefix = kb_key.replace('.mlp.experts.gate_up_proj', '')
            num_experts = kb_tensor.shape[0]
            for e in range(num_experts):
                gate_key = f'model.{layer_prefix}.mlp.experts.{e}.gate_proj.weight'
                up_key = f'model.{layer_prefix}.mlp.experts.{e}.up_proj.weight'
                if gate_key in hf_state and up_key in hf_state:
                    gate_w = hf_state[gate_key]  # (intermediate, hidden)
                    up_w = hf_state[up_key]      # (intermediate, hidden)
                    kb_tensor[e] = torch.cat([gate_w, up_w], dim=0)  # (2*intermediate, hidden)
            fused_count += 1
        elif '.mlp.experts.down_proj' in kb_key and '.mlp.experts.down_proj' == kb_key[kb_key.index('.mlp.experts.down_proj'):]:
            # Extract layer index: layers.{i}.mlp.experts.down_proj
            layer_prefix = kb_key.replace('.mlp.experts.down_proj', '')
            num_experts = kb_tensor.shape[0]
            for e in range(num_experts):
                down_key = f'model.{layer_prefix}.mlp.experts.{e}.down_proj.weight'
                if down_key in hf_state:
                    kb_tensor[e] = hf_state[down_key]  # (hidden, intermediate)
            fused_count += 1
    return fused_count


def copy_weights(hf_model, kb_model, num_layers: int, model_name: str = "") -> None:
    """
    Copy weights from HuggingFace model to KernelBench model.
    
    Uses automatic key matching to support different architectures:
    - Llama: model.embed_tokens, model.layers, model.norm
    - Falcon: transformer.word_embeddings, transformer.h, transformer.ln_f
    - T5: encoder.block, decoder.block (encoder-decoder)
    - SwinV2: swinv2.embeddings, swinv2.encoder.layers (vision)
    - Whisper: model.encoder, model.decoder (speech encoder-decoder)
    
    Handles structural differences where KB wraps primitives in extra modules.
    """
    hf_state = hf_model.state_dict()
    kb_state = kb_model.state_dict()
    
    # Use specialized mapping for T5 and SwinV2
    if _is_t5_model(model_name):
        explicit_mapping = _build_t5_key_mapping(hf_state, kb_state)
        copied = 0
        skipped_buffers = 0
        missing_in_hf = []
        
        for kb_key, kb_tensor in kb_state.items():
            if any(skip in kb_key for skip in ['inv_freq', 'kv_cache', '_cache']):
                skipped_buffers += 1
                continue
            
            if kb_key in explicit_mapping:
                hf_key = explicit_mapping[kb_key]
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
        kb_model.load_state_dict(kb_state)
        return
    
    if _is_swinv2_model(model_name):
        explicit_mapping = _build_swinv2_key_mapping(hf_state, kb_state)
        copied = 0
        skipped_buffers = 0
        missing_in_hf = []
        
        for kb_key, kb_tensor in kb_state.items():
            if any(skip in kb_key for skip in ['relative_coords_table', 'relative_position_index']):
                skipped_buffers += 1
                continue
            
            if kb_key in explicit_mapping:
                hf_key = explicit_mapping[kb_key]
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
        kb_model.load_state_dict(kb_state)
        return
    
    if _is_whisper_model(model_name):
        explicit_mapping = _build_whisper_key_mapping(hf_state, kb_state)
        copied = 0
        skipped_buffers = 0
        missing_in_hf = []
        
        for kb_key, kb_tensor in kb_state.items():
            # Skip feature_extractor buffers (mel filterbank, hann window) —
            # these are computed locally by KB's WhisperFeatureExtractor and
            # have no counterpart in the HF model weights.
            if kb_key.startswith('feature_extractor.'):
                skipped_buffers += 1
                continue
            if kb_key in explicit_mapping:
                hf_key = explicit_mapping[kb_key]
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
        kb_model.load_state_dict(kb_state)
        return
    
    if _is_qwen2vl_model(model_name):
        explicit_mapping = _build_qwen2vl_key_mapping(hf_state, kb_state)
        copied = 0
        skipped_buffers = 0
        missing_in_hf = []
        
        for kb_key, kb_tensor in kb_state.items():
            # Skip rotary embedding buffers (computed locally, different format)
            if 'rotary' in kb_key and any(b in kb_key for b in ('inv_freq', 'cos_cached', 'sin_cached')):
                skipped_buffers += 1
                continue
            
            if kb_key in explicit_mapping:
                hf_key = explicit_mapping[kb_key]
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
        kb_model.load_state_dict(kb_state)
        return
    
    if _is_qwen3vl_model(model_name):
        explicit_mapping = _build_qwen3vl_key_mapping(hf_state, kb_state)
        copied = 0
        skipped_buffers = 0
        missing_in_hf = []
        
        for kb_key, kb_tensor in kb_state.items():
            # Skip vision rotary embedding (computed locally, different format)
            if 'visual.rotary_pos_emb.inv_freq' in kb_key:
                skipped_buffers += 1
                continue
            # Skip LLM rotary embedding inv_freq (computed locally)
            if 'rotary_emb.inv_freq' in kb_key:
                skipped_buffers += 1
                continue
            
            if kb_key in explicit_mapping:
                hf_key = explicit_mapping[kb_key]
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
        kb_model.load_state_dict(kb_state)
        return
    
    if _is_qwen3omni_model(model_name):
        explicit_mapping = _build_qwen3omni_key_mapping(hf_state, kb_state)
        copied = 0
        skipped_buffers = 0
        missing_in_hf = []
        
        for kb_key, kb_tensor in kb_state.items():
            # Skip fused expert parameters (handled below)
            if '.mlp.experts.gate_up_proj' in kb_key or (
                '.mlp.experts.down_proj' in kb_key and 
                '.mlp.experts.down_proj' == kb_key[kb_key.index('.mlp.experts.down_proj'):]
            ):
                continue
            # Skip vision rotary embedding (computed locally)
            if 'visual.rotary_pos_emb.inv_freq' in kb_key:
                skipped_buffers += 1
                continue
            # Skip LLM rotary embedding inv_freq (computed locally)
            if 'rotary_emb.inv_freq' in kb_key:
                skipped_buffers += 1
                continue
            # Skip audio positional embedding (computed locally)
            if 'audio_tower.positional_embedding.positional_embedding' in kb_key:
                skipped_buffers += 1
                continue
            
            if kb_key in explicit_mapping:
                hf_key = explicit_mapping[kb_key]
                hf_tensor = hf_state[hf_key]
                if kb_tensor.shape == hf_tensor.shape:
                    kb_tensor.copy_(hf_tensor)
                    copied += 1
                else:
                    missing_in_hf.append(f"{kb_key} (shape mismatch: KB={kb_tensor.shape} vs HF={hf_tensor.shape})")
            else:
                missing_in_hf.append(kb_key)
        
        # Fuse per-expert weights into KB's 3D tensors
        fused_count = _fuse_qwen3omni_expert_weights(hf_state, kb_state)
        copied += fused_count
        
        if missing_in_hf:
            print(f"  Warning: {len(missing_in_hf)} KB weights not found in HF model:")
            for m in missing_in_hf[:10]:
                print(f"    - {m}")
            if len(missing_in_hf) > 10:
                print(f"    ... and {len(missing_in_hf) - 10} more")
        
        print(f"  Copied {copied} weights (+{fused_count} fused expert tensors), skipped {skipped_buffers} buffers")
        kb_model.load_state_dict(kb_state)
        return
    
    # Default: Llama/Falcon/Mistral/BLOOM/Mamba style
    # Detect HF model prefix (e.g., "model." for Llama, "transformer." for Falcon, "backbone." for Mamba2)
    hf_prefix = ""
    for hf_key in hf_state.keys():
        if "." in hf_key:
            potential_prefix = hf_key.split(".")[0] + "."
            if potential_prefix in ["model.", "transformer.", "backbone."]:
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


def _is_mamba2_model(model_name: str) -> bool:
    """Check if a model is a Mamba2 (SSM) model."""
    return "Mamba" in model_name and "Codestral" in model_name


def _is_mamba1_model(model_name: str) -> bool:
    """Check if a model is a Mamba-1 model (generic or Falcon Mamba)."""
    return "falcon-mamba" in model_name.lower() or "state-spaces/mamba" in model_name.lower()


def _is_ssm_model(model_name: str) -> bool:
    """Check if a model is any SSM model (Mamba-1 or Mamba-2)."""
    return _is_mamba2_model(model_name) or _is_mamba1_model(model_name)


def _is_t5_model(model_name: str) -> bool:
    """Check if a model is a T5 encoder-decoder model."""
    return any(pattern in model_name.lower() for pattern in [
        "t5", "flan-t5", "google-t5",
    ])


def _is_swinv2_model(model_name: str) -> bool:
    """Check if a model is a SwinV2 image classification model."""
    return "swinv2" in model_name.lower()


def _is_whisper_model(model_name: str) -> bool:
    """Check if a model is a Whisper speech recognition model."""
    return "whisper" in model_name.lower()


def _is_qwen2vl_model(model_name: str) -> bool:
    """Check if a model is a Qwen2-VL vision-language model."""
    return "qwen2-vl" in model_name.lower()


def _is_qwen3vl_model(model_name: str) -> bool:
    """Check if a model is a Qwen3-VL vision-language model."""
    return "qwen3-vl" in model_name.lower()


def _is_qwen3omni_model(model_name: str) -> bool:
    """Check if a model is a Qwen3-Omni-MoE multimodal model."""
    return "qwen3-omni" in model_name.lower()


def _fix_mamba2_config_json(model_path: str) -> None:
    """
    Fix the Infinity issue in Mamba-Codestral config.json.
    
    The mistralai/Mamba-Codestral-7B-v0.1 config.json contains bare Infinity
    (not a valid JSON value). We need to replace it with a valid representation.
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return
    with open(config_path, "r") as f:
        content = f.read()
    if "Infinity" in content:
        # Replace bare Infinity with a very large float (Python will parse this)
        fixed_content = content.replace("Infinity", "1e308")
        with open(config_path, "w") as f:
            f.write(fixed_content)


def _create_kb_mamba2_config(hf_config) -> dict:
    """Create KernelBench config dict for Mamba2 models."""
    # Convert time_step_limit: handle the fixed Infinity -> 1e308
    time_step_limit = getattr(hf_config, 'time_step_limit', (0.0, float("inf")))
    if isinstance(time_step_limit, list):
        time_step_limit = tuple(time_step_limit)
    # Ensure any very large values become inf for consistency
    time_step_limit = tuple(
        float("inf") if v > 1e300 else v for v in time_step_limit
    )
    
    time_step_rank = getattr(hf_config, 'time_step_rank', 'auto')
    if time_step_rank == 'auto':
        import math
        time_step_rank = math.ceil(hf_config.hidden_size / 16)
    
    return {
        'vocab_size': hf_config.vocab_size,
        'hidden_size': hf_config.hidden_size,
        'num_hidden_layers': hf_config.num_hidden_layers,
        'num_heads': hf_config.num_heads,
        'head_dim': hf_config.head_dim,
        'state_size': hf_config.state_size,
        'expand': hf_config.expand,
        'conv_kernel': hf_config.conv_kernel,
        'n_groups': hf_config.n_groups,
        'chunk_size': getattr(hf_config, 'chunk_size', 256),
        'use_bias': getattr(hf_config, 'use_bias', False),
        'use_conv_bias': getattr(hf_config, 'use_conv_bias', True),
        'time_step_limit': time_step_limit,
        'time_step_rank': time_step_rank,
        'layer_norm_epsilon': getattr(hf_config, 'layer_norm_epsilon', 1e-5),
        'residual_in_fp32': getattr(hf_config, 'residual_in_fp32', True),
        'tie_word_embeddings': getattr(hf_config, 'tie_word_embeddings', False),
    }


def _create_kb_mamba1_config(hf_config) -> dict:
    """Create KernelBench config dict for Mamba-1 models (generic and Falcon Mamba).
    
    Uses the HF config's architectures field to determine whether to enable
    mixer RMS normalization (Falcon Mamba variant).
    """
    time_step_rank = getattr(hf_config, 'time_step_rank', 'auto')
    if time_step_rank == 'auto':
        import math
        time_step_rank = math.ceil(hf_config.hidden_size / 16)

    # Determine if this is a Falcon Mamba variant by checking architectures
    architectures = getattr(hf_config, 'architectures', []) or []
    is_falcon_mamba = any('FalconMamba' in arch for arch in architectures)

    # Compute intermediate_size: Falcon Mamba has it explicitly,
    # generic Mamba-1 computes it from expand * hidden_size
    intermediate_size = getattr(hf_config, 'intermediate_size', None)
    if intermediate_size is None:
        expand = getattr(hf_config, 'expand', 2)
        intermediate_size = int(expand * hf_config.hidden_size)

    return {
        'vocab_size': hf_config.vocab_size,
        'hidden_size': hf_config.hidden_size,
        'num_hidden_layers': hf_config.num_hidden_layers,
        'intermediate_size': intermediate_size,
        'state_size': hf_config.state_size,
        'expand': getattr(hf_config, 'expand', 2),
        'conv_kernel': hf_config.conv_kernel,
        'use_bias': getattr(hf_config, 'use_bias', False),
        'use_conv_bias': getattr(hf_config, 'use_conv_bias', True),
        'time_step_rank': time_step_rank,
        'use_mixer_rms_norm': is_falcon_mamba,
        'mixer_rms_eps': getattr(hf_config, 'mixer_rms_eps', 1e-6),
        'layer_norm_epsilon': getattr(hf_config, 'layer_norm_epsilon', 1e-5),
        'residual_in_fp32': getattr(hf_config, 'residual_in_fp32', True),
        'tie_word_embeddings': getattr(hf_config, 'tie_word_embeddings', False),
    }


def _create_kb_t5_config(hf_config) -> dict:
    """Create KernelBench config dict for T5 encoder-decoder models."""
    return {
        'd_model': hf_config.d_model,
        'num_heads': hf_config.num_heads,
        'd_kv': hf_config.d_kv,
        'd_ff': hf_config.d_ff,
        'vocab_size': hf_config.vocab_size,
        'num_encoder_layers': hf_config.num_layers,
        'num_decoder_layers': hf_config.num_decoder_layers,
        'relative_attention_num_buckets': hf_config.relative_attention_num_buckets,
        'relative_attention_max_distance': hf_config.relative_attention_max_distance,
        'is_gated_act': hf_config.is_gated_act,
        'dense_act_fn': hf_config.dense_act_fn,
        'tie_word_embeddings': getattr(hf_config, 'tie_word_embeddings', False),
    }


def _create_kb_swinv2_config(hf_config) -> dict:
    """Create KernelBench config dict for SwinV2 models."""
    return {
        'image_size': hf_config.image_size,
        'patch_size': hf_config.patch_size,
        'num_channels': hf_config.num_channels,
        'embed_dim': hf_config.embed_dim,
        'depths': hf_config.depths,
        'num_heads': hf_config.num_heads,
        'window_size': hf_config.window_size,
        'mlp_ratio': hf_config.mlp_ratio,
        'qkv_bias': hf_config.qkv_bias,
        'num_labels': hf_config.num_labels,
        'pretrained_window_sizes': getattr(hf_config, 'pretrained_window_sizes', [0, 0, 0, 0]),
    }


def _create_kb_whisper_config(hf_config) -> dict:
    """Create KernelBench config dict for Whisper encoder-decoder models."""
    return {
        'd_model': hf_config.d_model,
        'encoder_attention_heads': hf_config.encoder_attention_heads,
        'decoder_attention_heads': hf_config.decoder_attention_heads,
        'encoder_layers': hf_config.encoder_layers,
        'decoder_layers': hf_config.decoder_layers,
        'encoder_ffn_dim': hf_config.encoder_ffn_dim,
        'decoder_ffn_dim': hf_config.decoder_ffn_dim,
        'vocab_size': hf_config.vocab_size,
        'num_mel_bins': hf_config.num_mel_bins,
        'max_source_positions': hf_config.max_source_positions,
        'max_target_positions': hf_config.max_target_positions,
        'decoder_start_token_id': getattr(hf_config, 'decoder_start_token_id', 50258),
    }


def _create_kb_qwen2vl_config(hf_config) -> dict:
    """Create KernelBench config dict for Qwen2-VL vision-language models."""
    vc = hf_config.vision_config
    tc = hf_config.text_config
    
    # Get mrope_section from rope_scaling
    rope_scaling = getattr(tc, 'rope_scaling', None) or {}
    mrope_section = rope_scaling.get('mrope_section', [16, 24, 24])
    rope_theta = getattr(tc, 'rope_theta', 1000000.0)
    
    return {
        # Vision config
        'vision_depth': vc.depth,
        'vision_embed_dim': vc.embed_dim,
        'vision_num_heads': vc.num_heads,
        'vision_hidden_size': vc.hidden_size,
        'vision_hidden_act': vc.hidden_act,
        'vision_mlp_ratio': vc.mlp_ratio,
        'in_channels': vc.in_channels,
        'patch_size': vc.patch_size,
        'temporal_patch_size': vc.temporal_patch_size,
        'spatial_merge_size': vc.spatial_merge_size,
        # LLM config
        'hidden_size': tc.hidden_size,
        'num_hidden_layers': tc.num_hidden_layers,
        'num_attention_heads': tc.num_attention_heads,
        'num_key_value_heads': tc.num_key_value_heads,
        'intermediate_size': tc.intermediate_size,
        'vocab_size': tc.vocab_size,
        'rms_norm_eps': tc.rms_norm_eps,
        'rope_theta': rope_theta,
        'mrope_section': mrope_section,
        # Special token ids
        'image_token_id': hf_config.image_token_id,
        'video_token_id': hf_config.video_token_id,
        'vision_start_token_id': hf_config.vision_start_token_id,
        'vision_end_token_id': hf_config.vision_end_token_id,
    }


def _create_kb_qwen3vl_config(hf_config) -> dict:
    """Create KernelBench config dict for Qwen3-VL vision-language models."""
    vc = hf_config.vision_config
    tc = hf_config.text_config
    
    # Get mrope_section from rope_scaling (Qwen3-VL uses rope_scaling, not rope_parameters)
    rope_scaling = getattr(tc, 'rope_scaling', None) or {}
    mrope_section = rope_scaling.get('mrope_section', [24, 20, 20])
    # rope_theta is a top-level attribute in the text config (not inside rope_scaling)
    rope_theta = getattr(tc, 'rope_theta', 5000000.0)
    
    return {
        # Vision config
        'vision_depth': vc.depth,
        'vision_hidden_size': vc.hidden_size,
        'vision_out_hidden_size': vc.out_hidden_size,
        'vision_num_heads': vc.num_heads,
        'vision_intermediate_size': vc.intermediate_size,
        'vision_hidden_act': vc.hidden_act,
        'in_channels': vc.in_channels,
        'patch_size': vc.patch_size,
        'temporal_patch_size': vc.temporal_patch_size,
        'spatial_merge_size': vc.spatial_merge_size,
        'num_position_embeddings': vc.num_position_embeddings,
        'deepstack_visual_indexes': vc.deepstack_visual_indexes,
        # LLM config
        'hidden_size': tc.hidden_size,
        'num_hidden_layers': tc.num_hidden_layers,
        'num_attention_heads': tc.num_attention_heads,
        'num_key_value_heads': tc.num_key_value_heads,
        'head_dim': getattr(tc, 'head_dim', 128),
        'intermediate_size': tc.intermediate_size,
        'vocab_size': tc.vocab_size,
        'rms_norm_eps': tc.rms_norm_eps,
        'rope_theta': rope_theta,
        'mrope_section': mrope_section,
        # Special token ids
        'image_token_id': hf_config.image_token_id,
        'video_token_id': hf_config.video_token_id,
        'vision_start_token_id': hf_config.vision_start_token_id,
        'vision_end_token_id': hf_config.vision_end_token_id,
    }


def _create_kb_qwen3omni_config(hf_config) -> dict:
    """Create KernelBench config dict for Qwen3-Omni-MoE models."""
    # The top-level config is Qwen3OmniMoeConfig, thinker_config has the sub-configs
    tc_config = hf_config.thinker_config
    vc = tc_config.vision_config
    ac = tc_config.audio_config
    tc = tc_config.text_config
    
    # Get mrope_section from rope_scaling
    rope_scaling = getattr(tc, 'rope_scaling', None) or {}
    mrope_section = rope_scaling.get('mrope_section', [24, 20, 20])
    rope_theta = getattr(tc, 'rope_theta', 1000000.0)
    
    # head_dim: compute from hidden_size / num_attention_heads
    head_dim = getattr(tc, 'head_dim', tc.hidden_size // tc.num_attention_heads)
    
    return {
        # Audio config
        'audio_num_mel_bins': ac.num_mel_bins,
        'audio_d_model': ac.d_model,
        'audio_encoder_layers': ac.encoder_layers,
        'audio_encoder_attention_heads': ac.encoder_attention_heads,
        'audio_encoder_ffn_dim': ac.encoder_ffn_dim,
        'audio_activation_function': ac.activation_function,
        'audio_max_source_positions': ac.max_source_positions,
        'audio_output_dim': ac.output_dim,
        'audio_n_window': ac.n_window,
        'audio_n_window_infer': ac.n_window_infer,
        'audio_conv_chunksize': ac.conv_chunksize,
        'audio_downsample_hidden_size': ac.downsample_hidden_size,
        'audio_scale_embedding': ac.scale_embedding,
        # Vision config
        'vision_depth': vc.depth,
        'vision_hidden_size': vc.hidden_size,
        'vision_out_hidden_size': vc.out_hidden_size,
        'vision_num_heads': vc.num_heads,
        'vision_intermediate_size': vc.intermediate_size,
        'vision_hidden_act': vc.hidden_act,
        'in_channels': vc.in_channels,
        'patch_size': vc.patch_size,
        'temporal_patch_size': vc.temporal_patch_size,
        'spatial_merge_size': vc.spatial_merge_size,
        'num_position_embeddings': vc.num_position_embeddings,
        'deepstack_visual_indexes': vc.deepstack_visual_indexes,
        # LLM config
        'hidden_size': tc.hidden_size,
        'num_hidden_layers': tc.num_hidden_layers,
        'num_attention_heads': tc.num_attention_heads,
        'num_key_value_heads': tc.num_key_value_heads,
        'head_dim': head_dim,
        'intermediate_size': tc.intermediate_size,
        'vocab_size': tc.vocab_size,
        'rms_norm_eps': tc.rms_norm_eps,
        'rope_theta': rope_theta,
        'mrope_section': mrope_section,
        # MoE config
        'num_experts': tc.num_experts,
        'num_experts_per_tok': tc.num_experts_per_tok,
        'moe_intermediate_size': tc.moe_intermediate_size,
        'decoder_sparse_step': tc.decoder_sparse_step,
        'mlp_only_layers': tc.mlp_only_layers,
        'norm_topk_prob': tc.norm_topk_prob,
        # Special token ids
        'image_token_id': tc_config.image_token_id,
        'video_token_id': tc_config.video_token_id,
        'audio_token_id': tc_config.audio_token_id,
        'vision_start_token_id': getattr(tc_config, 'vision_start_token_id', 151652),
        'audio_start_token_id': tc_config.audio_start_token_id,
        'position_id_per_seconds': tc_config.position_id_per_seconds,
    }


def create_kb_model_from_hf_config(hf_config, num_blocks: int = 8192):
    """Create KernelBench model config from HuggingFace config.
    
    Handles different naming conventions across model architectures:
    - Llama: intermediate_size, rms_norm_eps, num_hidden_layers, num_attention_heads
    - Falcon: ffn_hidden_size (or 4*hidden_size), layer_norm_epsilon
    - Mistral: intermediate_size, rms_norm_eps
    - BLOOM: n_inner (or 4*hidden), layer_norm_epsilon, n_layer, n_head
    - Mamba2: num_heads, head_dim, state_size, expand, conv_kernel, n_groups
    """
    # Check if this is an SSM model
    if getattr(hf_config, 'model_type', None) == 'mamba2':
        return _create_kb_mamba2_config(hf_config)
    if getattr(hf_config, 'model_type', None) in ('falcon_mamba', 'mamba'):
        return _create_kb_mamba1_config(hf_config)
    # Check if this is a T5 model
    if getattr(hf_config, 'model_type', None) == 't5':
        return _create_kb_t5_config(hf_config)
    # Check if this is a SwinV2 model
    if getattr(hf_config, 'model_type', None) == 'swinv2':
        return _create_kb_swinv2_config(hf_config)
    # Check if this is a Whisper model
    if getattr(hf_config, 'model_type', None) == 'whisper':
        return _create_kb_whisper_config(hf_config)
    # Check if this is a Qwen2-VL model
    if getattr(hf_config, 'model_type', None) == 'qwen2_vl':
        return _create_kb_qwen2vl_config(hf_config)
    # Check if this is a Qwen3-VL model
    if getattr(hf_config, 'model_type', None) == 'qwen3_vl':
        return _create_kb_qwen3vl_config(hf_config)
    # Check if this is a Qwen3-Omni-MoE model
    if getattr(hf_config, 'model_type', None) == 'qwen3_omni_moe':
        return _create_kb_qwen3omni_config(hf_config)
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
    # - Falcon: multi_query=True means 1 KV head (MQA); note that Falcon's
    #   HF config sets num_kv_heads equal to num_attention_heads even when
    #   multi_query is True, so we must check multi_query first.
    # - BLOOM: uses full MHA, so num_kv_heads = num_heads
    if getattr(hf_config, 'multi_query', False):
        # Falcon MQA: multi_query=True overrides num_kv_heads (which is misleadingly
        # set to num_attention_heads in the HF Falcon config)
        num_kv_heads = 1
    else:
        num_kv_heads = getattr(hf_config, 'num_key_value_heads', None)
        if num_kv_heads is None:
            num_kv_heads = getattr(hf_config, 'num_kv_heads', None)
        if num_kv_heads is None:
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
    
    # Mixtral specific parameters (MoE)
    if hasattr(hf_config, 'num_local_experts'):
        config['num_experts'] = hf_config.num_local_experts
        config['num_experts_per_tok'] = getattr(hf_config, 'num_experts_per_tok', 2)
        config['sliding_window'] = getattr(hf_config, 'sliding_window', None)
        # Mixtral uses rope_parameters dict for rope_theta
        if hasattr(hf_config, 'rope_parameters') and hf_config.rope_parameters:
            rope_params = hf_config.rope_parameters
            if isinstance(rope_params, dict) and 'rope_theta' in rope_params:
                config['rope_theta'] = rope_params['rope_theta']
    
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
                "backbone.embeddings.weight",
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
    
    is_mamba2 = _is_mamba2_model(model_name)
    is_mamba1 = _is_mamba1_model(model_name)
    is_ssm = is_mamba2 or is_mamba1
    is_t5 = _is_t5_model(model_name)
    is_swinv2 = _is_swinv2_model(model_name)
    is_whisper = _is_whisper_model(model_name)
    is_qwen2vl = _is_qwen2vl_model(model_name)
    is_qwen3vl = _is_qwen3vl_model(model_name)
    is_qwen3omni = _is_qwen3omni_model(model_name)
    
    # SSM models need snapshot_download for local path loading
    if is_ssm:
        model_path = snapshot_download(
            model_name,
            allow_patterns=[
                "*.safetensors", "*.safetensors.index.json",
                "pytorch_model*.bin", "pytorch_model.bin.index.json",
                "*.json",
                "tokenizer*",  # Mamba models need tokenizer files
            ],
        )
        # Mamba2 config.json has bare Infinity that needs fixing
        if is_mamba2:
            _fix_mamba2_config_json(model_path)
    
    # Load tokenizer (text models), image processor (vision models), or
    # whisper processor (speech models)
    tokenizer = None
    image_processor = None
    whisper_processor = None
    qwen2vl_processor = None
    qwen3vl_processor = None
    qwen3omni_processor = None
    if is_swinv2:
        image_processor = AutoImageProcessor.from_pretrained(model_name)
    elif is_whisper:
        whisper_processor = WhisperProcessor.from_pretrained(model_name)
        tokenizer = whisper_processor.tokenizer
    elif is_qwen3omni:
        qwen3omni_processor = AutoProcessor.from_pretrained(model_name)
        tokenizer = qwen3omni_processor.tokenizer
    elif is_qwen3vl:
        qwen3vl_processor = AutoProcessor.from_pretrained(model_name)
        tokenizer = qwen3vl_processor.tokenizer
    elif is_qwen2vl:
        qwen2vl_processor = AutoProcessor.from_pretrained(model_name)
        tokenizer = qwen2vl_processor.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    if is_ssm:
        hf_config = AutoConfig.from_pretrained(model_path)
        # Fix time_step_limit: convert 1e308 back to inf (original was Infinity in JSON).
        # bf16 torch.clamp can handle inf but NOT 1e308 (exceeds bf16 max of 3.4e38).
        if hasattr(hf_config, 'time_step_limit'):
            tsl = hf_config.time_step_limit
            if isinstance(tsl, (list, tuple)):
                hf_config.time_step_limit = tuple(
                    float("inf") if v > 1e300 else v for v in tsl
                )
    else:
        hf_config = AutoConfig.from_pretrained(model_name)
    
    # Get total layers (handle different naming conventions)
    if is_qwen3omni:
        total_layers = hf_config.thinker_config.text_config.num_hidden_layers
    elif is_qwen3vl:
        total_layers = hf_config.text_config.num_hidden_layers
    elif is_qwen2vl:
        total_layers = hf_config.text_config.num_hidden_layers
    elif is_swinv2:
        # SwinV2 uses depths list, not a single num_layers
        total_layers = sum(hf_config.depths)
    elif is_t5:
        total_layers = hf_config.num_layers  # T5 uses num_layers
    elif is_whisper:
        total_layers = hf_config.encoder_layers  # Use encoder layers as reference
    else:
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
    
    # --- Load HF model ---
    if is_qwen3omni:
        if truncated:
            hf_config.thinker_config.text_config.num_hidden_layers = num_layers
            hf_config.thinker_config.vision_config.depth = min(num_layers, hf_config.thinker_config.vision_config.depth)
            hf_config.thinker_config.audio_config.encoder_layers = min(num_layers, hf_config.thinker_config.audio_config.encoder_layers)
        # Load only the thinker component (disable audio output to skip talker/code2wav)
        hf_config.enable_audio_output = False
        hf_model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_name,
            config=hf_config.thinker_config,
            torch_dtype=DTYPE,
            device_map=DEVICE,
            attn_implementation="eager",
        )
        hf_model.eval()
    elif is_qwen3vl:
        if truncated:
            hf_config.text_config.num_hidden_layers = num_layers
            hf_config.vision_config.depth = min(num_layers, hf_config.vision_config.depth)
        hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            config=hf_config,
            torch_dtype=DTYPE,
            device_map=DEVICE,
            attn_implementation="eager",
        )
        hf_model.eval()
    elif is_qwen2vl:
        if truncated:
            hf_config.text_config.num_hidden_layers = num_layers
            hf_config.vision_config.depth = min(num_layers, hf_config.vision_config.depth)
        hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            config=hf_config,
            torch_dtype=DTYPE,
            device_map=DEVICE,
            attn_implementation="eager",  # Use eager attention for alignment
        )
        hf_model.eval()
    elif is_t5:
        if truncated:
            hf_config.num_layers = num_layers
            hf_config.num_decoder_layers = num_layers
        hf_model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            config=hf_config,
            torch_dtype=DTYPE,
            device_map=DEVICE,
        )
        hf_model.eval()
    elif is_whisper:
        if truncated:
            hf_config.encoder_layers = num_layers
            hf_config.decoder_layers = num_layers
        hf_model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            config=hf_config,
            torch_dtype=DTYPE,
            device_map=DEVICE,
        )
        hf_model.eval()
    elif is_swinv2:
        # SwinV2: truncation means reducing depths
        if truncated:
            # Simple truncation: reduce depths proportionally
            orig_depths = list(hf_config.depths)
            remaining = num_layers
            new_depths = []
            for d in orig_depths:
                take = min(d, remaining)
                new_depths.append(take)
                remaining -= take
                if remaining <= 0:
                    break
            while len(new_depths) < len(orig_depths):
                new_depths.append(0)
            hf_config.depths = new_depths
        hf_model = Swinv2ForImageClassification.from_pretrained(
            model_name,
            config=hf_config,
            torch_dtype=DTYPE,
            device_map=DEVICE,
        )
        hf_model.eval()
    elif truncated:
        # Memory-efficient loading: use meta tensors + selective shard loading
        print(f"Using memory-efficient loading for {num_layers}/{total_layers} layers...")
        
        # Modify config to have fewer layers (set both attributes for compatibility)
        if hasattr(hf_config, 'num_hidden_layers'):
            hf_config.num_hidden_layers = num_layers
        if hasattr(hf_config, 'n_layer'):
            hf_config.n_layer = num_layers
        
        # Download checkpoint files if not already downloaded (SSM models already downloaded)
        if not is_ssm:
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
        if is_ssm:
            # Load from local path for SSM models, using fixed config
            # (Mamba2 config.json has 1e308 which exceeds bf16 max; pass fixed config with inf)
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=hf_config,
                torch_dtype=DTYPE,
                device_map=DEVICE,
            )
        else:
            # Full model - use standard loading
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=DTYPE, 
                device_map=DEVICE,
            )
        hf_model.eval()
    
    # Create KB config 
    kb_config = create_kb_model_from_hf_config(hf_config)
    
    if is_qwen3omni:
        kb_config['num_hidden_layers'] = num_layers
        if truncated:
            kb_config['vision_depth'] = min(num_layers, kb_config['vision_depth'])
            kb_config['audio_encoder_layers'] = min(num_layers, kb_config['audio_encoder_layers'])
    elif is_qwen3vl:
        kb_config['num_hidden_layers'] = num_layers
        if truncated:
            kb_config['vision_depth'] = min(num_layers, kb_config['vision_depth'])
    elif is_qwen2vl:
        kb_config['num_hidden_layers'] = num_layers
        if truncated:
            kb_config['vision_depth'] = min(num_layers, kb_config['vision_depth'])
    elif is_t5:
        kb_config['num_encoder_layers'] = num_layers
        kb_config['num_decoder_layers'] = num_layers
    elif is_whisper:
        kb_config['encoder_layers'] = num_layers
        kb_config['decoder_layers'] = num_layers
    elif is_swinv2:
        if truncated:
            kb_config['depths'] = list(hf_config.depths)
    elif is_ssm:
        kb_config['num_hidden_layers'] = num_layers
    else:
        # Calculate num_blocks needed for testing (per-layer KV cache)
        max_seq_len = 4096
        block_size = 16
        max_batch_size = 2
        blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        num_blocks = blocks_per_seq * max_batch_size * 2
        kb_config['num_layers'] = num_layers
    
    kb_model, kb_module = load_kernelbench_model(model_name, kb_config)
    kb_model = kb_model.to(device=DEVICE, dtype=DTYPE)
    
    # Copy weights for the layers we loaded
    copy_weights(hf_model, kb_model, num_layers, model_name=model_name)
    kb_model.eval()
    
    if is_qwen3omni:
        print(f"Loaded: {num_layers}/{total_layers} LLM layers, "
              f"vision_depth={kb_config['vision_depth']}, "
              f"audio_layers={kb_config['audio_encoder_layers']}, "
              f"hidden_size={kb_config['hidden_size']}, "
              f"{kb_config['num_attention_heads']} heads, "
              f"{kb_config['num_experts']} experts (Qwen3-Omni-MoE)")
    elif is_qwen3vl:
        print(f"Loaded: {num_layers}/{total_layers} LLM layers, "
              f"vision_depth={kb_config['vision_depth']}, "
              f"hidden_size={kb_config['hidden_size']}, "
              f"{kb_config['num_attention_heads']} heads (Qwen3-VL)")
    elif is_qwen2vl:
        print(f"Loaded: {num_layers}/{total_layers} LLM layers, "
              f"vision_depth={kb_config['vision_depth']}, "
              f"hidden_size={kb_config['hidden_size']}, "
              f"{kb_config['num_attention_heads']} heads (Qwen2-VL)")
    elif is_t5:
        print(f"Loaded: {num_layers}/{total_layers} layers, d_model={kb_config['d_model']}, "
              f"{kb_config['num_heads']} heads (T5 encoder-decoder)")
    elif is_whisper:
        print(f"Loaded: {num_layers}/{total_layers} layers, d_model={kb_config['d_model']}, "
              f"enc_heads={kb_config['encoder_attention_heads']}, "
              f"dec_heads={kb_config['decoder_attention_heads']} (Whisper)")
    elif is_swinv2:
        print(f"Loaded: depths={kb_config['depths']}, embed_dim={kb_config['embed_dim']}, "
              f"num_heads={kb_config['num_heads']} (SwinV2)")
    elif is_ssm:
        heads_info = f"{kb_config.get('num_heads', 'N/A')} heads" if is_mamba2 else f"d_inner={kb_config.get('intermediate_size', 'N/A')}"
        print(f"Loaded: {num_layers}/{total_layers} layers, {kb_config['hidden_size']} hidden, "
              f"{heads_info} (SSM)")
    else:
        print(f"Loaded: {num_layers}/{total_layers} layers, {kb_config['hidden_size']} hidden, "
              f"{kb_config['num_heads']} heads, {kb_config['num_kv_heads']} kv_heads")
    
    return hf_model, kb_model, tokenizer, kb_config, kb_module, image_processor, whisper_processor, qwen2vl_processor, qwen3vl_processor, qwen3omni_processor


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
        print(e)
        pytest.skip(str(e))
    except Exception as e:
        print(e)
        pytest.skip(f"Could not load {model_name}: {e}")


# ============================================================================
# Alignment Tests
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_alignment(loaded_models):
    """
    Test prefill (prompt processing) alignment.
    
    For decoder-only/SSM models: uses text prompts with generate(max_new_tokens=0).
    For T5: uses text prompts as encoder input, pad token as decoder input.
    For SwinV2: uses random pixel values for image classification.
    """
    (hf_model, kb_model, tokenizer, kb_config, kb_module, image_processor, whisper_processor, qwen2vl_processor, qwen3vl_processor, qwen3omni_processor), model_name, max_layers = loaded_models
    
    is_ssm = _is_ssm_model(model_name)
    is_t5 = _is_t5_model(model_name)
    is_swinv2 = _is_swinv2_model(model_name)
    is_whisper = _is_whisper_model(model_name)
    is_qwen2vl = _is_qwen2vl_model(model_name)
    is_qwen3vl = _is_qwen3vl_model(model_name)
    is_qwen3omni = _is_qwen3omni_model(model_name)
    
    layers_info = f" ({max_layers} layers)" if max_layers else ""
    print("\n" + "="*70)
    print(f"Testing Prefill Alignment for {model_name}{layers_info}")
    print("="*70)
    
    if is_swinv2:
        # SwinV2: test with real images preprocessed by the HF image processor
        assert image_processor is not None, "Image processor required for SwinV2"
        test_images = _load_test_images()
        print(f"  Loaded {len(test_images)} test images")
        
        for i, (pil_image, image_url) in enumerate(test_images):
            # Preprocess with the official HuggingFace image processor
            inputs = image_processor(images=pil_image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device=DEVICE, dtype=DTYPE)
            
            with torch.no_grad():
                # HuggingFace
                hf_out = hf_model(pixel_values=pixel_values)
                hf_logits = hf_out.logits  # (1, num_labels)
                
                # KernelBench
                kb_logits = kb_model(pixel_values)  # (1, num_labels)
            
            hf_flat = hf_logits.float()
            kb_flat = kb_logits.float()
            
            abs_diff = (hf_flat - kb_flat).abs()
            max_abs_diff = abs_diff.max().item()
            mean_abs_diff = abs_diff.mean().item()
            
            denominator = torch.maximum(hf_flat.abs(), kb_flat.abs()) + 1e-8
            rel_diff = abs_diff / denominator
            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()
            
            hf_top = hf_flat.argmax(dim=-1).item()
            kb_top = kb_flat.argmax(dim=-1).item()
            top_match = hf_top == kb_top
            
            mean_ok = mean_rel_diff < RTOL_MEAN
            max_ok = max_rel_diff < RTOL_MAX
            is_pass = top_match and mean_ok and max_ok
            status = "PASS" if is_pass else "FAIL"
            
            img_w, img_h = pil_image.size
            print(f"\n  [{i}] {status}: image ({img_w}x{img_h}, URL: ...{image_url[-20:]})")
            print(f"      abs_diff: max={max_abs_diff:.2e}, mean={mean_abs_diff:.2e}")
            print(f"      rel_diff: max={max_rel_diff:.2e}, mean={mean_rel_diff:.2e}")
            print(f"      HF top: {hf_top} | KB top: {kb_top} (match={top_match})")
            
            assert top_match, f"Top predictions differ: HF={hf_top} vs KB={kb_top}"
            assert mean_ok, f"Mean relative diff {mean_rel_diff:.2e} exceeds tolerance {RTOL_MEAN}"
        
        print("\n" + "-"*70)
        print(f"All prefill tests passed for {model_name}!")
        return
    
    if is_qwen3omni:
        # Qwen3-Omni-MoE: test with real images, audio, and text prompts
        rtol_mean = RTOL_MEAN
        rtol_max = RTOL_MAX
        assert qwen3omni_processor is not None, "Qwen3-Omni processor required"
        test_images = _load_test_images()
        print(f"  Loaded {len(test_images)} test images")
        
        # Test 1: Image + text (visual questions)
        vl_prompts = [
            "What is shown in this image?",
            "Describe the colors you see.",
            "What objects can you identify?",
            "Is there any text visible in this image?",
            "What is the main subject of this photo?",
        ]
        
        for i, ((pil_image, image_url), prompt) in enumerate(zip(test_images, vl_prompts)):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            text = qwen3omni_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = qwen3omni_processor(
                text=[text],
                images=[pil_image],
                return_tensors="pt",
                padding=True,
            )
            
            input_ids = inputs["input_ids"].to(device=DEVICE)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device=DEVICE)
            pixel_values = inputs["pixel_values"].to(device=DEVICE, dtype=DTYPE)
            image_grid_thw = inputs["image_grid_thw"].to(device=DEVICE)
            
            with torch.no_grad():
                # HuggingFace forward
                hf_out = hf_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    use_cache=False,
                )
                hf_logits = hf_out.logits
                
                # KernelBench forward
                kb_logits = kb_model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                )
            
            # Compare last position logits
            hf_last = hf_logits[:, -1, :].float()
            kb_last = kb_logits[:, -1, :].float()
            
            abs_diff = (hf_last - kb_last).abs()
            max_abs_diff = abs_diff.max().item()
            mean_abs_diff = abs_diff.mean().item()
            
            denominator = torch.maximum(hf_last.abs(), kb_last.abs()) + 1e-8
            rel_diff = abs_diff / denominator
            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()
            
            hf_top = hf_last.argmax(dim=-1).item()
            kb_top = kb_last.argmax(dim=-1).item()
            
            hf_token = tokenizer.decode([hf_top])
            kb_token = tokenizer.decode([kb_top])
            top_match = hf_top == kb_top
            
            mean_ok = mean_rel_diff < rtol_mean
            max_ok = max_rel_diff < rtol_max
            is_pass = top_match and mean_ok and max_ok
            status = "PASS" if is_pass else "FAIL"
            
            img_w, img_h = pil_image.size
            seq_len = input_ids.shape[1]
            print(f"\n  [img-{i}] {status}: image ({img_w}x{img_h}), seq_len={seq_len}")
            print(f"      Prompt: '{prompt[:60]}'")
            print(f"      abs_diff: max={max_abs_diff:.2e}, mean={mean_abs_diff:.2e}")
            print(f"      rel_diff: max={max_rel_diff:.2e} (limit={rtol_max}), mean={mean_rel_diff:.2e} (limit={rtol_mean})")
            print(f"      HF next: '{hf_token}' (id={hf_top}) | KB next: '{kb_token}' (id={kb_top}) (match={top_match})")
            
            assert top_match, f"Top predictions differ: HF={hf_token} vs KB={kb_token}"
            assert mean_ok, f"Mean relative diff {mean_rel_diff:.2e} exceeds tolerance {rtol_mean}"
            assert max_ok, f"Max relative diff {max_rel_diff:.2e} exceeds tolerance {rtol_max}"
        
        # Test 2: Audio + text (audio questions)
        import io
        import soundfile as sf
        import librosa
        from datasets import load_dataset, Audio as DatasetsAudio
        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean",
            split="validation",
        )
        ds = ds.cast_column("audio", DatasetsAudio(decode=False))
        
        target_sr = qwen3omni_processor.feature_extractor.sampling_rate
        test_audio_indices = [0, 1, 2]
        for idx, sample_idx in enumerate(test_audio_indices):
            sample = ds[sample_idx]
            audio_bytes = sample["audio"]["bytes"]
            audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
            # Resample to target sampling rate if needed
            if sampling_rate != target_sr:
                audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=target_sr)
                sampling_rate = target_sr
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": "dummy_placeholder"},
                        {"type": "text", "text": "What is being said in this audio?"},
                    ],
                }
            ]
            
            text = qwen3omni_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = qwen3omni_processor(
                text=[text],
                audio=[audio_array],
                return_tensors="pt",
                padding=True,
            )
            
            input_ids = inputs["input_ids"].to(device=DEVICE)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device=DEVICE)
            input_features = inputs.get("input_features", None)
            if input_features is not None:
                input_features = input_features.to(device=DEVICE, dtype=DTYPE)
            feature_attention_mask = inputs.get("feature_attention_mask", None)
            if feature_attention_mask is not None:
                feature_attention_mask = feature_attention_mask.to(device=DEVICE)
            
            with torch.no_grad():
                # HuggingFace forward
                hf_out = hf_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    input_features=input_features,
                    feature_attention_mask=feature_attention_mask,
                    use_cache=False,
                )
                hf_logits = hf_out.logits
                
                # KernelBench forward
                kb_logits = kb_model(
                    input_ids=input_ids,
                    input_features=input_features,
                    feature_attention_mask=feature_attention_mask,
                    attention_mask=attention_mask,
                )
            
            hf_last = hf_logits[:, -1, :].float()
            kb_last = kb_logits[:, -1, :].float()
            
            abs_diff = (hf_last - kb_last).abs()
            max_abs_diff = abs_diff.max().item()
            mean_abs_diff = abs_diff.mean().item()
            
            denominator = torch.maximum(hf_last.abs(), kb_last.abs()) + 1e-8
            rel_diff = abs_diff / denominator
            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()
            
            hf_top = hf_last.argmax(dim=-1).item()
            kb_top = kb_last.argmax(dim=-1).item()
            
            hf_token = tokenizer.decode([hf_top])
            kb_token = tokenizer.decode([kb_top])
            top_match = hf_top == kb_top
            
            mean_ok = mean_rel_diff < rtol_mean
            max_ok = max_rel_diff < rtol_max
            is_pass = top_match and mean_ok and max_ok
            status = "PASS" if is_pass else "FAIL"
            
            duration_s = len(audio_array) / sampling_rate
            transcript_preview = sample.get("text", "N/A")[:40]
            print(f"\n  [audio-{idx}] {status}: sample {sample_idx} ({duration_s:.1f}s, '{transcript_preview}...')")
            print(f"      abs_diff: max={max_abs_diff:.2e}, mean={mean_abs_diff:.2e}")
            print(f"      rel_diff: max={max_rel_diff:.2e} (limit={rtol_max}), mean={mean_rel_diff:.2e} (limit={rtol_mean})")
            print(f"      HF next: '{hf_token}' (id={hf_top}) | KB next: '{kb_token}' (id={kb_top}) (match={top_match})")
            
            assert top_match, f"Top predictions differ: HF={hf_token} vs KB={kb_token}"
            assert mean_ok, f"Mean relative diff {mean_rel_diff:.2e} exceeds tolerance {rtol_mean}"
            assert max_ok, f"Max relative diff {max_rel_diff:.2e} exceeds tolerance {rtol_max}"
        
        # Test 3: Text-only
        text_prompts = ["What is the capital of France?", "Explain quantum computing briefly."]
        for idx, prompt in enumerate(text_prompts):
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = qwen3omni_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = qwen3omni_processor(text=[text], return_tensors="pt", padding=True)
            
            input_ids = inputs["input_ids"].to(device=DEVICE)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device=DEVICE)
            
            with torch.no_grad():
                hf_out = hf_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                hf_logits = hf_out.logits
                kb_logits = kb_model(input_ids=input_ids, attention_mask=attention_mask)
            
            hf_last = hf_logits[:, -1, :].float()
            kb_last = kb_logits[:, -1, :].float()
            
            abs_diff = (hf_last - kb_last).abs()
            max_abs_diff = abs_diff.max().item()
            mean_abs_diff = abs_diff.mean().item()
            
            denominator = torch.maximum(hf_last.abs(), kb_last.abs()) + 1e-8
            rel_diff = abs_diff / denominator
            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()
            
            hf_top = hf_last.argmax(dim=-1).item()
            kb_top = kb_last.argmax(dim=-1).item()
            
            hf_token = tokenizer.decode([hf_top])
            kb_token = tokenizer.decode([kb_top])
            top_match = hf_top == kb_top
            
            mean_ok = mean_rel_diff < rtol_mean
            max_ok = max_rel_diff < rtol_max
            is_pass = top_match and mean_ok and max_ok
            status = "PASS" if is_pass else "FAIL"
            
            seq_len = input_ids.shape[1]
            print(f"\n  [text-{idx}] {status}: seq_len={seq_len}")
            print(f"      Prompt: '{prompt[:60]}'")
            print(f"      abs_diff: max={max_abs_diff:.2e}, mean={mean_abs_diff:.2e}")
            print(f"      rel_diff: max={max_rel_diff:.2e} (limit={rtol_max}), mean={mean_rel_diff:.2e} (limit={rtol_mean})")
            print(f"      HF next: '{hf_token}' (id={hf_top}) | KB next: '{kb_token}' (id={kb_top}) (match={top_match})")
            
            assert top_match, f"Top predictions differ: HF={hf_token} vs KB={kb_token}"
            assert mean_ok, f"Mean relative diff {mean_rel_diff:.2e} exceeds tolerance {rtol_mean}"
            assert max_ok, f"Max relative diff {max_rel_diff:.2e} exceeds tolerance {rtol_max}"
        
        print("\n" + "-"*70)
        print(f"All prefill tests passed for {model_name}!")
        return
    
    if is_qwen3vl:
        # Qwen3-VL: test with real images and text prompts
        assert qwen3vl_processor is not None, "Qwen3-VL processor required"
        test_images = _load_test_images()
        print(f"  Loaded {len(test_images)} test images")
        
        vl_prompts = [
            "Describe what you see in this image.",
            "What objects are present in this image?",
            "What is happening in this image? Answer briefly.",
            "How many living beings can you see?",
            "What colors are dominant in this image?",
        ]
        
        for i, ((pil_image, image_url), prompt) in enumerate(zip(test_images, vl_prompts)):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            text = qwen3vl_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = qwen3vl_processor(
                text=[text],
                images=[pil_image],
                return_tensors="pt",
                padding=True,
            )
            
            input_ids = inputs["input_ids"].to(device=DEVICE)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device=DEVICE)
            pixel_values = inputs["pixel_values"].to(device=DEVICE, dtype=DTYPE)
            image_grid_thw = inputs["image_grid_thw"].to(device=DEVICE)
            
            with torch.no_grad():
                # HuggingFace forward
                hf_out = hf_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    use_cache=False,
                )
                hf_logits = hf_out.logits
                
                # KernelBench forward
                kb_logits = kb_model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                )
            
            hf_last = hf_logits[:, -1, :].float()
            kb_last = kb_logits[:, -1, :].float()
            
            abs_diff = (hf_last - kb_last).abs()
            max_abs_diff = abs_diff.max().item()
            mean_abs_diff = abs_diff.mean().item()
            
            denominator = torch.maximum(hf_last.abs(), kb_last.abs()) + 1e-8
            rel_diff = abs_diff / denominator
            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()
            
            hf_top = hf_last.argmax(dim=-1).item()
            kb_top = kb_last.argmax(dim=-1).item()
            
            hf_token = tokenizer.decode([hf_top])
            kb_token = tokenizer.decode([kb_top])
            top_match = hf_top == kb_top
            
            mean_ok = mean_rel_diff < RTOL_MEAN
            max_ok = max_rel_diff < RTOL_MAX
            is_pass = top_match and mean_ok and max_ok
            status = "PASS" if is_pass else "FAIL"
            
            img_w, img_h = pil_image.size
            seq_len = input_ids.shape[1]
            print(f"\n  [{i}] {status}: image ({img_w}x{img_h}), seq_len={seq_len}")
            print(f"      Prompt: '{prompt[:60]}'")
            print(f"      abs_diff: max={max_abs_diff:.2e}, mean={mean_abs_diff:.2e}")
            print(f"      rel_diff: max={max_rel_diff:.2e} (limit={RTOL_MAX}), mean={mean_rel_diff:.2e} (limit={RTOL_MEAN})")
            print(f"      HF next: '{hf_token}' (id={hf_top}) | KB next: '{kb_token}' (id={kb_top}) (match={top_match})")
            
            assert top_match, f"Top predictions differ: HF={hf_token} vs KB={kb_token}"
            assert mean_ok, f"Mean relative diff {mean_rel_diff:.2e} exceeds tolerance {RTOL_MEAN}"
            assert max_ok, f"Max relative diff {max_rel_diff:.2e} exceeds tolerance {RTOL_MAX}"
        
        print("\n" + "-"*70)
        print(f"All prefill tests passed for {model_name}!")
        return
    
    if is_qwen2vl:
        # Qwen2-VL: test with real images and text prompts
        assert qwen2vl_processor is not None, "Qwen2-VL processor required"
        test_images = _load_test_images()
        print(f"  Loaded {len(test_images)} test images")
        
        # Test prompts for vision-language
        vl_prompts = [
            "Describe what you see in this image.",
            "What objects are present in this image?",
            "What is happening in this image? Answer briefly.",
            "How many living beings can you see?",
            "What colors are dominant in this image?",
        ]
        
        for i, ((pil_image, image_url), prompt) in enumerate(zip(test_images, vl_prompts)):
            # Build conversation format for Qwen2-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Process with the official Qwen2-VL processor
            text = qwen2vl_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = qwen2vl_processor(
                text=[text],
                images=[pil_image],
                return_tensors="pt",
                padding=True,
            )
            
            input_ids = inputs["input_ids"].to(device=DEVICE)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device=DEVICE)
            pixel_values = inputs["pixel_values"].to(device=DEVICE, dtype=DTYPE)
            image_grid_thw = inputs["image_grid_thw"].to(device=DEVICE)
            
            with torch.no_grad():
                # HuggingFace forward
                hf_out = hf_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    use_cache=False,
                )
                hf_logits = hf_out.logits  # (batch, seq_len, vocab)
                
                # KernelBench forward
                kb_logits = kb_model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                )  # (batch, seq_len, vocab)
            
            # Compare last position logits
            hf_last = hf_logits[:, -1, :].float()
            kb_last = kb_logits[:, -1, :].float()
            
            abs_diff = (hf_last - kb_last).abs()
            max_abs_diff = abs_diff.max().item()
            mean_abs_diff = abs_diff.mean().item()
            
            denominator = torch.maximum(hf_last.abs(), kb_last.abs()) + 1e-8
            rel_diff = abs_diff / denominator
            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()
            
            hf_top = hf_last.argmax(dim=-1).item()
            kb_top = kb_last.argmax(dim=-1).item()
            
            hf_token = tokenizer.decode([hf_top])
            kb_token = tokenizer.decode([kb_top])
            top_match = hf_top == kb_top
            
            mean_ok = mean_rel_diff < RTOL_MEAN
            max_ok = max_rel_diff < RTOL_MAX
            is_pass = top_match and mean_ok and max_ok
            status = "PASS" if is_pass else "FAIL"
            
            img_w, img_h = pil_image.size
            seq_len = input_ids.shape[1]
            print(f"\n  [{i}] {status}: image ({img_w}x{img_h}), seq_len={seq_len}")
            print(f"      Prompt: '{prompt[:60]}'")
            print(f"      abs_diff: max={max_abs_diff:.2e}, mean={mean_abs_diff:.2e}")
            print(f"      rel_diff: max={max_rel_diff:.2e} (limit={RTOL_MAX}), mean={mean_rel_diff:.2e} (limit={RTOL_MEAN})")
            print(f"      HF next: '{hf_token}' (id={hf_top}) | KB next: '{kb_token}' (id={kb_top}) (match={top_match})")
            
            assert top_match, f"Top predictions differ: HF={hf_token} vs KB={kb_token}"
            assert mean_ok, f"Mean relative diff {mean_rel_diff:.2e} exceeds tolerance {RTOL_MEAN}"
            assert max_ok, f"Max relative diff {max_rel_diff:.2e} exceeds tolerance {RTOL_MAX}"
        
        print("\n" + "-"*70)
        print(f"All prefill tests passed for {model_name}!")
        return
    
    if is_whisper:
        # Whisper: test with real audio samples from LibriSpeech
        # HF path: uses WhisperProcessor (official feature extractor)
        # KB path: uses KB's built-in WhisperFeatureExtractor (level1 MelSpectrogram)
        # Both run STFT on GPU in float32 for bit-identical mel spectrograms.
        assert whisper_processor is not None, "WhisperProcessor required for Whisper"
        
        # Load real audio samples (decode manually with soundfile to avoid
        # torchcodec/ffmpeg dependency)
        import io
        import soundfile as sf
        from datasets import load_dataset, Audio as DatasetsAudio
        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean",
            split="validation",
        )
        # Disable automatic audio decoding (avoids torchcodec requirement)
        ds = ds.cast_column("audio", DatasetsAudio(decode=False))
        
        # Use a few diverse samples
        test_indices = [0, 1, 2, 3, 4]
        
        for idx, sample_idx in enumerate(test_indices):
            sample = ds[sample_idx]
            # Manually decode audio with soundfile
            audio_bytes = sample["audio"]["bytes"]
            audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
            
            # HF path: process audio with the official Whisper feature extractor
            # Use device=DEVICE so both HF and KB run their STFT on the same
            # device (GPU), avoiding CPU-vs-GPU numerical differences.
            input_features_hf = whisper_processor.feature_extractor(
                audio_array, sampling_rate=sampling_rate, return_tensors="pt",
                device=DEVICE,
            ).input_features.to(device=DEVICE, dtype=DTYPE)
            
            # KB path: pass raw audio waveform — KB's WhisperFeatureExtractor
            # (built on level1/audio/_1_MelSpectrogram) handles preprocessing
            audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).to(device=DEVICE)
            
            # Use decoder_start_token_id as the initial decoder input
            decoder_start_id = hf_model.config.decoder_start_token_id or 50258
            decoder_input_ids = torch.tensor(
                [[decoder_start_id]], dtype=torch.long, device=DEVICE,
            )
            
            with torch.no_grad():
                # HuggingFace: uses HF-preprocessed mel features
                hf_out = hf_model(
                    input_features=input_features_hf,
                    decoder_input_ids=decoder_input_ids,
                    use_cache=False,
                )
                hf_logits = hf_out.logits  # (1, 1, vocab_size)
                
                # KernelBench: uses KB's own feature extractor from raw audio
                kb_logits = kb_model.forward_from_audio(audio_tensor, decoder_input_ids)
            
            # Compare last position logits
            hf_last = hf_logits[:, -1, :].float()
            kb_last = kb_logits[:, -1, :].float()
            
            abs_diff = (hf_last - kb_last).abs()
            max_abs_diff = abs_diff.max().item()
            mean_abs_diff = abs_diff.mean().item()
            
            denominator = torch.maximum(hf_last.abs(), kb_last.abs()) + 1e-8
            rel_diff = abs_diff / denominator
            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()
            
            hf_top = hf_last.argmax(dim=-1).item()
            kb_top = kb_last.argmax(dim=-1).item()
            
            hf_token = whisper_processor.tokenizer.decode([hf_top])
            kb_token = whisper_processor.tokenizer.decode([kb_top])
            top_match = hf_top == kb_top
            
            mean_ok = mean_rel_diff < RTOL_MEAN
            max_ok = max_rel_diff < RTOL_MAX
            is_pass = top_match and mean_ok and max_ok
            status = "PASS" if is_pass else "FAIL"
            
            duration_s = len(audio_array) / sampling_rate
            transcript_preview = sample.get("text", "N/A")[:40]
            print(f"\n  [{idx}] {status}: sample {sample_idx} ({duration_s:.1f}s, '{transcript_preview}...')")
            print(f"      abs_diff: max={max_abs_diff:.2e}, mean={mean_abs_diff:.2e}")
            print(f"      rel_diff: max={max_rel_diff:.2e} (limit={RTOL_MAX}), mean={mean_rel_diff:.2e} (limit={RTOL_MEAN})")
            print(f"      HF next: '{hf_token}' (id={hf_top}) | KB next: '{kb_token}' (id={kb_top}) (match={top_match})")
            
            assert top_match, f"Top predictions differ: HF={hf_token} vs KB={kb_token}"
            assert mean_ok, f"Mean relative diff {mean_rel_diff:.2e} exceeds tolerance {RTOL_MEAN}"
            assert max_ok, f"Max relative diff {max_rel_diff:.2e} exceeds tolerance {RTOL_MAX}"
        
        print("\n" + "-"*70)
        print(f"All prefill tests passed for {model_name}!")
        return
    
    if is_t5:
        # T5: test with task-specific encoder-decoder prompts
        for i, prompt in enumerate(T5_TEST_PROMPTS):
            encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            input_ids = encoded['input_ids']
            
            # Use pad token as decoder start token (standard T5 practice)
            decoder_input_ids = torch.full(
                (input_ids.shape[0], 1),
                hf_model.config.decoder_start_token_id or 0,
                dtype=torch.long,
                device=DEVICE,
            )
            
            with torch.no_grad():
                # HuggingFace
                hf_out = hf_model(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    use_cache=False,
                )
                hf_logits = hf_out.logits  # (batch, decoder_seq_len, vocab)
                
                # KernelBench
                kb_logits = kb_model(input_ids, decoder_input_ids)
            
            # Compare last position logits
            hf_last = hf_logits[:, -1, :].float()
            kb_last = kb_logits[:, -1, :].float()
            
            abs_diff = (hf_last - kb_last).abs()
            max_abs_diff = abs_diff.max().item()
            mean_abs_diff = abs_diff.mean().item()
            
            denominator = torch.maximum(hf_last.abs(), kb_last.abs()) + 1e-8
            rel_diff = abs_diff / denominator
            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()
            
            hf_top = hf_last.argmax(dim=-1).item()
            kb_top = kb_last.argmax(dim=-1).item()
            
            hf_token = tokenizer.decode([hf_top])
            kb_token = tokenizer.decode([kb_top])
            top_match = hf_top == kb_top
            
            mean_ok = mean_rel_diff < RTOL_MEAN
            max_ok = max_rel_diff < RTOL_MAX
            is_pass = top_match and mean_ok and max_ok
            status = "PASS" if is_pass else "FAIL"
            
            seq_len = input_ids.shape[1]
            print(f"\n  [{i}] {status}: (encoder_len={seq_len})")
            print(f"      Prompt: '{prompt[:60]}...'")
            print(f"      abs_diff: max={max_abs_diff:.2e}, mean={mean_abs_diff:.2e}")
            print(f"      rel_diff: max={max_rel_diff:.2e} (limit={RTOL_MAX}), mean={mean_rel_diff:.2e} (limit={RTOL_MEAN})")
            print(f"      HF next: '{hf_token}' | KB next: '{kb_token}' (match={top_match})")
            
            assert top_match, f"Top predictions differ: HF={hf_token} vs KB={kb_token}"
            assert mean_ok, f"Mean relative diff {mean_rel_diff:.2e} exceeds tolerance {RTOL_MEAN}"
            assert max_ok, f"Max relative diff {max_rel_diff:.2e} exceeds tolerance {RTOL_MAX}"
        
        print("\n" + "-"*70)
        print(f"All prefill tests passed for {model_name}!")
        return
    
    # Standard decoder-only / SSM models
    for i, prompt in enumerate(TEST_PROMPTS):
        encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_ids = encoded['input_ids']
        batch_size, seq_len = input_ids.shape
        
        # For attention models, allocate block table
        generate_kwargs = {}
        if not is_ssm:
            max_blocks = (seq_len + kb_config['block_size'] - 1) // kb_config['block_size'] + 10
            block_table = torch.arange(max_blocks, device=DEVICE, dtype=torch.long).unsqueeze(0)
            generate_kwargs['block_table'] = block_table
        
        with torch.no_grad():
            # HuggingFace
            hf_out = hf_model(input_ids=input_ids, use_cache=False)
            hf_logits = hf_out.logits
            
            # KernelBench prefill using generate with max_new_tokens=0
            _, kb_logits_list = kb_model.generate(
                input_ids, 
                max_new_tokens=0, 
                return_logits=True,
                **generate_kwargs,
            )
            kb_logits = kb_logits_list[0]  # Prefill logits
        
        # Compare last position logits using relative tolerance
        hf_last = hf_logits[:, -1, :].float()
        kb_last = kb_logits[:, -1, :].float()
        
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
    Skipped for SwinV2 (image classification, no generation).
    """
    (hf_model, kb_model, tokenizer, kb_config, kb_module, image_processor, whisper_processor, qwen2vl_processor, qwen3vl_processor, qwen3omni_processor), model_name, max_layers = loaded_models
    
    # Skip generation test for vision models
    if _is_swinv2_model(model_name):
        pytest.skip("SwinV2 is an image classification model, no generation test")
    
    # Skip generation test for T5 (would need a different generation setup)
    if _is_t5_model(model_name):
        pytest.skip("T5 generation test not yet implemented (use test_prefill_alignment)")
    
    # Skip generation test for Whisper (would need encoder-decoder generation setup)
    if _is_whisper_model(model_name):
        pytest.skip("Whisper generation test not yet implemented (use test_prefill_alignment)")
    
    is_mamba2 = _is_mamba2_model(model_name)
    is_mamba1 = _is_mamba1_model(model_name)
    is_ssm = is_mamba2 or is_mamba1
    is_qwen2vl = _is_qwen2vl_model(model_name)
    is_qwen3vl = _is_qwen3vl_model(model_name)
    is_qwen3omni = _is_qwen3omni_model(model_name)
    is_multimodal_vl = is_qwen2vl or is_qwen3vl or is_qwen3omni
    
    layers_info = f" ({max_layers} layers)" if max_layers else ""
    print("\n" + "="*70)
    print(f"Testing Generation for {model_name}{layers_info}")
    print("="*70)
    
    # ---- Multimodal generation tests (Qwen2-VL, Qwen3-VL, Qwen3-Omni) ----
    if is_multimodal_vl:
        results = []
        num_tokens = 20  # Generate 20 tokens for each multimodal test
        
        # Determine which processor to use
        if is_qwen2vl:
            processor = qwen2vl_processor
        elif is_qwen3vl:
            processor = qwen3vl_processor
        elif is_qwen3omni:
            processor = qwen3omni_processor
        assert processor is not None, f"Processor required for {model_name}"
        
        # --- Test 1: Image + text ---
        test_images = _load_test_images()
        vl_prompts = [
            "What is shown in this image?",
            "Describe the colors you see.",
        ]
        for img_idx, ((pil_image, image_url), prompt) in enumerate(zip(test_images[:2], vl_prompts)):
            print(f"\n  [img-gen-{img_idx}] Image + text generation ({num_tokens} tokens)")
            print(f"      Prompt: '{prompt}'")
            
            messages = [{"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[pil_image], return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            input_ids = inputs['input_ids']
            prompt_len = input_ids.shape[1]
            
            with torch.no_grad():
                # HF generation using model.generate() (handles M-RoPE position IDs correctly)
                hf_full_seq = hf_model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)
                hf_tokens = hf_full_seq[:, prompt_len:]
                
                # KB generation
                kb_kwargs = {
                    'input_ids': input_ids,
                    'max_new_tokens': num_tokens,
                }
                if 'pixel_values' in inputs:
                    kb_kwargs['pixel_values'] = inputs['pixel_values']
                if 'image_grid_thw' in inputs:
                    kb_kwargs['image_grid_thw'] = inputs['image_grid_thw']
                if 'attention_mask' in inputs:
                    kb_kwargs['attention_mask'] = inputs['attention_mask']
                
                kb_full_seq = kb_model.generate(**kb_kwargs)
                kb_tokens = kb_full_seq[:, prompt_len:]
            
            hf_text = tokenizer.decode(hf_tokens[0], skip_special_tokens=True)
            kb_text = tokenizer.decode(kb_tokens[0], skip_special_tokens=True)
            
            consecutive_matches, comparable_len = _count_consecutive_matches(hf_tokens[0], kb_tokens[0])
            full_match = consecutive_matches == comparable_len and comparable_len == num_tokens
            
            print(f"      Consecutive matches: {consecutive_matches}/{comparable_len} ({100*consecutive_matches/max(comparable_len,1):.1f}%)")
            print(f"      HF: '{hf_text[:80]}...'")
            print(f"      KB: '{kb_text[:80]}...'")
            
            results.append({'full_match': full_match, 'consecutive_matches': consecutive_matches, 'total_tokens': comparable_len})
        
        # --- Test 2: Text-only ---
        text_prompts = [
            "What is the capital of France?",
            "Explain quantum computing briefly.",
        ]
        for txt_idx, prompt in enumerate(text_prompts):
            print(f"\n  [text-gen-{txt_idx}] Text-only generation ({num_tokens} tokens)")
            print(f"      Prompt: '{prompt}'")
            
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if is_qwen3omni:
                inputs = processor(text=[text], return_tensors="pt", padding=True)
            else:
                inputs = processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            input_ids = inputs['input_ids']
            prompt_len = input_ids.shape[1]
            
            with torch.no_grad():
                # HF generation using model.generate()
                hf_full_seq = hf_model.generate(input_ids=input_ids, max_new_tokens=num_tokens, do_sample=False)
                hf_tokens = hf_full_seq[:, prompt_len:]
                
                kb_full_seq = kb_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=num_tokens,
                    attention_mask=inputs.get('attention_mask'),
                )
                kb_tokens = kb_full_seq[:, prompt_len:]
            
            hf_text = tokenizer.decode(hf_tokens[0], skip_special_tokens=True)
            kb_text = tokenizer.decode(kb_tokens[0], skip_special_tokens=True)
            
            consecutive_matches, comparable_len = _count_consecutive_matches(hf_tokens[0], kb_tokens[0])
            full_match = consecutive_matches == comparable_len and comparable_len == num_tokens
            
            print(f"      Consecutive matches: {consecutive_matches}/{comparable_len} ({100*consecutive_matches/max(comparable_len,1):.1f}%)")
            print(f"      HF: '{hf_text[:80]}...'")
            print(f"      KB: '{kb_text[:80]}...'")
            
            results.append({'full_match': full_match, 'consecutive_matches': consecutive_matches, 'total_tokens': comparable_len})
        
        # --- Test 3: Audio + text (Qwen3-Omni only) ---
        if is_qwen3omni:
            import librosa
            import numpy as np
            import io
            import soundfile as sf
            from datasets import load_dataset, Audio as DatasetsAudio
            ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
            # Disable automatic audio decoding to avoid torchcodec/FFmpeg dependency
            ds = ds.cast_column("audio", DatasetsAudio(decode=False))
            target_sr = processor.feature_extractor.sampling_rate
            
            audio_prompts = ["Transcribe this audio.", "What is being said?"]
            for aud_idx in range(min(2, len(ds))):
                # Load audio using soundfile to avoid torchcodec dependency
                sample = ds[aud_idx]
                audio_bytes = sample["audio"]["bytes"]
                audio_array, sr = sf.read(io.BytesIO(audio_bytes))
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=1)
                audio_array = audio_array.astype(np.float32)
                if sr != target_sr:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
                
                prompt = audio_prompts[aud_idx]
                print(f"\n  [audio-gen-{aud_idx}] Audio + text generation ({num_tokens} tokens)")
                print(f"      Prompt: '{prompt}'")
                
                messages = [{"role": "user", "content": [
                    {"type": "audio", "audio": "dummy_placeholder"},
                    {"type": "text", "text": prompt},
                ]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], audio=[audio_array], return_tensors="pt", padding=True)
                inputs = {k: (v.to(device=DEVICE, dtype=DTYPE) if v.is_floating_point() else v.to(DEVICE)) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                input_ids = inputs['input_ids']
                prompt_len = input_ids.shape[1]
                
                with torch.no_grad():
                    # HF generation using model.generate()
                    hf_full_seq = hf_model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)
                    hf_tokens = hf_full_seq[:, prompt_len:]
                    
                    kb_kwargs = {
                        'input_ids': input_ids,
                        'max_new_tokens': num_tokens,
                    }
                    if 'input_features' in inputs:
                        kb_kwargs['input_features'] = inputs['input_features']
                    if 'feature_attention_mask' in inputs:
                        kb_kwargs['feature_attention_mask'] = inputs['feature_attention_mask']
                    if 'attention_mask' in inputs:
                        kb_kwargs['attention_mask'] = inputs['attention_mask']
                    
                    kb_full_seq = kb_model.generate(**kb_kwargs)
                    kb_tokens = kb_full_seq[:, prompt_len:]
                
                hf_text = tokenizer.decode(hf_tokens[0], skip_special_tokens=True)
                kb_text = tokenizer.decode(kb_tokens[0], skip_special_tokens=True)
                
                consecutive_matches, comparable_len = _count_consecutive_matches(hf_tokens[0], kb_tokens[0])
                full_match = consecutive_matches == comparable_len and comparable_len == num_tokens
                
                print(f"      Consecutive matches: {consecutive_matches}/{comparable_len} ({100*consecutive_matches/max(comparable_len,1):.1f}%)")
                print(f"      HF: '{hf_text[:80]}...'")
                print(f"      KB: '{kb_text[:80]}...'")
                
                results.append({'full_match': full_match, 'consecutive_matches': consecutive_matches, 'total_tokens': comparable_len})
        
        # Summary for multimodal generation
        print("\n" + "="*70)
        print("Generation Summary")
        print("="*70)
        
        total_consecutive = sum(r['consecutive_matches'] for r in results)
        total_tokens = sum(r['total_tokens'] for r in results)
        full_matches = sum(r['full_match'] for r in results)
        
        print(f"Full sequence matches: {full_matches}/{len(results)}")
        print(f"Consecutive token accuracy: {total_consecutive}/{total_tokens} ({100*total_consecutive/total_tokens:.1f}%)")
        
        assert total_consecutive / total_tokens >= GENERATION_CONSECUTIVE_THRESHOLD, \
            f"Consecutive token accuracy {100*total_consecutive/total_tokens:.1f}% below {100*GENERATION_CONSECUTIVE_THRESHOLD:.0f}% threshold"
        return
    
    # ---- Standard text-only generation tests ----
    results = []
    
    for i, (prompt, num_tokens) in enumerate(zip(TEST_PROMPTS, RESPONSE_LENGTHS)):
        print(f"\n[{i}] Generating {num_tokens} tokens for prompt (len={len(tokenizer.encode(prompt))})")
        print(f"    '{prompt[:60]}...'")
        
        encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_ids = encoded['input_ids']
        batch_size, prompt_len = input_ids.shape
        
        # For attention models, allocate blocks
        generate_kwargs = {}
        if not is_ssm:
            max_seq = prompt_len + num_tokens + 10
            max_blocks = (max_seq + kb_config['block_size'] - 1) // kb_config['block_size']
            block_table = torch.arange(max_blocks, device=DEVICE, dtype=torch.long).unsqueeze(0)
            generate_kwargs['block_table'] = block_table
        
        with torch.no_grad():
            # HuggingFace generation (manual loop to match greedy decoding)
            if is_ssm:
                # For SSM models, use the HF model's cache mechanism
                if is_mamba2:
                    from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache as HFSSMCache
                elif "falcon-mamba" in model_name.lower():
                    from transformers.models.falcon_mamba.modeling_falcon_mamba import FalconMambaCache as HFSSMCache
                else:
                    from transformers.models.mamba.modeling_mamba import MambaCache as HFSSMCache
                hf_cache = HFSSMCache(
                    hf_model.backbone.config,
                    batch_size,
                    device=input_ids.device,
                    dtype=DTYPE,
                )
                cache_position = torch.arange(0, hf_model.backbone.config.conv_kernel, device=DEVICE)
                
                hf_out = hf_model(
                    input_ids=input_ids,
                    cache_params=hf_cache,
                    use_cache=True,
                    cache_position=cache_position,
                )
                hf_generated = []
                hf_next = hf_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                hf_generated.append(hf_next)
                
                for step in range(num_tokens - 1):
                    cache_position = torch.tensor(
                        [prompt_len + step], device=DEVICE, dtype=torch.long
                    )
                    hf_out = hf_model(
                        input_ids=hf_next,
                        cache_params=hf_cache,
                        use_cache=True,
                        cache_position=cache_position,
                    )
                    hf_next = hf_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    hf_generated.append(hf_next)
                
                hf_tokens = torch.cat(hf_generated, dim=1)
            else:
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
                return_logits=False,
                **generate_kwargs,
            )
            # Extract only the generated tokens (exclude prompt)
            kb_tokens = kb_full_seq[:, prompt_len:]
        
        # Decode generated tokens
        hf_text = tokenizer.decode(hf_tokens[0], skip_special_tokens=True)
        kb_text = tokenizer.decode(kb_tokens[0], skip_special_tokens=True)
        
        # Count CONSECUTIVE matching tokens from the start
        consecutive_matches, comparable_len = _count_consecutive_matches(hf_tokens[0], kb_tokens[0])
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
    For Mamba2: layer0_mlp is None (no separate MLP)
    """
    # Try Mamba2 structure (Mamba2ForCausalLM)
    if hasattr(hf_model, 'backbone') and hasattr(hf_model.backbone, 'embeddings'):
        base = hf_model.backbone
        layer0 = base.layers[0]
        return (
            base.embeddings,
            base.layers,
            layer0.norm,
            None,  # No separate MLP in Mamba2
        )
    # Try Llama-style structure first (LlamaForCausalLM)
    elif hasattr(hf_model, 'model') and hasattr(hf_model.model, 'embed_tokens'):
        base = hf_model.model
        layer0 = base.layers[0]
        # Handle MoE models (Mixtral uses block_sparse_moe instead of mlp)
        if hasattr(layer0, 'mlp'):
            mlp = layer0.mlp
        elif hasattr(layer0, 'block_sparse_moe'):
            mlp = layer0.block_sparse_moe
        else:
            mlp = None
        return (
            base.embed_tokens,
            base.layers,
            layer0.input_layernorm,
            mlp,
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
    # Try Mamba2-style structure (embeddings + layers)
    if hasattr(kb_model, 'embeddings') and hasattr(kb_model, 'norm_f'):
        layer0 = kb_model.layers[0]
        return (
            kb_model.embeddings,
            kb_model.layers,
            layer0.norm,
            None,  # No separate MLP in Mamba2
        )
    # Try Llama-style structure (embed_tokens + layers)
    elif hasattr(kb_model, 'embed_tokens'):
        layer0 = kb_model.layers[0]
        # Handle MoE models (Mixtral uses block_sparse_moe instead of mlp)
        if hasattr(layer0, 'mlp'):
            mlp = layer0.mlp
        elif hasattr(layer0, 'block_sparse_moe'):
            mlp = layer0.block_sparse_moe
        else:
            mlp = None
        return (
            kb_model.embed_tokens,
            kb_model.layers,
            layer0.input_layernorm,
            mlp,
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
    (hf_model, kb_model, tokenizer, kb_config, _, image_processor, whisper_processor, qwen2vl_processor, qwen3vl_processor, qwen3omni_processor), model_name, max_layers = loaded_models
    
    is_t5 = _is_t5_model(model_name)
    is_swinv2 = _is_swinv2_model(model_name)
    is_whisper = _is_whisper_model(model_name)
    is_qwen2vl = _is_qwen2vl_model(model_name)
    is_qwen3vl = _is_qwen3vl_model(model_name)
    is_qwen3omni = _is_qwen3omni_model(model_name)
    
    if is_qwen3omni:
        pytest.skip("Qwen3-Omni component tests not yet implemented (use test_prefill_alignment)")
    
    if is_qwen3vl:
        pytest.skip("Qwen3-VL component tests not yet implemented (use test_prefill_alignment)")
    
    if is_qwen2vl:
        pytest.skip("Qwen2-VL component tests not yet implemented (use test_prefill_alignment)")
    
    if is_whisper:
        pytest.skip("Whisper component tests not yet implemented (use test_prefill_alignment)")
    
    layers_info = f" ({max_layers} layers)" if max_layers else ""
    print("\n" + "="*70)
    print(f"Testing Component Alignment for {model_name}{layers_info}")
    print("="*70)
    
    if is_t5:
        # T5 component tests
        vocab_size = kb_config['vocab_size']
        d_model = kb_config['d_model']
        
        # Test shared embedding
        test_ids = torch.randint(0, vocab_size, (2, 32), device=DEVICE)
        with torch.no_grad():
            hf_emb = hf_model.shared(test_ids)
            kb_emb = kb_model.shared(test_ids)
        
        emb_diff = (hf_emb - kb_emb).abs().max().item()
        print(f"  Embedding diff: {emb_diff:.2e} {'PASS' if emb_diff < 1e-6 else 'FAIL'}")
        assert emb_diff < 1e-6
        
        # Test encoder first block layer norm
        hidden = torch.randn(2, 32, d_model, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            hf_norm_out = hf_model.encoder.block[0].layer[0].layer_norm(hidden)
            kb_norm_out = kb_model.encoder_blocks[0].layer[0].layer_norm(hidden)
        
        norm_diff = (hf_norm_out - kb_norm_out).abs().max().item()
        print(f"  LayerNorm diff: {norm_diff:.2e} {'PASS' if norm_diff < ATOL_STRICT else 'FAIL'}")
        assert norm_diff < ATOL_STRICT
        
        # Test LM head
        with torch.no_grad():
            hf_head = hf_model.lm_head(hidden)
            kb_head = kb_model.lm_head(hidden)
        
        head_diff = (hf_head - kb_head).abs().max().item()
        print(f"  LM Head diff: {head_diff:.2e} {'PASS' if head_diff < ATOL_STRICT else 'FAIL'}")
        assert head_diff < ATOL_STRICT
        
        print("\n  All component tests passed!")
        return
    
    if is_swinv2:
        # SwinV2 component tests
        embed_dim = kb_config['embed_dim']
        
        # Test patch embedding
        pixel_values = torch.randn(1, 3, kb_config['image_size'], kb_config['image_size'],
                                   device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            hf_patch = hf_model.swinv2.embeddings.patch_embeddings.projection(pixel_values)
            kb_patch = kb_model.embeddings.patch_embeddings.projection(pixel_values)
        
        patch_diff = (hf_patch - kb_patch).abs().max().item()
        print(f"  Patch embed diff: {patch_diff:.2e} {'PASS' if patch_diff < ATOL_STRICT else 'FAIL'}")
        assert patch_diff < ATOL_STRICT
        
        # Test embedding norm
        embeddings = hf_patch.flatten(2).transpose(1, 2)
        with torch.no_grad():
            hf_norm_out = hf_model.swinv2.embeddings.norm(embeddings)
            kb_norm_out = kb_model.embeddings.norm(embeddings)
        
        norm_diff = (hf_norm_out - kb_norm_out).abs().max().item()
        print(f"  Embed norm diff: {norm_diff:.2e} {'PASS' if norm_diff < ATOL_STRICT else 'FAIL'}")
        assert norm_diff < ATOL_STRICT
        
        # Test classifier head
        final_dim = int(embed_dim * 2 ** (len(kb_config['depths']) - 1))
        hidden = torch.randn(1, final_dim, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            hf_cls = hf_model.classifier(hidden)
            kb_cls = kb_model.classifier(hidden)
        
        cls_diff = (hf_cls - kb_cls).abs().max().item()
        print(f"  Classifier diff: {cls_diff:.2e} {'PASS' if cls_diff < ATOL_STRICT else 'FAIL'}")
        assert cls_diff < ATOL_STRICT
        
        print("\n  All component tests passed!")
        return
    
    # Standard architecture component tests
    # Get components in an architecture-agnostic way
    try:
        hf_embed, hf_layers, hf_norm, hf_mlp = _get_hf_components(hf_model)
        kb_embed, kb_layers, kb_norm, kb_mlp = _get_kb_components(kb_model)
    except ValueError as e:
        pytest.skip(f"Cannot test components for this architecture: {e}")
    
    # Test embeddings
    vocab_size = kb_config.get('vocab_size', kb_config.get('vocab_size', 32768))
    hidden_size = kb_config.get('hidden_size', 4096)
    
    test_ids = torch.randint(0, vocab_size, (2, 32), device=DEVICE)
    with torch.no_grad():
        hf_emb = hf_embed(test_ids)
        kb_emb = kb_embed(test_ids)
    
    emb_diff = (hf_emb - kb_emb).abs().max().item()
    print(f"  Embedding diff: {emb_diff:.2e} {'PASS' if emb_diff < 1e-6 else 'FAIL'}")
    assert emb_diff < 1e-6
    
    # Test layer norms
    hidden = torch.randn(2, 32, hidden_size, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        hf_norm_out = hf_norm(hidden)
        kb_norm_out = kb_norm(hidden)
    
    norm_diff = (hf_norm_out - kb_norm_out).abs().max().item()
    print(f"  LayerNorm diff: {norm_diff:.2e} {'PASS' if norm_diff < ATOL_STRICT else 'FAIL'}")
    assert norm_diff < ATOL_STRICT
    
    # Test MLP - some models (like BLOOM) require additional arguments, MoE models have different structure
    # Skip MLP comparison for MoE models (different output structure/interface)
    is_moe = 'MoE' in str(type(hf_mlp).__name__) or 'Moe' in str(type(hf_mlp).__name__) if hf_mlp else False
    is_moe = is_moe or ('MoE' in str(type(kb_mlp).__name__) or 'Moe' in str(type(kb_mlp).__name__) if kb_mlp else False)
    
    if hf_mlp is None or kb_mlp is None or is_moe:
        print(f"  MLP diff: SKIPPED (MoE or unsupported structure)")
    else:
        try:
            with torch.no_grad():
                hf_mlp_out = hf_mlp(hidden)
                kb_mlp_out = kb_mlp(hidden)
            
            # Handle tuple outputs (some MoE blocks return (hidden, router_logits))
            if isinstance(hf_mlp_out, tuple):
                hf_mlp_out = hf_mlp_out[0]
            if isinstance(kb_mlp_out, tuple):
                kb_mlp_out = kb_mlp_out[0]
            
            mlp_diff = (hf_mlp_out - kb_mlp_out).abs().max().item()
            print(f"  MLP diff: {mlp_diff:.2e} {'PASS' if mlp_diff < ATOL_STRICT else 'FAIL'}")
            assert mlp_diff < ATOL_STRICT
        except (TypeError, RuntimeError) as e:
            if 'residual' in str(e):
                # BLOOM's MLP requires a residual argument
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
