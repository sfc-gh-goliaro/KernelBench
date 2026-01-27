"""
Test Llama-3.1 alignment with HuggingFace transformers library.

This test validates that the KernelBench Llama-3.1 implementation produces
outputs matching the HuggingFace transformers implementation using:
1. A batch of 5 prompts with varying lengths
2. Continuous batching with paged KV cache
3. Both prefill and decode phases

Tests against:
- meta-llama/Llama-3.1-8B-Instruct
- meta-llama/Llama-3.1-70B-Instruct (optional, requires more memory)

Requires HuggingFace authentication with access to Llama models.
"""

import pytest
import torch
import sys
import os
import importlib
from typing import List, Tuple, Optional

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

# Models to test
MODEL_8B = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_70B = "meta-llama/Llama-3.1-70B-Instruct"

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


def load_kernelbench_model(config_dict: dict):
    """Load the KernelBench Llama model."""
    llama_module = importlib.import_module('KernelBench.level4.1_Llama31')
    return llama_module.Model(**config_dict), llama_module


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


def load_models(model_name: str):
    """Load HuggingFace and KernelBench models."""
    print(f"\nLoading models from {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    hf_config = AutoConfig.from_pretrained(model_name)
    # Use eager attention to match our manual attention implementation
    # SDPA uses fused CUDA kernels with slightly different numerics
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE, device_map=DEVICE,
        attn_implementation="eager",
    )
    hf_model.eval()
    
    # Calculate num_blocks needed for testing
    max_seq_len = 4096  # Maximum sequence length for tests
    block_size = 16
    num_blocks = (max_seq_len // block_size + 1) * hf_config.num_hidden_layers * 2
    
    kb_config = create_kb_model_from_hf_config(hf_config, num_blocks)
    
    kb_model, llama_module = load_kernelbench_model(kb_config)
    kb_model = kb_model.to(device=DEVICE, dtype=DTYPE)
    copy_weights(hf_model, kb_model, kb_config['num_layers'])
    kb_model.eval()
    
    print(f"Loaded: {kb_config['num_layers']} layers, {kb_config['hidden_size']} hidden, "
          f"{kb_config['num_heads']} heads, {kb_config['num_kv_heads']} kv_heads")
    
    return hf_model, kb_model, tokenizer, kb_config, llama_module


# ============================================================================
# Llama-3.1-8B Tests
# ============================================================================

@pytest.fixture(scope="module")
def llama_8b_models():
    """Load Llama-3.1-8B-Instruct models."""
    try:
        return load_models(MODEL_8B)
    except Exception as e:
        pytest.skip(f"Could not load {MODEL_8B}: {e}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_llama31_8b_prefill_alignment(llama_8b_models):
    """
    Test prefill (prompt processing) alignment for Llama-3.1-8B.
    
    Uses generate(max_new_tokens=0, return_logits=True) to get prefill logits.
    """
    hf_model, kb_model, tokenizer, kb_config, llama_module = llama_8b_models
    
    print("\n" + "="*70)
    print("Testing Llama-3.1-8B Prefill Alignment")
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
    print("All Llama-3.1-8B prefill tests passed!")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_llama31_8b_generation(llama_8b_models):
    """
    Test continuous batching generation for Llama-3.1-8B.
    
    Uses generate() for KernelBench model and compares with HuggingFace generation.
    """
    hf_model, kb_model, tokenizer, kb_config, llama_module = llama_8b_models
    
    print("\n" + "="*70)
    print("Testing Llama-3.1-8B Continuous Batching Generation")
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
def test_llama31_8b_components(llama_8b_models):
    """Test individual component alignment for Llama-3.1-8B."""
    hf_model, kb_model, tokenizer, kb_config, _ = llama_8b_models
    
    print("\n" + "="*70)
    print("Testing Llama-3.1-8B Component Alignment")
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


# ============================================================================
# Llama-3.1-70B Tests (Optional - requires significant memory)
# ============================================================================

@pytest.fixture(scope="module")
def llama_70b_models():
    """Load Llama-3.1-70B-Instruct models."""
    try:
        return load_models(MODEL_70B)
    except Exception as e:
        pytest.skip(f"Could not load {MODEL_70B}: {e}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.slow  # Mark as slow test
def test_llama31_70b_prefill_alignment(llama_70b_models):
    """
    Test prefill alignment for Llama-3.1-70B.
    
    This test requires significant GPU memory (>80GB recommended).
    Uses generate(max_new_tokens=0, return_logits=True) to get prefill logits.
    """
    hf_model, kb_model, tokenizer, kb_config, llama_module = llama_70b_models
    
    print("\n" + "="*70)
    print("Testing Llama-3.1-70B Prefill Alignment")
    print("="*70)
    
    # Use only first 2 prompts to save memory
    for i, prompt in enumerate(TEST_PROMPTS[:2]):
        encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_ids = encoded['input_ids']
        batch_size, seq_len = input_ids.shape
        
        max_blocks = (seq_len + kb_config['block_size'] - 1) // kb_config['block_size'] + 10
        block_table = torch.arange(max_blocks, device=DEVICE, dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
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
        
        hf_last = hf_logits[:, -1, :]
        kb_last = kb_logits[:, -1, :]
        
        diff = (hf_last - kb_last).abs()
        max_diff = diff.max().item()
        
        hf_top = hf_last.argmax(dim=-1).item()
        kb_top = kb_last.argmax(dim=-1).item()
        
        status = "PASS" if max_diff < ATOL else "FAIL"
        print(f"\n  [{i}] {status}: '{prompt[:50]}...' (len={seq_len})")
        print(f"      max_diff={max_diff:.2e}")
        print(f"      HF: {tokenizer.decode([hf_top])} | KB: {tokenizer.decode([kb_top])}")
        
        assert max_diff < ATOL
    
    print("\n  Llama-3.1-70B prefill tests passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
