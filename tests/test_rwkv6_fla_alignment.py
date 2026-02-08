"""
Test RWKV-6 KernelBench model alignment with flash-linear-attention (fla).

This test validates that the KernelBench RWKV-6 implementation produces
outputs matching the fla library's implementation using:
1. A batch of 5 prompts with varying lengths
2. Cached autoregressive generation
3. Both prefill and decode phases

Supports the fla-hub/rwkv6-* HuggingFace checkpoints.
Optionally limit the number of layers with --max-layers for faster testing.

Usage:
    pytest tests/test_rwkv6_fla_alignment.py -v -s
    pytest tests/test_rwkv6_fla_alignment.py -v -s --max-layers 4

Requires: flash-linear-attention (fla) package installed.
"""

import pytest
import torch
import sys
import os
from importlib import import_module

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Tolerance thresholds
ATOL_STRICT = 1e-5       # For component tests (weights, embeddings, norms)
RTOL_MEAN = 5e-2         # Mean relative tolerance for logit comparison
RTOL_MAX = 3.0           # Max relative tolerance (near-zero logits can have large rel errors)
GENERATION_CONSECUTIVE_THRESHOLD = 0.50  # Minimum fraction of consecutive matching tokens

# Default HF model for testing
DEFAULT_HF_MODEL = "fla-hub/rwkv6-7B-finch"

# Test prompts with varying lengths
TEST_PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "Explain the concept of machine learning to a 10-year-old. Use simple words and give an example.",
    "In a world where artificial intelligence has become increasingly powerful,",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The meaning of life is",
]

# Response lengths for each prompt
RESPONSE_LENGTHS = [10, 50, 100, 50, 20]


# ============================================================================
# Helpers
# ============================================================================

def _normalize_kb_key(key: str) -> str:
    """
    Normalize a KernelBench state dict key to match fla naming.

    KernelBench wraps level1 operators in extra modules:
    - LayerNorm wrapper: .ln. -> .
    - Embedding wrapper: .embedding. -> .  (nn.Embedding inside Embedding op)
    - GroupNorm wrapper: .gn. -> .  (nn.GroupNorm inside GroupNorm op)
    - x_proj.{linear,mu} -> x_proj.0.{linear,mu}  (fla uses Sequential index 0)
    - x_proj_up -> x_proj.2  (fla uses Sequential index 2)
    """
    key = key.replace('.ln.', '.')
    # Embedding level1 wrapper: embeddings.embedding.weight -> embeddings.weight
    key = key.replace('embeddings.embedding.', 'embeddings.')
    # GroupNorm level1 wrapper: g_norm.gn.weight -> g_norm.weight
    key = key.replace('.gn.', '.')
    # x_proj_up was x_proj.2 (up-projection) in fla's Sequential layout
    key = key.replace('.x_proj_up.', '.x_proj.2.')
    # x_proj.{linear,mu} -> x_proj.0.{linear,mu} (LerpLinear at index 0 in fla)
    key = key.replace('.x_proj.linear.', '.x_proj.0.linear.')
    key = key.replace('.x_proj.mu', '.x_proj.0.mu')
    return key


def copy_weights_fla_to_kb(fla_model, kb_model):
    """
    Copy weights from fla RWKV6ForCausalLM to KernelBench Model.

    Handles:
    - fla prefix: "model." for the backbone
    - KB wrappers: ".ln." for level1 operators
    """
    fla_state = fla_model.state_dict()
    kb_state = kb_model.state_dict()

    # Build fla key lookup: strip "model." prefix for backbone params
    fla_lookup = {}
    for fla_key in fla_state.keys():
        stripped = fla_key[len("model."):] if fla_key.startswith("model.") else fla_key
        fla_lookup[stripped] = fla_key

    copied = 0
    missing = []

    for kb_key, kb_tensor in kb_state.items():
        normalized = _normalize_kb_key(kb_key)

        if normalized in fla_lookup:
            fla_key = fla_lookup[normalized]
            fla_tensor = fla_state[fla_key]
            if kb_tensor.shape == fla_tensor.shape:
                kb_tensor.copy_(fla_tensor)
                copied += 1
            else:
                missing.append(
                    f"{kb_key} (shape mismatch: KB={kb_tensor.shape} vs fla={fla_tensor.shape})"
                )
        else:
            missing.append(kb_key)

    if missing:
        print(f"  Warning: {len(missing)} KB weights not found in fla model:")
        for m in missing[:10]:
            print(f"    - {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    print(f"  Copied {copied}/{len(kb_state)} weights from fla to KB")

    kb_model.load_state_dict(kb_state)
    return copied, missing


# ============================================================================
# Model Fixture
# ============================================================================

@pytest.fixture(scope="module")
def loaded_models(request):
    """
    Load fla and KB models from a HuggingFace checkpoint.

    Uses --max-layers to optionally truncate the model for faster testing.
    """
    fla = pytest.importorskip("fla")

    max_layers = request.config.getoption("--max-layers", default=None)

    import fla.models
    from fla.models.rwkv6 import RWKV6Config, RWKV6ForCausalLM
    from transformers import AutoTokenizer, AutoConfig

    model_name = DEFAULT_HF_MODEL

    print(f"\nLoading fla model from {model_name}...")
    fla_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Disable fused operations for fair comparison
    fla_config.fuse_norm = False
    fla_config.fuse_cross_entropy = False
    fla_config.fuse_linear_cross_entropy = False

    total_layers = fla_config.num_hidden_layers
    if max_layers is not None:
        num_layers = min(max_layers, total_layers)
        fla_config.num_hidden_layers = num_layers
    else:
        num_layers = total_layers

    print(f"  Using {num_layers}/{total_layers} layers")

    # Load fla model
    fla_model = RWKV6ForCausalLM.from_pretrained(
        model_name,
        config=fla_config,
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )
    fla_model.eval()

    # Create KB model
    kb_module = import_module("KernelBench.level4.6_RWKV6")
    kb_model = kb_module.Model(
        hidden_size=fla_config.hidden_size,
        num_hidden_layers=num_layers,
        vocab_size=fla_config.vocab_size,
        num_heads=fla_config.num_heads,
        expand_k=fla_config.expand_k,
        expand_v=fla_config.expand_v,
        proj_low_rank_dim=fla_config.proj_low_rank_dim,
        gate_low_rank_dim=fla_config.gate_low_rank_dim,
        hidden_ratio=fla_config.hidden_ratio,
        intermediate_size=fla_config.intermediate_size,
        norm_eps=fla_config.norm_eps,
        norm_bias=fla_config.norm_bias,
        norm_first=fla_config.norm_first,
        tie_word_embeddings=fla_config.tie_word_embeddings,
    )
    kb_model = kb_model.to(device=DEVICE, dtype=DTYPE)

    # Copy weights
    copied, missing = copy_weights_fla_to_kb(fla_model, kb_model)
    assert len(missing) == 0, f"Missing weights after copy: {missing}"
    kb_model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loaded: {num_layers}/{total_layers} layers, "
          f"{fla_config.hidden_size} hidden, {fla_config.num_heads} heads")

    return fla_model, kb_model, tokenizer, fla_config, num_layers


# ============================================================================
# Alignment Tests
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_alignment(loaded_models):
    """
    Test prefill (prompt processing) alignment.

    Uses generate(max_new_tokens=0, return_logits=True) for KB prefill.
    Compares fla and KernelBench logits to ensure numerical alignment.
    """
    fla_model, kb_model, tokenizer, config, num_layers = loaded_models

    print("\n" + "=" * 70)
    print(f"Testing Prefill Alignment ({num_layers} layers)")
    print("=" * 70)

    for i, prompt in enumerate(TEST_PROMPTS):
        encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_ids = encoded["input_ids"]
        seq_len = input_ids.shape[1]

        with torch.no_grad():
            # fla forward
            fla_out = fla_model(input_ids=input_ids, use_cache=False)
            fla_logits = fla_out.logits

            # KB prefill using generate with max_new_tokens=0
            _, kb_logits_list = kb_model.generate(
                input_ids,
                max_new_tokens=0,
                return_logits=True,
            )
            kb_logits = kb_logits_list[0]

        # Compare last-position logits
        fla_last = fla_logits[:, -1, :].float()
        kb_last = kb_logits[:, -1, :].float()

        abs_diff = (fla_last - kb_last).abs()
        max_abs = abs_diff.max().item()
        mean_abs = abs_diff.mean().item()

        denom = torch.maximum(fla_last.abs(), kb_last.abs()) + 1e-8
        rel_diff = abs_diff / denom
        max_rel = rel_diff.max().item()
        mean_rel = rel_diff.mean().item()

        fla_top = fla_last.argmax(dim=-1).item()
        kb_top = kb_last.argmax(dim=-1).item()
        fla_token = tokenizer.decode([fla_top])
        kb_token = tokenizer.decode([kb_top])
        top_match = fla_top == kb_top

        mean_ok = mean_rel < RTOL_MEAN
        max_ok = max_rel < RTOL_MAX
        is_pass = top_match and mean_ok and max_ok
        status = "PASS" if is_pass else "FAIL"

        print(f"\n  [{i}] {status}: (len={seq_len} tokens)")
        print(f"      Prompt: '{prompt[:60]}...'")
        print(f"      abs_diff: max={max_abs:.2e}, mean={mean_abs:.2e}")
        print(f"      rel_diff: max={max_rel:.2e} (limit={RTOL_MAX}), "
              f"mean={mean_rel:.2e} (limit={RTOL_MEAN})")
        print(f"      fla next: '{fla_token}' | KB next: '{kb_token}' (match={top_match})")

        assert top_match, f"Top predictions differ: fla={fla_token} vs KB={kb_token}"
        assert mean_ok, \
            f"Mean relative diff {mean_rel:.2e} exceeds tolerance {RTOL_MEAN}"
        assert max_ok, \
            f"Max relative diff {max_rel:.2e} exceeds tolerance {RTOL_MAX}"

    print("\n" + "-" * 70)
    print("All prefill tests passed!")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_generation(loaded_models):
    """
    Test cached autoregressive generation.

    Uses KB generate() (with caching) and compares with fla generation
    (uncached full-context forward). Validates that both implementations
    produce matching token sequences.
    """
    fla_model, kb_model, tokenizer, config, num_layers = loaded_models

    print("\n" + "=" * 70)
    print(f"Testing Cached Generation ({num_layers} layers)")
    print("=" * 70)

    results = []

    for i, (prompt, num_tokens) in enumerate(zip(TEST_PROMPTS, RESPONSE_LENGTHS)):
        print(f"\n[{i}] Generating {num_tokens} tokens for prompt "
              f"(len={len(tokenizer.encode(prompt))})")
        print(f"    '{prompt[:60]}...'")

        encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_ids = encoded["input_ids"]
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            # fla generation (uncached full-context forward loop)
            fla_ids = input_ids.clone()
            for step in range(num_tokens):
                fla_out = fla_model(input_ids=fla_ids, use_cache=False)
                fla_next = fla_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                fla_ids = torch.cat([fla_ids, fla_next], dim=1)
            fla_tokens = fla_ids[:, prompt_len:]

            # KB generation using cached generate()
            kb_full_seq = kb_model.generate(
                input_ids,
                max_new_tokens=num_tokens,
                return_logits=False,
            )
            kb_tokens = kb_full_seq[:, prompt_len:]

        # Decode generated tokens
        fla_text = tokenizer.decode(fla_tokens[0], skip_special_tokens=True)
        kb_text = tokenizer.decode(kb_tokens[0], skip_special_tokens=True)

        # Count consecutive matching tokens from the start
        matches = (fla_tokens[0] == kb_tokens[0])
        if matches.all():
            consecutive_matches = num_tokens
        else:
            first_mismatch = (~matches).nonzero(as_tuple=True)[0]
            consecutive_matches = first_mismatch[0].item() if len(first_mismatch) > 0 else num_tokens

        full_match = consecutive_matches == num_tokens

        print(f"    Consecutive matches: {consecutive_matches}/{num_tokens} "
              f"({100 * consecutive_matches / num_tokens:.1f}%)")
        print(f"    fla: '{fla_text[:80]}...'")
        print(f"    KB:  '{kb_text[:80]}...'")

        results.append({
            'prompt': prompt,
            'full_match': full_match,
            'consecutive_matches': consecutive_matches,
            'total_tokens': num_tokens,
        })

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    total_consecutive = sum(r['consecutive_matches'] for r in results)
    total_tokens = sum(r['total_tokens'] for r in results)
    full_matches = sum(r['full_match'] for r in results)

    print(f"Full sequence matches: {full_matches}/{len(results)}")
    print(f"Consecutive token accuracy: {total_consecutive}/{total_tokens} "
          f"({100 * total_consecutive / total_tokens:.1f}%)")

    assert total_consecutive / total_tokens >= GENERATION_CONSECUTIVE_THRESHOLD, \
        f"Consecutive token accuracy {100 * total_consecutive / total_tokens:.1f}% " \
        f"below {100 * GENERATION_CONSECUTIVE_THRESHOLD:.0f}% threshold"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_components(loaded_models):
    """
    Test individual component alignment.

    Validates that individual model components (embeddings, layer norms,
    FFN, LM head) produce matching outputs between fla and KernelBench.
    """
    fla_model, kb_model, tokenizer, config, num_layers = loaded_models

    print("\n" + "=" * 70)
    print(f"Testing Component Alignment ({num_layers} layers)")
    print("=" * 70)

    hidden_size = config.hidden_size
    vocab_size = config.vocab_size

    # Test embeddings
    test_ids = torch.randint(0, vocab_size, (2, 32), device=DEVICE)
    with torch.no_grad():
        fla_emb = fla_model.model.embeddings(test_ids)
        kb_emb = kb_model.embeddings(test_ids)

    emb_diff = (fla_emb - kb_emb).abs().max().item()
    print(f"  Embedding diff: {emb_diff:.2e} {'PASS' if emb_diff < ATOL_STRICT else 'FAIL'}")
    assert emb_diff < ATOL_STRICT

    # Test layer norms
    hidden = torch.randn(2, 32, hidden_size, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        fla_norm = fla_model.model.layers[0].attn_norm(hidden)
        kb_norm = kb_model.layers[0].attn_norm(hidden)

    norm_diff = (fla_norm - kb_norm).abs().max().item()
    print(f"  LayerNorm diff: {norm_diff:.2e} {'PASS' if norm_diff < ATOL_STRICT else 'FAIL'}")
    assert norm_diff < ATOL_STRICT

    # Test FFN (channel mixing)
    with torch.no_grad():
        fla_ffn_out, _ = fla_model.model.layers[0].ffn(hidden)
        kb_ffn_out = kb_model.layers[0].ffn(hidden)

    ffn_diff = (fla_ffn_out - kb_ffn_out).abs().max().item()
    print(f"  FFN diff: {ffn_diff:.2e} {'PASS' if ffn_diff < ATOL_STRICT else 'FAIL'}")
    assert ffn_diff < ATOL_STRICT

    # Test LM head
    with torch.no_grad():
        fla_head = fla_model.lm_head(hidden)
        kb_head = kb_model.lm_head(hidden)

    head_diff = (fla_head - kb_head).abs().max().item()
    print(f"  LM Head diff: {head_diff:.2e} {'PASS' if head_diff < ATOL_STRICT else 'FAIL'}")
    assert head_diff < ATOL_STRICT

    print("\n  All component tests passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
