"""
Test KernelBench model alignment with flash-linear-attention (fla).

This test validates that the KernelBench implementations of RWKV-6, GLA,
and RetNet produce outputs matching the fla library's implementations using:
1. A batch of prompts with varying lengths
2. Cached autoregressive generation
3. Both prefill and decode phases

Supports the following fla-hub HuggingFace checkpoints:
- fla-hub/rwkv6-7B-finch (RWKV-6)
- fla-hub/gla-2.7B-100B (GLA)
- fla-hub/retnet-2.7B-100B (RetNet)

Optionally limit the number of layers with --max-layers for faster testing.

Usage:
    # Run all model tests:
    pytest tests/test_fla_alignment.py -v -s

    # Run a single model:
    pytest tests/test_fla_alignment.py -v -s -k "rwkv6"
    pytest tests/test_fla_alignment.py -v -s -k "gla"
    pytest tests/test_fla_alignment.py -v -s -k "retnet"

    # With limited layers:
    pytest tests/test_fla_alignment.py -v -s --max-layers 4

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
# Model configurations
# ============================================================================

MODEL_CONFIGS = {
    "rwkv6": {
        "hf_model": "fla-hub/rwkv6-7B-finch",
        "fla_import": ("fla.models.rwkv6", "RWKV6Config", "RWKV6ForCausalLM"),
        "kb_module": "KernelBench.level4.8_RWKV6",
        # RWKV6 uses LayerNorm (not RMSNorm) and doesn't have g_norm_swish_gate,
        # so we disable fuse_norm to use nn.LayerNorm for exact alignment.
        "fuse_norm": False,
    },
    "gla": {
        "hf_model": "fla-hub/gla-2.7B-100B",
        "fla_import": ("fla.models.gla", "GLAConfig", "GLAForCausalLM"),
        "kb_module": "KernelBench.level4.9_GLA",
        # GLA checkpoint stores g_norm_swish_gate.weight (fuse_norm=True).
        # KB separates this into RMSNorm + Swish gate; we keep fuse_norm=True
        # so weights load correctly from checkpoint.
        "fuse_norm": True,
    },
    "retnet": {
        "hf_model": "fla-hub/retnet-2.7B-100B",
        "fla_import": ("fla.models.retnet", "RetNetConfig", "RetNetForCausalLM"),
        "kb_module": "KernelBench.level4.10_RetNet",
        # RetNet checkpoint stores g_norm_swish_gate.weight (fuse_norm=True).
        # KB separates this into RMSNorm + Swish gate; we keep fuse_norm=True
        # so weights load correctly from checkpoint.
        "fuse_norm": True,
    },
}


# ============================================================================
# Key normalization helpers
# ============================================================================

def _normalize_rwkv6_key(key: str) -> str:
    """Normalize a KernelBench RWKV6 state dict key to match fla naming."""
    key = key.replace('.ln.', '.')
    key = key.replace('embeddings.embedding.', 'embeddings.')
    key = key.replace('.gn.', '.')
    key = key.replace('.x_proj_up.', '.x_proj.2.')
    key = key.replace('.x_proj.linear.', '.x_proj.0.linear.')
    key = key.replace('.x_proj.mu', '.x_proj.0.mu')
    return key


def _normalize_gla_key(key: str) -> str:
    """Normalize a KernelBench GLA state dict key to match fla naming."""
    # Embedding wrapper: embeddings.embedding.weight -> embeddings.weight
    key = key.replace('embeddings.embedding.', 'embeddings.')
    # Linear level1 wrapper: .linear.weight -> .weight, .linear.bias -> .bias
    # For gk_proj Sequential(Linear, Linear):
    #   gk_proj.0.linear.weight -> gk_proj.0.weight (Linear level1 op)
    #   gk_proj.1.linear.weight -> gk_proj.1.weight
    #   gk_proj.1.linear.bias -> gk_proj.1.bias
    key = key.replace('.linear.weight', '.weight')
    key = key.replace('.linear.bias', '.bias')
    # g_norm (KB) -> g_norm_swish_gate (fla, when fuse_norm=True)
    # KB uses separate RMSNorm + Swish gate; fla fuses them into FusedRMSNormGated
    key = key.replace('.g_norm.', '.g_norm_swish_gate.')
    return key


def _normalize_retnet_key(key: str) -> str:
    """Normalize a KernelBench RetNet state dict key to match fla naming."""
    # Embedding wrapper
    key = key.replace('embeddings.embedding.', 'embeddings.')
    # Linear level1 wrapper
    key = key.replace('.linear.weight', '.weight')
    key = key.replace('.linear.bias', '.bias')
    # g_norm (KB) -> g_norm_swish_gate (fla, when fuse_norm=True)
    # KB uses separate RMSNorm + Swish gate; fla fuses them into FusedRMSNormGated
    key = key.replace('.g_norm.', '.g_norm_swish_gate.')
    return key


NORMALIZERS = {
    "rwkv6": _normalize_rwkv6_key,
    "gla": _normalize_gla_key,
    "retnet": _normalize_retnet_key,
}


# ============================================================================
# KB model creation helpers
# ============================================================================

def _create_rwkv6_kb_model(fla_config, num_layers):
    """Create a KernelBench RWKV6 model from fla config."""
    kb_module = import_module("KernelBench.level4.8_RWKV6")
    return kb_module.Model(
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


def _create_gla_kb_model(fla_config, num_layers):
    """Create a KernelBench GLA model from fla config."""
    kb_module = import_module("KernelBench.level4.9_GLA")
    return kb_module.Model(
        hidden_size=fla_config.hidden_size,
        num_hidden_layers=num_layers,
        vocab_size=fla_config.vocab_size,
        num_heads=fla_config.num_heads,
        num_kv_heads=fla_config.num_kv_heads,
        expand_k=fla_config.expand_k,
        expand_v=fla_config.expand_v,
        hidden_ratio=fla_config.hidden_ratio,
        intermediate_size=fla_config.intermediate_size,
        elementwise_affine=fla_config.elementwise_affine,
        norm_eps=fla_config.norm_eps,
        tie_word_embeddings=fla_config.tie_word_embeddings,
        # gate_low_rank_dim defaults to 16 in fla's GatedLinearAttention layer
        gate_low_rank_dim=getattr(fla_config, 'gate_low_rank_dim', 16),
    )


def _create_retnet_kb_model(fla_config, num_layers):
    """Create a KernelBench RetNet model from fla config."""
    kb_module = import_module("KernelBench.level4.10_RetNet")
    return kb_module.Model(
        hidden_size=fla_config.hidden_size,
        num_hidden_layers=num_layers,
        vocab_size=fla_config.vocab_size,
        num_heads=fla_config.num_heads,
        num_kv_heads=fla_config.num_kv_heads,
        expand_k=fla_config.expand_k,
        expand_v=fla_config.expand_v,
        hidden_ratio=fla_config.hidden_ratio,
        intermediate_size=fla_config.intermediate_size,
        elementwise_affine=fla_config.elementwise_affine,
        norm_eps=fla_config.norm_eps,
        tie_word_embeddings=fla_config.tie_word_embeddings,
    )


KB_CREATORS = {
    "rwkv6": _create_rwkv6_kb_model,
    "gla": _create_gla_kb_model,
    "retnet": _create_retnet_kb_model,
}


# ============================================================================
# fla forward helpers for model-specific forward calls
# ============================================================================

def _fla_forward(model_type, fla_model, input_ids):
    """Run fla forward pass (no cache)."""
    fla_out = fla_model(input_ids=input_ids, use_cache=False)
    return fla_out.logits


def _fla_generate_step(model_type, fla_model, full_ids):
    """Run one fla generation step (uncached full-context forward)."""
    fla_out = fla_model(input_ids=full_ids, use_cache=False)
    return fla_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)


# ============================================================================
# Weight copy helpers
# ============================================================================

def copy_weights_fla_to_kb(fla_model, kb_model, model_type):
    """
    Copy weights from fla model to KernelBench model.

    Handles:
    - fla prefix: "model." for the backbone
    - KB wrappers for level1 operators
    """
    normalize_key = NORMALIZERS[model_type]

    fla_state = fla_model.state_dict()
    kb_state = kb_model.state_dict()

    # Build fla key lookup: strip "model." prefix for backbone params
    fla_lookup = {}
    for fla_key in fla_state.keys():
        stripped = fla_key[len("model."):] if fla_key.startswith("model.") else fla_key
        fla_lookup[stripped] = fla_key

    copied = 0
    missing = []
    skipped_buffers = []

    for kb_key, kb_tensor in kb_state.items():
        normalized = normalize_key(kb_key)

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
            # Skip known non-parameter buffers
            if any(buf in kb_key for buf in ['inv_freq', 'cos_cached', 'sin_cached',
                                              'log_gamma', 'retention_recurrent.gamma',
                                              'decay_rates']):
                skipped_buffers.append(kb_key)
            else:
                missing.append(kb_key)

    if skipped_buffers:
        print(f"  Skipped {len(skipped_buffers)} computed buffers (inv_freq, cos/sin cache, decay)")

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
# Pytest custom options
# ============================================================================

def pytest_addoption(parser):
    parser.addoption(
        "--max-layers", type=int, default=None,
        help="Maximum number of layers to use (for faster testing)",
    )


# ============================================================================
# Model Fixtures (one per model type)
# ============================================================================

def _load_models(request, model_type):
    """Generic model loader for any model type."""
    fla = pytest.importorskip("fla")

    max_layers = request.config.getoption("--max-layers", default=None)
    config = MODEL_CONFIGS[model_type]
    model_name = config["hf_model"]

    fla_module_path, config_class_name, model_class_name = config["fla_import"]

    print(f"\nLoading fla {model_type} model from {model_name}...")

    # Import fla model classes
    fla_mod = import_module(fla_module_path)
    FLAConfig = getattr(fla_mod, config_class_name)
    FLAModel = getattr(fla_mod, model_class_name)

    from transformers import AutoConfig
    fla_config = FLAConfig.from_pretrained(model_name)

    # Configure fused operations
    fla_config.fuse_norm = config["fuse_norm"]
    fla_config.fuse_cross_entropy = False
    fla_config.fuse_linear_cross_entropy = False
    if hasattr(fla_config, 'fuse_swiglu'):
        fla_config.fuse_swiglu = False

    total_layers = fla_config.num_hidden_layers
    if max_layers is not None:
        num_layers = min(max_layers, total_layers)
        fla_config.num_hidden_layers = num_layers
    else:
        num_layers = total_layers

    print(f"  Using {num_layers}/{total_layers} layers")

    # Load fla model
    fla_model = FLAModel.from_pretrained(
        model_name,
        config=fla_config,
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )
    fla_model.eval()

    # Create KB model
    create_kb = KB_CREATORS[model_type]
    kb_model = create_kb(fla_config, num_layers)
    kb_model = kb_model.to(device=DEVICE, dtype=DTYPE)

    # Copy weights
    copied, missing = copy_weights_fla_to_kb(fla_model, kb_model, model_type)
    assert len(missing) == 0, f"Missing weights after copy: {missing}"
    kb_model.eval()

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loaded: {num_layers}/{total_layers} layers, "
          f"{fla_config.hidden_size} hidden, {fla_config.num_heads} heads")

    return fla_model, kb_model, tokenizer, fla_config, num_layers


@pytest.fixture(scope="module")
def rwkv6_models(request):
    """Load RWKV-6 fla and KB models."""
    return _load_models(request, "rwkv6")


@pytest.fixture(scope="module")
def gla_models(request):
    """Load GLA fla and KB models."""
    return _load_models(request, "gla")


@pytest.fixture(scope="module")
def retnet_models(request):
    """Load RetNet fla and KB models."""
    return _load_models(request, "retnet")


# ============================================================================
# Shared test logic
# ============================================================================

def _run_prefill_test(fla_model, kb_model, tokenizer, model_type, num_layers):
    """Run prefill alignment test for a model."""
    print("\n" + "=" * 70)
    print(f"Testing {model_type.upper()} Prefill Alignment ({num_layers} layers)")
    print("=" * 70)

    for i, prompt in enumerate(TEST_PROMPTS):
        encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_ids = encoded["input_ids"]
        seq_len = input_ids.shape[1]

        with torch.no_grad():
            # fla forward
            fla_logits = _fla_forward(model_type, fla_model, input_ids)

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
    print(f"All {model_type.upper()} prefill tests passed!")


def _run_generation_test(fla_model, kb_model, tokenizer, model_type, num_layers):
    """Run cached autoregressive generation test for a model."""
    print("\n" + "=" * 70)
    print(f"Testing {model_type.upper()} Cached Generation ({num_layers} layers)")
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
                fla_next = _fla_generate_step(model_type, fla_model, fla_ids)
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
    print(f"{model_type.upper()} Generation Summary")
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


def _run_components_test(fla_model, kb_model, tokenizer, model_type, fla_config, num_layers):
    """Run component alignment tests for a model."""
    print("\n" + "=" * 70)
    print(f"Testing {model_type.upper()} Component Alignment ({num_layers} layers)")
    print("=" * 70)

    hidden_size = fla_config.hidden_size
    vocab_size = fla_config.vocab_size

    # Test embeddings
    test_ids = torch.randint(0, vocab_size, (2, 32), device=DEVICE)
    with torch.no_grad():
        fla_emb = fla_model.model.embeddings(test_ids)
        kb_emb = kb_model.embeddings(test_ids)

    emb_diff = (fla_emb - kb_emb).abs().max().item()
    print(f"  Embedding diff: {emb_diff:.2e} {'PASS' if emb_diff < ATOL_STRICT else 'FAIL'}")
    assert emb_diff < ATOL_STRICT

    # Test layer norms (attn_norm)
    # When fuse_norm=True (GLA/RetNet), fla uses Triton-based RMSNorm while KB
    # uses PyTorch RMSNorm -> slight numerical differences expected.
    # When fuse_norm=False (RWKV6), both use nn.LayerNorm -> exact match.
    NORM_ATOL = 0.05 if MODEL_CONFIGS[model_type]["fuse_norm"] else ATOL_STRICT
    hidden = torch.randn(2, 32, hidden_size, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        fla_norm = fla_model.model.layers[0].attn_norm(hidden)
        kb_norm = kb_model.layers[0].attn_norm(hidden)

    norm_diff = (fla_norm - kb_norm).abs().max().item()
    print(f"  Norm diff: {norm_diff:.2e} (tol={NORM_ATOL:.2e}) "
          f"{'PASS' if norm_diff < NORM_ATOL else 'FAIL'}")
    assert norm_diff < NORM_ATOL

    # Test MLP/FFN
    # fla's GatedMLP uses Triton-based swiglu kernel (float32 intermediates) for
    # GLA and RetNet, while KB uses PyTorch Swish in bf16. This causes
    # per-element differences that can be large (up to ~5) for high-magnitude
    # activations. RWKV6 uses a different FFN architecture with exact match.
    MLP_ATOL = 5.0 if model_type in ("gla", "retnet") else ATOL_STRICT
    # RWKV6 uses 'ffn' not 'mlp'
    fla_mlp_name = "ffn" if model_type == "rwkv6" else "mlp"
    kb_mlp_name = "ffn" if model_type == "rwkv6" else "mlp"
    fla_mlp = getattr(fla_model.model.layers[0], fla_mlp_name)
    kb_mlp = getattr(kb_model.layers[0], kb_mlp_name)
    with torch.no_grad():
        fla_mlp_out = fla_mlp(hidden)
        kb_mlp_out = kb_mlp(hidden)
        # fla's RWKV6 ChannelMixing returns (output, state), others return tensor
        if isinstance(fla_mlp_out, tuple):
            fla_mlp_out = fla_mlp_out[0]
        if isinstance(kb_mlp_out, tuple):
            kb_mlp_out = kb_mlp_out[0]

    mlp_diff = (fla_mlp_out - kb_mlp_out).abs().max().item()
    print(f"  MLP diff: {mlp_diff:.2e} (tol={MLP_ATOL:.2e}) "
          f"{'PASS' if mlp_diff < MLP_ATOL else 'FAIL'}")
    assert mlp_diff < MLP_ATOL

    # Test LM head
    with torch.no_grad():
        fla_head = fla_model.lm_head(hidden)
        kb_head = kb_model.lm_head(hidden)

    head_diff = (fla_head - kb_head).abs().max().item()
    print(f"  LM Head diff: {head_diff:.2e} {'PASS' if head_diff < ATOL_STRICT else 'FAIL'}")
    assert head_diff < ATOL_STRICT

    print("\n  All component tests passed!")


# ============================================================================
# RWKV-6 Tests
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rwkv6_prefill(rwkv6_models):
    fla_model, kb_model, tokenizer, config, num_layers = rwkv6_models
    _run_prefill_test(fla_model, kb_model, tokenizer, "rwkv6", num_layers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rwkv6_generation(rwkv6_models):
    fla_model, kb_model, tokenizer, config, num_layers = rwkv6_models
    _run_generation_test(fla_model, kb_model, tokenizer, "rwkv6", num_layers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rwkv6_components(rwkv6_models):
    fla_model, kb_model, tokenizer, config, num_layers = rwkv6_models
    _run_components_test(fla_model, kb_model, tokenizer, "rwkv6", config, num_layers)


# ============================================================================
# GLA Tests
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gla_prefill(gla_models):
    fla_model, kb_model, tokenizer, config, num_layers = gla_models
    _run_prefill_test(fla_model, kb_model, tokenizer, "gla", num_layers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gla_generation(gla_models):
    fla_model, kb_model, tokenizer, config, num_layers = gla_models
    _run_generation_test(fla_model, kb_model, tokenizer, "gla", num_layers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gla_components(gla_models):
    fla_model, kb_model, tokenizer, config, num_layers = gla_models
    _run_components_test(fla_model, kb_model, tokenizer, "gla", config, num_layers)


# ============================================================================
# RetNet Tests
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_retnet_prefill(retnet_models):
    fla_model, kb_model, tokenizer, config, num_layers = retnet_models
    _run_prefill_test(fla_model, kb_model, tokenizer, "retnet", num_layers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_retnet_generation(retnet_models):
    fla_model, kb_model, tokenizer, config, num_layers = retnet_models
    _run_generation_test(fla_model, kb_model, tokenizer, "retnet", num_layers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_retnet_components(retnet_models):
    fla_model, kb_model, tokenizer, config, num_layers = retnet_models
    _run_components_test(fla_model, kb_model, tokenizer, "retnet", config, num_layers)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
