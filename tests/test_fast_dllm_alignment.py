"""
Test Fast-dLLM alignment with the official NVlabs/Fast-dLLM repository.

This test validates that our KernelBench Fast-dLLM implementation produces outputs
matching the official Fast-dLLM library (https://github.com/NVlabs/Fast-dLLM):
1. LLaDA backbone: forward pass logits alignment with the reference model
2. Generation utilities: add_gumbel_noise, get_num_transfer_tokens, get_transfer_index
3. Full generation: generate, generate_with_prefix_cache, generate_with_dual_cache

The reference model is GSAI-ML/LLaDA-8B-Instruct loaded via the reference's
LLaDAModelLM.from_pretrained() with trust_remote_code=True.

Usage:
    pytest tests/test_fast_dllm_alignment.py -v -s

Requires:
    - The Fast-dLLM repo at Fast-dLLM/ in the workspace root
    - GPU with sufficient memory (~16GB for 8B model in bfloat16)
    - GSAI-ML/LLaDA-8B-Instruct model weights
"""

import pytest
import torch
import torch.nn.functional as F
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
REPO_ROOT = os.path.join(TEST_DIR, "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "KernelBench"))
sys.path.insert(0, os.path.join(REPO_ROOT, "Fast-dLLM", "llada"))

# Skip all tests if required packages are not available
transformers = pytest.importorskip("transformers")
from transformers import AutoTokenizer

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
MASK_ID = 126336

# Test prompts
TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
]

# Tolerances
# Backbone logits: both implementations produce bit-identical results when
# using the same weights, inputs, and computation order. The RoPE cos/sin
# cache is computed on CPU (matching the reference) and SDPA uses the same
# Flash Attention backend. Tolerances are set to near-zero.
ATOL_LOGITS = 1e-6
RTOL_LOGITS = 1e-6
ATOL_UTILS = 1e-7
RTOL_UTILS = 1e-7


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def tokenizer():
    """Load the LLaDA tokenizer."""
    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


@pytest.fixture(scope="module")
def ref_model():
    """Load the reference LLaDA model from the Fast-dLLM repo."""
    from model.modeling_llada import LLaDAModelLM
    model = LLaDAModelLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE,
    ).to(DEVICE).eval()
    return model


@pytest.fixture(scope="module")
def kb_model(ref_model):
    """Create KernelBench Fast-dLLM model with weights from the reference."""
    kb_mod = importlib.import_module("KernelBench.level4.36_FastDLLM")
    KBModel = kb_mod.Model

    config = kb_mod.VARIANTS["LLaDA-8B-Instruct"].copy()
    config.pop("model_name", None)
    model = KBModel(**config)

    _load_weights_from_reference(model, ref_model)
    model = model.to(dtype=DTYPE, device=DEVICE).eval()
    return model


@pytest.fixture(scope="module")
def ref_generate_module():
    """Import the reference generate module."""
    import generate as gen_mod
    return gen_mod


@pytest.fixture(scope="module")
def kb_generate_module():
    """Import the KernelBench generate functions."""
    return importlib.import_module("KernelBench.level4.36_FastDLLM")


def _load_weights_from_reference(kb_model, ref_model):
    """
    Copy weights from reference LLaDAModelLM to KernelBench Model.

    Reference naming (LLaDALlamaBlock):
        model.transformer.wte          -> kb: embed_tokens
        model.transformer.blocks.i.attn_norm  -> kb: layers.i.input_layernorm
        model.transformer.blocks.i.q_proj     -> kb: layers.i.self_attn.q_proj
        model.transformer.blocks.i.k_proj     -> kb: layers.i.self_attn.k_proj
        model.transformer.blocks.i.v_proj     -> kb: layers.i.self_attn.v_proj
        model.transformer.blocks.i.attn_out   -> kb: layers.i.self_attn.o_proj
        model.transformer.blocks.i.ff_norm    -> kb: layers.i.post_attention_layernorm
        model.transformer.blocks.i.ff_proj    -> kb: layers.i.mlp.gate_proj
        model.transformer.blocks.i.up_proj    -> kb: layers.i.mlp.up_proj
        model.transformer.blocks.i.ff_out     -> kb: layers.i.mlp.down_proj
        model.transformer.ln_f                -> kb: norm
        model.transformer.ff_out              -> kb: lm_head
    """
    ref = ref_model.model  # LLaDAModel inside LLaDAModelLM

    # Embedding
    kb_model.embed_tokens.weight.data.copy_(ref.transformer.wte.weight.data)

    # Transformer blocks
    n_layers = len(kb_model.layers)
    ref_blocks = ref.transformer.blocks
    assert len(ref_blocks) == n_layers, (
        f"Layer count mismatch: reference has {len(ref_blocks)}, KB has {n_layers}"
    )

    for i in range(n_layers):
        ref_blk = ref_blocks[i]
        kb_layer = kb_model.layers[i]

        # Input layernorm
        kb_layer.input_layernorm.weight.data.copy_(ref_blk.attn_norm.weight.data)

        # Attention projections
        kb_layer.self_attn.q_proj.weight.data.copy_(ref_blk.q_proj.weight.data)
        kb_layer.self_attn.k_proj.weight.data.copy_(ref_blk.k_proj.weight.data)
        kb_layer.self_attn.v_proj.weight.data.copy_(ref_blk.v_proj.weight.data)
        kb_layer.self_attn.o_proj.weight.data.copy_(ref_blk.attn_out.weight.data)

        # Post-attention layernorm
        kb_layer.post_attention_layernorm.weight.data.copy_(ref_blk.ff_norm.weight.data)

        # MLP projections
        kb_layer.mlp.gate_proj.weight.data.copy_(ref_blk.ff_proj.weight.data)
        kb_layer.mlp.up_proj.weight.data.copy_(ref_blk.up_proj.weight.data)
        kb_layer.mlp.down_proj.weight.data.copy_(ref_blk.ff_out.weight.data)

    # Final layernorm
    kb_model.norm.weight.data.copy_(ref.transformer.ln_f.weight.data)

    # LM head
    if hasattr(ref.transformer, "ff_out") and ref.transformer.ff_out is not None:
        if kb_model.lm_head is not None:
            kb_model.lm_head.weight.data.copy_(ref.transformer.ff_out.weight.data)


def _format_prompt(prompt: str, tokenizer) -> torch.Tensor:
    """Format prompt using LLaDA instruct template and tokenize."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    input_ids = tokenizer(formatted)["input_ids"]
    return torch.tensor(input_ids, device=DEVICE).unsqueeze(0)


# ============================================================================
# Test Class 1: LLaDA Backbone Alignment
# ============================================================================

class TestLLaDABackboneAlignment:
    """Verify that our LLaDA backbone produces the same logits as the reference."""

    def test_forward_logits_match(self, ref_model, kb_model, tokenizer):
        """Given a tokenized prompt with MASK tokens appended, compare logits."""
        for prompt_text in TEST_PROMPTS:
            input_ids = _format_prompt(prompt_text, tokenizer)
            gen_length = 32

            mask_tokens = torch.full(
                (1, gen_length), MASK_ID, dtype=torch.long, device=DEVICE,
            )
            full_input = torch.cat([input_ids, mask_tokens], dim=1)

            with torch.no_grad():
                ref_out = ref_model(full_input)
                kb_out = kb_model(full_input)

            ref_logits = ref_out.logits
            kb_logits = kb_out.logits

            assert ref_logits.shape == kb_logits.shape, (
                f"Shape mismatch: ref={ref_logits.shape}, kb={kb_logits.shape}"
            )

            max_diff = (ref_logits.float() - kb_logits.float()).abs().max().item()
            print(f"  Prompt: {prompt_text[:40]}... | "
                  f"Logits max diff: {max_diff:.6e}")

            torch.testing.assert_close(
                kb_logits.float(), ref_logits.float(),
                atol=ATOL_LOGITS, rtol=RTOL_LOGITS,
                msg=f"Logits mismatch for prompt: {prompt_text[:40]}...",
            )

    def test_forward_with_various_mask_patterns(self, ref_model, kb_model, tokenizer):
        """Test with different mask patterns: all masked, partially masked, no masked."""
        prompt_text = TEST_PROMPTS[0]
        input_ids = _format_prompt(prompt_text, tokenizer)
        gen_length = 16

        patterns = {
            "all_masked": torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=DEVICE),
            "half_masked": torch.cat([
                torch.randint(0, 1000, (1, gen_length // 2), device=DEVICE),
                torch.full((1, gen_length // 2), MASK_ID, dtype=torch.long, device=DEVICE),
            ], dim=1),
            "no_masked": torch.randint(0, 1000, (1, gen_length), device=DEVICE),
        }

        for pattern_name, suffix in patterns.items():
            full_input = torch.cat([input_ids, suffix], dim=1)

            with torch.no_grad():
                ref_out = ref_model(full_input)
                kb_out = kb_model(full_input)

            max_diff = (ref_out.logits.float() - kb_out.logits.float()).abs().max().item()
            print(f"  Pattern: {pattern_name} | Max diff: {max_diff:.6e}")

            torch.testing.assert_close(
                kb_out.logits.float(), ref_out.logits.float(),
                atol=ATOL_LOGITS, rtol=RTOL_LOGITS,
                msg=f"Logits mismatch for pattern: {pattern_name}",
            )


# ============================================================================
# Test Class 2: Generation Utilities Alignment
# ============================================================================

class TestGenerationUtilsAlignment:
    """Verify generation helper functions match the reference."""

    def test_get_num_transfer_tokens(self, ref_generate_module, kb_generate_module):
        """Compare output for various mask counts and step counts."""
        ref_fn = ref_generate_module.get_num_transfer_tokens
        kb_fn = kb_generate_module.get_num_transfer_tokens

        test_cases = [
            (torch.tensor([[True, True, True, True, True, True, True, True]], device=DEVICE), 4),
            (torch.tensor([[True, True, True, False, False, True, True, True]], device=DEVICE), 3),
            (torch.tensor([[True] * 32], device=DEVICE), 8),
            (torch.tensor([[True] * 7], device=DEVICE), 3),
            (torch.tensor([[False] * 16], device=DEVICE), 4),
        ]

        for block_mask, steps in test_cases:
            ref_result = ref_fn(block_mask, steps)
            kb_result = kb_fn(block_mask, steps)

            torch.testing.assert_close(
                kb_result, ref_result,
                atol=0, rtol=0,
                msg=f"get_num_transfer_tokens mismatch for mask sum={block_mask.sum().item()}, steps={steps}",
            )
            print(f"  mask_sum={block_mask.sum().item():2d}, steps={steps} -> {ref_result.tolist()} [OK]")

    def test_add_gumbel_noise_greedy(self, ref_generate_module, kb_generate_module):
        """Verify that temperature=0 returns logits unchanged."""
        ref_fn = ref_generate_module.add_gumbel_noise
        kb_fn = kb_generate_module.add_gumbel_noise

        logits = torch.randn(1, 16, 100, device=DEVICE)

        ref_result = ref_fn(logits, temperature=0)
        kb_result = kb_fn(logits, temperature=0)

        torch.testing.assert_close(
            kb_result, ref_result, atol=0, rtol=0,
            msg="add_gumbel_noise(temperature=0) should return logits unchanged",
        )

    def test_add_gumbel_noise_with_temperature(self, ref_generate_module, kb_generate_module):
        """Verify identical output with same seed."""
        ref_fn = ref_generate_module.add_gumbel_noise
        kb_fn = kb_generate_module.add_gumbel_noise

        logits = torch.randn(1, 16, 100, device=DEVICE)

        torch.manual_seed(42)
        ref_result = ref_fn(logits, temperature=0.5)
        torch.manual_seed(42)
        kb_result = kb_fn(logits, temperature=0.5)

        torch.testing.assert_close(
            kb_result, ref_result,
            atol=ATOL_UTILS, rtol=RTOL_UTILS,
            msg="add_gumbel_noise mismatch with same seed",
        )

    def test_get_transfer_index_low_confidence(self, ref_generate_module, kb_generate_module):
        """Compare which tokens get unmasked given identical logits/masks."""
        ref_fn = ref_generate_module.get_transfer_index
        kb_fn = kb_generate_module.get_transfer_index

        B, L, V = 1, 32, 100
        logits = torch.randn(B, L, V, device=DEVICE, dtype=torch.float32)
        mask_index = torch.zeros(B, L, device=DEVICE, dtype=torch.bool)
        mask_index[0, 16:] = True
        x = torch.randint(0, V, (B, L), device=DEVICE)
        x[mask_index] = MASK_ID
        num_transfer = torch.tensor([4], device=DEVICE)

        ref_x0, ref_transfer = ref_fn(logits, 0.0, "low_confidence", mask_index, x, num_transfer)
        kb_x0, kb_transfer = kb_fn(logits, 0.0, "low_confidence", mask_index, x, num_transfer)

        torch.testing.assert_close(
            kb_x0, ref_x0, atol=0, rtol=0,
            msg="get_transfer_index x0 mismatch",
        )
        assert (kb_transfer == ref_transfer).all(), "get_transfer_index transfer_index mismatch"
        print(f"  Transferred {ref_transfer.sum().item()} tokens [OK]")

    def test_get_transfer_index_with_threshold(self, ref_generate_module, kb_generate_module):
        """Test threshold-based transfer."""
        ref_fn = ref_generate_module.get_transfer_index
        kb_fn = kb_generate_module.get_transfer_index

        B, L, V = 1, 32, 100
        logits = torch.randn(B, L, V, device=DEVICE, dtype=torch.float32)
        mask_index = torch.zeros(B, L, device=DEVICE, dtype=torch.bool)
        mask_index[0, 16:] = True
        x = torch.randint(0, V, (B, L), device=DEVICE)
        x[mask_index] = MASK_ID

        ref_x0, ref_transfer = ref_fn(logits, 0.0, "low_confidence", mask_index, x, None, threshold=0.5)
        kb_x0, kb_transfer = kb_fn(logits, 0.0, "low_confidence", mask_index, x, None, threshold=0.5)

        torch.testing.assert_close(
            kb_x0, ref_x0, atol=0, rtol=0,
            msg="get_transfer_index (threshold) x0 mismatch",
        )
        assert (kb_transfer == ref_transfer).all(), "get_transfer_index (threshold) transfer_index mismatch"
        print(f"  Threshold=0.5: transferred {ref_transfer.sum().item()} tokens [OK]")

    def test_get_transfer_index_random_remasking(self, ref_generate_module, kb_generate_module):
        """Test random remasking strategy."""
        ref_fn = ref_generate_module.get_transfer_index
        kb_fn = kb_generate_module.get_transfer_index

        B, L, V = 1, 32, 100
        logits = torch.randn(B, L, V, device=DEVICE, dtype=torch.float32)
        mask_index = torch.ones(B, L, device=DEVICE, dtype=torch.bool)
        x = torch.full((B, L), MASK_ID, device=DEVICE)
        num_transfer = torch.tensor([8], device=DEVICE)

        torch.manual_seed(123)
        ref_x0, ref_transfer = ref_fn(logits, 0.0, "random", mask_index, x, num_transfer)
        torch.manual_seed(123)
        kb_x0, kb_transfer = kb_fn(logits, 0.0, "random", mask_index, x, num_transfer)

        torch.testing.assert_close(
            kb_x0, ref_x0, atol=0, rtol=0,
            msg="get_transfer_index (random) x0 mismatch",
        )
        assert (kb_transfer == ref_transfer).all(), "get_transfer_index (random) transfer_index mismatch"
        print(f"  Random remasking: transferred {ref_transfer.sum().item()} tokens [OK]")


# ============================================================================
# Test Class 3: Full Generation Alignment
# ============================================================================

class TestFastDLLMGenerationAlignment:
    """Verify full generation loop alignment."""

    def test_generate_vanilla(self, ref_model, kb_model, tokenizer,
                              ref_generate_module, kb_generate_module):
        """Run generate() on both, compare output tokens and NFE."""
        prompt_text = TEST_PROMPTS[0]
        input_ids = _format_prompt(prompt_text, tokenizer)

        gen_length = 64
        steps = 64
        block_length = 32

        with torch.no_grad():
            torch.manual_seed(0)
            ref_x, ref_nfe = ref_generate_module.generate(
                ref_model, input_ids,
                steps=steps, gen_length=gen_length,
                block_length=block_length, temperature=0.0,
                remasking="low_confidence", mask_id=MASK_ID,
            )

            torch.manual_seed(0)
            kb_x, kb_nfe = kb_generate_module.generate(
                kb_model, input_ids,
                steps=steps, gen_length=gen_length,
                block_length=block_length, temperature=0.0,
                remasking="low_confidence", mask_id=MASK_ID,
            )

        ref_gen = ref_x[:, input_ids.shape[1]:]
        kb_gen = kb_x[:, input_ids.shape[1]:]

        ref_text = tokenizer.decode(ref_gen[0], skip_special_tokens=True)
        kb_text = tokenizer.decode(kb_gen[0], skip_special_tokens=True)

        print(f"\n  Prompt: {prompt_text}")
        print(f"  Reference output: {ref_text[:100]}...")
        print(f"  KernelBench output: {kb_text[:100]}...")
        print(f"  NFE: ref={ref_nfe}, kb={kb_nfe}")

        match_rate = (ref_gen == kb_gen).float().mean().item()
        print(f"  Token match rate: {match_rate:.2%}")

        assert ref_nfe == kb_nfe, f"NFE mismatch: ref={ref_nfe}, kb={kb_nfe}"
        assert match_rate > 0.9, f"Token match rate too low: {match_rate:.2%}"

    def test_generate_with_prefix_cache(self, ref_model, kb_model, tokenizer,
                                         ref_generate_module, kb_generate_module):
        """Run generate_with_prefix_cache() on both, compare output tokens."""
        prompt_text = TEST_PROMPTS[0]
        input_ids = _format_prompt(prompt_text, tokenizer)

        gen_length = 64
        steps = 64
        block_length = 32

        with torch.no_grad():
            torch.manual_seed(0)
            ref_x, ref_nfe = ref_generate_module.generate_with_prefix_cache(
                ref_model, input_ids,
                steps=steps, gen_length=gen_length,
                block_length=block_length, temperature=0.0,
                remasking="low_confidence", mask_id=MASK_ID,
            )

            torch.manual_seed(0)
            kb_x, kb_nfe = kb_generate_module.generate_with_prefix_cache(
                kb_model, input_ids,
                steps=steps, gen_length=gen_length,
                block_length=block_length, temperature=0.0,
                remasking="low_confidence", mask_id=MASK_ID,
            )

        ref_gen = ref_x[:, input_ids.shape[1]:]
        kb_gen = kb_x[:, input_ids.shape[1]:]

        ref_text = tokenizer.decode(ref_gen[0], skip_special_tokens=True)
        kb_text = tokenizer.decode(kb_gen[0], skip_special_tokens=True)

        print(f"\n  Prompt: {prompt_text}")
        print(f"  Reference output: {ref_text[:100]}...")
        print(f"  KernelBench output: {kb_text[:100]}...")
        print(f"  NFE: ref={ref_nfe}, kb={kb_nfe}")

        match_rate = (ref_gen == kb_gen).float().mean().item()
        print(f"  Token match rate: {match_rate:.2%}")

        assert ref_nfe == kb_nfe, f"NFE mismatch: ref={ref_nfe}, kb={kb_nfe}"
        assert match_rate > 0.9, f"Token match rate too low: {match_rate:.2%}"

    def test_generate_with_dual_cache(self, ref_model, kb_model, tokenizer,
                                       ref_generate_module, kb_generate_module):
        """Run generate_with_dual_cache() on both, compare output tokens."""
        prompt_text = TEST_PROMPTS[0]
        input_ids = _format_prompt(prompt_text, tokenizer)

        gen_length = 64
        steps = 64
        block_length = 32

        with torch.no_grad():
            torch.manual_seed(0)
            ref_x, ref_nfe = ref_generate_module.generate_with_dual_cache(
                ref_model, input_ids,
                steps=steps, gen_length=gen_length,
                block_length=block_length, temperature=0.0,
                remasking="low_confidence", mask_id=MASK_ID,
            )

            torch.manual_seed(0)
            kb_x, kb_nfe = kb_generate_module.generate_with_dual_cache(
                kb_model, input_ids,
                steps=steps, gen_length=gen_length,
                block_length=block_length, temperature=0.0,
                remasking="low_confidence", mask_id=MASK_ID,
            )

        ref_gen = ref_x[:, input_ids.shape[1]:]
        kb_gen = kb_x[:, input_ids.shape[1]:]

        ref_text = tokenizer.decode(ref_gen[0], skip_special_tokens=True)
        kb_text = tokenizer.decode(kb_gen[0], skip_special_tokens=True)

        print(f"\n  Prompt: {prompt_text}")
        print(f"  Reference output: {ref_text[:100]}...")
        print(f"  KernelBench output: {kb_text[:100]}...")
        print(f"  NFE: ref={ref_nfe}, kb={kb_nfe}")

        match_rate = (ref_gen == kb_gen).float().mean().item()
        print(f"  Token match rate: {match_rate:.2%}")

        assert ref_nfe == kb_nfe, f"NFE mismatch: ref={ref_nfe}, kb={kb_nfe}"
        assert match_rate > 0.9, f"Token match rate too low: {match_rate:.2%}"

    def test_vanilla_vs_prefix_cache_consistency(self, ref_model, tokenizer,
                                                  ref_generate_module):
        """The reference generate and generate_with_prefix_cache should produce same tokens."""
        prompt_text = TEST_PROMPTS[0]
        input_ids = _format_prompt(prompt_text, tokenizer)

        gen_length = 64
        steps = 64
        block_length = 32

        with torch.no_grad():
            torch.manual_seed(0)
            vanilla_x, vanilla_nfe = ref_generate_module.generate(
                ref_model, input_ids,
                steps=steps, gen_length=gen_length,
                block_length=block_length, temperature=0.0,
                remasking="low_confidence", mask_id=MASK_ID,
            )

            torch.manual_seed(0)
            cache_x, cache_nfe = ref_generate_module.generate_with_prefix_cache(
                ref_model, input_ids,
                steps=steps, gen_length=gen_length,
                block_length=block_length, temperature=0.0,
                remasking="low_confidence", mask_id=MASK_ID,
            )

        vanilla_gen = vanilla_x[:, input_ids.shape[1]:]
        cache_gen = cache_x[:, input_ids.shape[1]:]
        match_rate = (vanilla_gen == cache_gen).float().mean().item()

        print(f"\n  Vanilla vs Prefix Cache consistency: {match_rate:.2%}")
        print(f"  NFE: vanilla={vanilla_nfe}, cache={cache_nfe}")
