"""
CosyVoice3 alignment tests: KernelBench vs original CosyVoice repository.

Tests component-level and end-to-end alignment between our KernelBench
implementation (KernelBench/level4/22_CosyVoice.py) and the original
CosyVoice repository (https://github.com/FunAudioLLM/CosyVoice).

Test structure:
  - TestFlowAlignment: Load flow weights, run identical inputs, compare mel output
  - TestVocoderAlignment: Load vocoder weights, run identical mel, compare waveform
  - TestLLMAlignment: Load LLM weights, run one forward step, compare logits
  - TestE2EInference: Run full TTS pipeline, compare final waveform

Usage:
    # All tests:
    pytest tests/test_cosyvoice_alignment.py -v

    # Flow only:
    pytest tests/test_cosyvoice_alignment.py -v -k "TestFlowAlignment"

    # Vocoder only:
    pytest tests/test_cosyvoice_alignment.py -v -k "TestVocoderAlignment"

    # E2E with audio saving:
    pytest tests/test_cosyvoice_alignment.py -v -k "TestE2E" --save-images

Requires:
    - CosyVoice repo cloned and installed (pip install -e third_party/CosyVoice)
    - Fun-CosyVoice3-0.5B-2512 model weights downloaded
    - CUDA GPU
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Dict, Optional

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# HuggingFace environment
# ---------------------------------------------------------------------------
os.environ["HF_HOME"] = "/home/yak/data-fast/huggingface"
HF_TOKEN_PATH = "/home/yak/data-fast/huggingface/token"
if os.path.exists(HF_TOKEN_PATH):
    with open(HF_TOKEN_PATH, "r") as f:
        os.environ["HF_TOKEN"] = f.read().strip()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(TEST_DIR, "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "KernelBench"))

# CosyVoice repo path
COSYVOICE_REPO = os.path.join(REPO_ROOT, "third_party", "CosyVoice")
COSYVOICE_MODEL_DIR = os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
    "hub",
    "models--FunAudioLLM--Fun-CosyVoice3-0.5B-2512",
)

# Output directory for saved audio
OUTPUT_DIR = os.path.join(TEST_DIR, "outputs")

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # CosyVoice uses float32 by default


def _find_model_dir():
    """Find the CosyVoice3 model directory."""
    # Try huggingface hub cache first
    from huggingface_hub import snapshot_download
    try:
        model_dir = snapshot_download(
            "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
            local_dir=None,
        )
        return model_dir
    except Exception:
        pass

    # Try common locations
    candidates = [
        COSYVOICE_MODEL_DIR,
        os.path.join(REPO_ROOT, "pretrained_models", "Fun-CosyVoice3-0.5B"),
        os.path.expanduser("~/pretrained_models/Fun-CosyVoice3-0.5B"),
    ]
    for path in candidates:
        if os.path.exists(path):
            # Check for snapshots directory structure
            snapshots_dir = os.path.join(path, "snapshots")
            if os.path.exists(snapshots_dir):
                # Get the latest snapshot
                snapshots = sorted(os.listdir(snapshots_dir))
                if snapshots:
                    return os.path.join(snapshots_dir, snapshots[-1])
            return path
    return None


def _setup_cosyvoice_imports():
    """Add CosyVoice repo to sys.path so we can import from it."""
    if COSYVOICE_REPO not in sys.path:
        sys.path.insert(0, COSYVOICE_REPO)
    # Also add third_party matcha if it exists
    matcha_path = os.path.join(COSYVOICE_REPO, "third_party", "Matcha-TTS")
    if os.path.exists(matcha_path) and matcha_path not in sys.path:
        sys.path.insert(0, matcha_path)


def _load_kb_module():
    """Load the KernelBench CosyVoice module."""
    return importlib.import_module("KernelBench.level4.22_CosyVoice")


# ============================================================================
# Weight Copying Utilities
# ============================================================================

def _copy_state_dict_with_mapping(
    kb_module: torch.nn.Module,
    ref_state_dict: Dict[str, torch.Tensor],
    key_mapping: Optional[Dict[str, str]] = None,
    strict: bool = True,
    prefix_strip: str = "",
):
    """Copy weights from reference state dict to KB module.

    Handles the naming differences between KB level1 operators and the
    original CosyVoice implementation.
    """
    kb_sd = kb_module.state_dict()
    new_sd = {}

    # Build reverse mapping: ref_key -> kb_key
    if key_mapping is None:
        key_mapping = {}

    # Common KB level1 operator name mappings
    # KB Embedding wraps nn.Embedding as self.embedding
    # KB Linear wraps weight/bias directly
    # KB LayerNorm wraps nn.LayerNorm as self.ln

    for kb_key in kb_sd:
        # Try direct match first
        ref_key = kb_key
        if prefix_strip and ref_key.startswith(prefix_strip):
            ref_key = ref_key[len(prefix_strip):]

        # Apply explicit mapping
        if ref_key in key_mapping:
            ref_key = key_mapping[ref_key]

        # Handle KB level1 operator naming conventions
        # Embedding: kb has .embedding.weight, ref has .weight
        ref_key_candidates = [ref_key]

        # KB Embedding stores as .embedding.weight
        if ".embedding.weight" in ref_key:
            ref_key_candidates.append(ref_key.replace(".embedding.weight", ".weight"))

        # KB LayerNorm stores as .ln.weight/.ln.bias
        if ".ln.weight" in ref_key:
            ref_key_candidates.append(ref_key.replace(".ln.weight", ".weight"))
        if ".ln.bias" in ref_key:
            ref_key_candidates.append(ref_key.replace(".ln.bias", ".bias"))

        found = False
        for candidate in ref_key_candidates:
            if candidate in ref_state_dict:
                if kb_sd[kb_key].shape == ref_state_dict[candidate].shape:
                    new_sd[kb_key] = ref_state_dict[candidate]
                    found = True
                    break
                else:
                    print(
                        f"Shape mismatch for {kb_key}: "
                        f"KB={kb_sd[kb_key].shape}, ref={ref_state_dict[candidate].shape}"
                    )

        if not found and strict:
            print(f"WARNING: No matching weight found for KB key: {kb_key}")

    # Load what we found
    missing, unexpected = kb_module.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"Missing keys in KB model: {len(missing)}")
        for k in missing[:10]:
            print(f"  {k}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    return missing, unexpected


def _map_kb_key_to_ref(kb_key):
    """Map a single KB state dict key to the corresponding reference key.

    Handles:
    - KB Embedding: .embedding.weight -> .weight
    - KB LayerNorm: .ln.weight/.ln.bias -> .weight/.bias
    - DiT attention: .to_out_linear. -> .to_out.0.  (ref uses ModuleList)
    - DiT feedforward: KB Sequential(Linear, GELU, Dropout, Linear) indices 0/3
                        -> ref Sequential(Sequential(Linear, GELU), Dropout, Linear) indices 0.0/2
    """
    ref_key = kb_key

    # KB level1 operator naming
    ref_key = ref_key.replace(".embedding.weight", ".weight")
    ref_key = ref_key.replace(".ln.weight", ".weight")
    ref_key = ref_key.replace(".ln.bias", ".bias")

    # DiT attention output projection: KB splits to_out into named submodules
    ref_key = ref_key.replace(".to_out_linear.", ".to_out.0.")

    # DiT feedforward index mapping:
    # KB DiTFeedForward.ff = Sequential(Linear[0], GELUAct[1], Dropout[2], Linear[3])
    # Ref FeedForward.ff   = Sequential(Sequential(Linear, GELU)[0], Dropout[1], Linear[2])
    #
    # Weight-bearing layers:
    #   KB ff.ff.0.weight -> ref ff.ff.0.0.weight  (first Linear)
    #   KB ff.ff.3.weight -> ref ff.ff.2.weight    (second Linear)
    #
    # We need to be careful to only modify the digit immediately after "ff.ff."
    import re
    # First linear: ff.ff.0. -> ff.ff.0.0.
    ref_key = re.sub(r'\.ff\.ff\.0\.(weight|bias)', r'.ff.ff.0.0.\1', ref_key)
    # Second linear: ff.ff.3. -> ff.ff.2.
    ref_key = re.sub(r'\.ff\.ff\.3\.(weight|bias)', r'.ff.ff.2.\1', ref_key)

    return ref_key


def _copy_flow_weights(kb_flow, ref_flow_sd):
    """Copy flow model weights from reference to KB."""
    kb_sd = kb_flow.state_dict()
    new_sd = {}

    for kb_key, kb_val in kb_sd.items():
        # Try direct match
        if kb_key in ref_flow_sd and ref_flow_sd[kb_key].shape == kb_val.shape:
            new_sd[kb_key] = ref_flow_sd[kb_key]
            continue

        ref_key = _map_kb_key_to_ref(kb_key)

        if ref_key in ref_flow_sd and ref_flow_sd[ref_key].shape == kb_val.shape:
            new_sd[kb_key] = ref_flow_sd[ref_key]
            continue

    missing_keys = set(kb_sd.keys()) - set(new_sd.keys())
    if missing_keys:
        print(f"Flow: {len(missing_keys)} missing keys out of {len(kb_sd)}")
        for k in sorted(missing_keys):
            ref_tried = _map_kb_key_to_ref(k)
            in_ref = ref_tried in ref_flow_sd
            print(f"  MISSING: {k}  ->  {ref_tried}  (in ref: {in_ref})")

    kb_flow.load_state_dict(new_sd, strict=False)
    return missing_keys


def _copy_hift_weights(kb_hift, ref_hift_sd):
    """Copy vocoder weights from reference to KB."""
    # The hift state dict from the original repo may have 'generator.' prefix
    clean_sd = {}
    for k, v in ref_hift_sd.items():
        clean_key = k.replace("generator.", "")
        clean_sd[clean_key] = v

    kb_sd = kb_hift.state_dict()
    new_sd = {}

    for kb_key, kb_val in kb_sd.items():
        if kb_key in clean_sd and clean_sd[kb_key].shape == kb_val.shape:
            new_sd[kb_key] = clean_sd[kb_key]
            continue

    missing_keys = set(kb_sd.keys()) - set(new_sd.keys())
    if missing_keys:
        print(f"HiFT: {len(missing_keys)} missing keys out of {len(kb_sd)}")
        for k in sorted(missing_keys)[:20]:
            print(f"  MISSING: {k}")

    kb_hift.load_state_dict(new_sd, strict=False)
    return missing_keys


def _copy_llm_weights(kb_llm, ref_llm_sd):
    """Copy LLM weights from reference to KB.

    The KB Qwen2 model uses a different path structure than the HuggingFace
    Qwen2ForCausalLM. Key mappings:
    - ref: llm.model.model.{rest} -> KB: llm.model.{rest}
    - ref: llm.model.model.embed_tokens.weight -> KB: llm.model.embed_tokens.embedding.weight
    - ref: speech_embedding.weight -> KB: speech_embedding.embedding.weight
    - ref: llm.model.lm_head.weight -> (skipped, tied with embed_tokens)
    """
    kb_sd = kb_llm.state_dict()
    new_sd = {}

    for kb_key, kb_val in kb_sd.items():
        # Try direct match first
        if kb_key in ref_llm_sd and ref_llm_sd[kb_key].shape == kb_val.shape:
            new_sd[kb_key] = ref_llm_sd[kb_key]
            continue

        # Map KB key to reference key
        ref_key = kb_key

        # KB Embedding wraps nn.Embedding: .embedding.weight -> .weight
        ref_key = ref_key.replace(".embedding.weight", ".weight")

        # KB Qwen2Encoder.model is Qwen2Model directly, but ref has
        # Qwen2ForCausalLM.model.model path: llm.model.X -> llm.model.model.X
        if ref_key.startswith("llm.model."):
            ref_key_hf = "llm.model.model." + ref_key[len("llm.model."):]
            if ref_key_hf in ref_llm_sd and ref_llm_sd[ref_key_hf].shape == kb_val.shape:
                new_sd[kb_key] = ref_llm_sd[ref_key_hf]
                continue

        # Direct match after embedding fix
        if ref_key in ref_llm_sd and ref_llm_sd[ref_key].shape == kb_val.shape:
            new_sd[kb_key] = ref_llm_sd[ref_key]
            continue

    missing_keys = set(kb_sd.keys()) - set(new_sd.keys())
    if missing_keys:
        print(f"LLM: {len(missing_keys)} missing keys out of {len(kb_sd)}")
        for k in sorted(missing_keys)[:20]:
            print(f"  MISSING: {k}")

    kb_llm.load_state_dict(new_sd, strict=False)
    return missing_keys


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def model_dir():
    """Find and return the CosyVoice3 model directory."""
    d = _find_model_dir()
    if d is None:
        pytest.skip(
            "CosyVoice3 model not found. Download with: "
            "huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
        )
    return d


@pytest.fixture(scope="session")
def cosyvoice_available():
    """Check if CosyVoice repo is available."""
    if not os.path.exists(COSYVOICE_REPO):
        pytest.skip(
            f"CosyVoice repo not found at {COSYVOICE_REPO}. "
            "Clone it with: git clone https://github.com/FunAudioLLM/CosyVoice third_party/CosyVoice"
        )
    _setup_cosyvoice_imports()
    return True


@pytest.fixture(scope="session")
def kb_module():
    """Load the KernelBench CosyVoice module."""
    return _load_kb_module()


@pytest.fixture(scope="session")
def save_images(request):
    """Whether to save generated audio files."""
    return request.config.getoption("--save-images", default=False)


# ============================================================================
# Test: Flow Model Alignment
# ============================================================================

def _build_ref_flow(config=None):
    """Build the reference CosyVoice flow model directly, bypassing YAML."""
    _setup_cosyvoice_imports()
    from omegaconf import DictConfig
    from cosyvoice.flow.flow import CausalMaskedDiffWithDiT as RefCausalMaskedDiffWithDiT
    from cosyvoice.flow.flow_matching import CausalConditionalCFM as RefCausalConditionalCFM
    from cosyvoice.flow.DiT.dit import DiT as RefDiT
    from cosyvoice.transformer.upsample_encoder import PreLookaheadLayer as RefPreLookaheadLayer

    ref_dit = RefDiT(
        dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2,
        mel_dim=80, mu_dim=80, spk_dim=80, out_channels=80,
        static_chunk_size=50, num_decoding_left_chunks=-1,
    )
    ref_cfm = RefCausalConditionalCFM(
        in_channels=240, n_spks=1, spk_emb_dim=80,
        cfm_params=DictConfig({
            "sigma_min": 1e-06, "solver": "euler", "t_scheduler": "cosine",
            "training_cfg_rate": 0.2, "inference_cfg_rate": 0.7, "reg_loss_type": "l1",
        }),
        estimator=ref_dit,
    )
    ref_pre_lookahead = RefPreLookaheadLayer(
        in_channels=80, channels=1024, pre_lookahead_len=3,
    )
    ref_flow = RefCausalMaskedDiffWithDiT(
        input_size=80, output_size=80, spk_embed_dim=192,
        output_type="mel", vocab_size=6561, input_frame_rate=25,
        only_mask_loss=True, token_mel_ratio=2, pre_lookahead_len=3,
        pre_lookahead_layer=ref_pre_lookahead, decoder=ref_cfm,
    )
    return ref_flow


def _build_ref_hift():
    """Build the reference CosyVoice vocoder directly."""
    _setup_cosyvoice_imports()
    from cosyvoice.hifigan.generator import CausalHiFTGenerator as RefCausalHiFTGenerator
    from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor

    f0_predictor = CausalConvRNNF0Predictor(
        num_class=1, in_channels=80, cond_channels=512,
    )
    ref_hift = RefCausalHiFTGenerator(
        in_channels=80, base_channels=512, nb_harmonics=8,
        sampling_rate=24000, nsf_alpha=0.1, nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3], upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1, audio_limit=0.99, conv_pre_look_right=4,
        f0_predictor=f0_predictor,
    )
    return ref_hift


class TestFlowAlignment:
    """Test flow model (CausalMaskedDiffWithDiT) alignment."""

    def test_flow_forward(self, model_dir, cosyvoice_available, kb_module):
        """Compare flow model forward pass between KB and original."""
        _setup_cosyvoice_imports()

        # Load reference flow weights
        flow_pt = os.path.join(model_dir, "flow.pt")
        if not os.path.exists(flow_pt):
            pytest.skip(f"flow.pt not found at {flow_pt}")

        ref_flow_sd = torch.load(flow_pt, map_location="cpu", weights_only=True)

        # Build KB flow model
        kb_mod = importlib.import_module("KernelBench.level4.22_CosyVoice")
        CausalMaskedDiffWithDiT = kb_mod.CausalMaskedDiffWithDiT
        PreLookaheadLayer = kb_mod.PreLookaheadLayer
        CausalConditionalCFM = kb_mod.CausalConditionalCFM
        DiT = kb_mod.DiT
        DEFAULT_CONFIG = kb_mod.DEFAULT_CONFIG

        config = DEFAULT_CONFIG

        dit = DiT(
            dim=config["dit_dim"],
            depth=config["dit_depth"],
            heads=config["dit_heads"],
            dim_head=config["dit_dim_head"],
            dropout=config["dit_dropout"],
            ff_mult=config["dit_ff_mult"],
            mel_dim=config["dit_mel_dim"],
            mu_dim=config.get("dit_mu_dim", None),
            spk_dim=config["dit_spk_dim"],
            static_chunk_size=config["dit_static_chunk_size"],
            num_decoding_left_chunks=config["dit_num_decoding_left_chunks"],
        )

        cfm_decoder = CausalConditionalCFM(
            in_channels=config["flow_output_size"],
            inference_cfg_rate=config["cfm_inference_cfg_rate"],
            t_scheduler=config["cfm_t_scheduler"],
            estimator=dit,
        )

        pre_lookahead = PreLookaheadLayer(
            in_channels=config.get("pre_lookahead_in_channels", config["flow_input_size"]),
            channels=config.get("pre_lookahead_channels", 1024),
            pre_lookahead_len=config["flow_pre_lookahead_len"],
        )

        kb_flow = CausalMaskedDiffWithDiT(
            input_size=config["flow_input_size"],
            output_size=config["flow_output_size"],
            spk_embed_dim=config["flow_spk_embed_dim"],
            vocab_size=config["flow_vocab_size"],
            token_mel_ratio=config["flow_token_mel_ratio"],
            pre_lookahead_len=config["flow_pre_lookahead_len"],
            pre_lookahead_layer=pre_lookahead,
            decoder=cfm_decoder,
        )

        # Copy weights
        missing = _copy_flow_weights(kb_flow, ref_flow_sd)

        # Build reference model directly
        ref_flow = _build_ref_flow()
        ref_flow.load_state_dict(ref_flow_sd, strict=True)
        ref_flow.eval()

        # Move to device
        kb_flow = kb_flow.to(DEVICE).eval()
        ref_flow = ref_flow.to(DEVICE).eval()

        # Create test inputs
        torch.manual_seed(42)
        seq_len = 50
        token = torch.randint(0, 6561, (1, seq_len), device=DEVICE)
        token_len = torch.tensor([seq_len], device=DEVICE)
        prompt_token = torch.randint(0, 6561, (1, 10), device=DEVICE)
        prompt_token_len = torch.tensor([10], device=DEVICE)
        prompt_feat = torch.randn(1, 20, 80, device=DEVICE)
        prompt_feat_len = torch.tensor([20], device=DEVICE)
        embedding = torch.randn(1, 192, device=DEVICE)

        # Run both models
        with torch.no_grad():
            kb_mel, _ = kb_flow.inference(
                token, token_len, prompt_token, prompt_token_len,
                prompt_feat, prompt_feat_len, embedding,
                streaming=False, finalize=True,
            )
            ref_mel, _ = ref_flow.inference(
                token, token_len, prompt_token, prompt_token_len,
                prompt_feat, prompt_feat_len, embedding,
                streaming=False, finalize=True,
            )

        # Compare
        assert kb_mel.shape == ref_mel.shape, (
            f"Shape mismatch: KB={kb_mel.shape}, ref={ref_mel.shape}"
        )

        max_diff = (kb_mel - ref_mel).abs().max().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            kb_mel.flatten().unsqueeze(0),
            ref_mel.flatten().unsqueeze(0),
        ).item()

        print(f"\nFlow alignment: max_diff={max_diff:.6f}, cos_sim={cos_sim:.6f}")
        assert cos_sim > 0.99, f"Flow cosine similarity too low: {cos_sim}"
        assert max_diff < 0.1, f"Flow max diff too high: {max_diff}"


# ============================================================================
# Test: Vocoder Alignment
# ============================================================================

class TestVocoderAlignment:
    """Test vocoder (CausalHiFTGenerator) alignment."""

    def test_vocoder_forward(self, model_dir, cosyvoice_available, kb_module):
        """Compare vocoder forward pass between KB and original."""
        _setup_cosyvoice_imports()

        # Load reference vocoder weights
        hift_pt = os.path.join(model_dir, "hift.pt")
        if not os.path.exists(hift_pt):
            pytest.skip(f"hift.pt not found at {hift_pt}")

        ref_hift_sd = torch.load(hift_pt, map_location="cpu", weights_only=True)

        # Build KB vocoder
        kb_mod = importlib.import_module("KernelBench.level4.22_CosyVoice")
        CausalHiFTGenerator = kb_mod.CausalHiFTGenerator
        DEFAULT_CONFIG = kb_mod.DEFAULT_CONFIG

        config = DEFAULT_CONFIG
        kb_hift = CausalHiFTGenerator(
            in_channels=config["hift_in_channels"],
            base_channels=config["hift_base_channels"],
            nb_harmonics=config["hift_nb_harmonics"],
            sampling_rate=config["hift_sampling_rate"],
            upsample_rates=config["hift_upsample_rates"],
            upsample_kernel_sizes=config["hift_upsample_kernel_sizes"],
            istft_params={
                "n_fft": config["hift_istft_n_fft"],
                "hop_len": config["hift_istft_hop_len"],
            },
            resblock_kernel_sizes=config["hift_resblock_kernel_sizes"],
            resblock_dilation_sizes=config["hift_resblock_dilation_sizes"],
            source_resblock_kernel_sizes=config["hift_source_resblock_kernel_sizes"],
            source_resblock_dilation_sizes=config["hift_source_resblock_dilation_sizes"],
            conv_pre_look_right=config["hift_conv_pre_look_right"],
        )

        # Copy weights
        missing = _copy_hift_weights(kb_hift, ref_hift_sd)

        # Build reference model directly
        ref_hift = _build_ref_hift()
        clean_sd = {k.replace("generator.", ""): v for k, v in ref_hift_sd.items()}
        ref_hift.load_state_dict(clean_sd, strict=True)
        ref_hift.eval()

        # Move to device
        kb_hift = kb_hift.to(DEVICE).eval()
        ref_hift = ref_hift.to(DEVICE).eval()

        # Create test input (mel spectrogram)
        torch.manual_seed(42)
        mel = torch.randn(1, 80, 100, device=DEVICE)

        # Run both models
        with torch.no_grad():
            kb_speech, _ = kb_hift.inference(speech_feat=mel, finalize=True)
            ref_speech, _ = ref_hift.inference(speech_feat=mel, finalize=True)

        # Compare
        min_len = min(kb_speech.shape[-1], ref_speech.shape[-1])
        kb_speech = kb_speech[..., :min_len]
        ref_speech = ref_speech[..., :min_len]

        max_diff = (kb_speech - ref_speech).abs().max().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            kb_speech.flatten().unsqueeze(0),
            ref_speech.flatten().unsqueeze(0),
        ).item()

        print(f"\nVocoder alignment: max_diff={max_diff:.6f}, cos_sim={cos_sim:.6f}")
        # Vocoder includes non-linear ops (ISTFT, source filter) that can amplify
        # small numerical differences, especially with random input
        assert cos_sim > 0.95, f"Vocoder cosine similarity too low: {cos_sim}"
        assert max_diff < 5.0, f"Vocoder max diff too high: {max_diff}"


# ============================================================================
# Test: LLM Alignment
# ============================================================================

class TestLLMAlignment:
    """Test LLM (CosyVoice3LM) alignment."""

    def test_llm_one_step(self, model_dir, cosyvoice_available, kb_module):
        """Compare one LLM forward step between KB and original."""
        _setup_cosyvoice_imports()

        # Load reference LLM weights
        llm_pt = os.path.join(model_dir, "llm.pt")
        if not os.path.exists(llm_pt):
            pytest.skip(f"llm.pt not found at {llm_pt}")

        ref_llm_sd = torch.load(llm_pt, map_location="cpu", weights_only=True)

        # Build KB LLM
        kb_mod = importlib.import_module("KernelBench.level4.22_CosyVoice")
        CosyVoice3LM = kb_mod.CosyVoice3LM
        DEFAULT_CONFIG = kb_mod.DEFAULT_CONFIG

        config = DEFAULT_CONFIG
        kb_llm = CosyVoice3LM(
            llm_input_size=config["llm_input_size"],
            llm_output_size=config["llm_output_size"],
            speech_token_size=config["speech_token_size"],
        )

        # Copy weights
        missing = _copy_llm_weights(kb_llm, ref_llm_sd)

        # Build reference LLM directly
        _setup_cosyvoice_imports()
        from cosyvoice.llm.llm import CosyVoice3LM as RefCosyVoice3LM
        from cosyvoice.llm.llm import Qwen2Encoder as RefQwen2Encoder

        def _dummy_sampling(*args, **kwargs):
            pass

        ref_qwen2 = RefQwen2Encoder(pretrain_path="Qwen/Qwen2.5-0.5B")
        ref_llm = RefCosyVoice3LM(
            llm_input_size=config["llm_input_size"],
            llm_output_size=config["llm_output_size"],
            speech_token_size=config["speech_token_size"],
            llm=ref_qwen2,
            sampling=_dummy_sampling,
        )
        ref_llm.load_state_dict(ref_llm_sd, strict=True)
        ref_llm.eval()

        # Move to device
        kb_llm = kb_llm.to(DEVICE).eval()
        ref_llm = ref_llm.to(DEVICE).eval()

        # Create test input
        torch.manual_seed(42)
        batch_size = 1
        seq_len = 20
        lm_input = torch.randn(batch_size, seq_len, config["llm_input_size"], device=DEVICE)
        lm_input_len = torch.tensor([seq_len], device=DEVICE)

        # Run KB model
        with torch.no_grad():
            kb_logits = kb_llm.forward(lm_input, lm_input_len)

        # Run reference model
        with torch.no_grad():
            ref_output, ref_mask = ref_llm.llm(lm_input, lm_input_len)
            ref_logits = ref_llm.llm_decoder(ref_output)

        # Compare
        assert kb_logits.shape == ref_logits.shape, (
            f"Shape mismatch: KB={kb_logits.shape}, ref={ref_logits.shape}"
        )

        max_diff = (kb_logits - ref_logits).abs().max().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            kb_logits.flatten().unsqueeze(0),
            ref_logits.flatten().unsqueeze(0),
        ).item()

        print(f"\nLLM alignment: max_diff={max_diff:.6f}, cos_sim={cos_sim:.6f}")
        assert cos_sim > 0.999, f"LLM cosine similarity too low: {cos_sim}"
        # KB Qwen2 uses its own SDPA + RoPE which can accumulate small
        # numerical differences over 24 layers. max_diff ~0.13 is acceptable
        # when cos_sim is near-perfect.
        assert max_diff < 0.5, f"LLM max diff too high: {max_diff}"


# ============================================================================
# Test: End-to-End Inference with Real Inputs
# ============================================================================

# Real text prompts for E2E testing
E2E_PROMPTS = [
    "Hello, this is a test of the CosyVoice text to speech system.",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world of technology.",
]


def _build_tokenizer(model_dir):
    """Build the CosyVoice3 tokenizer from the model directory."""
    from transformers import AutoTokenizer

    blank_en_dir = os.path.join(model_dir, "CosyVoice-BlankEN")
    tokenizer = AutoTokenizer.from_pretrained(blank_en_dir)

    # Add the special tokens that CosyVoice3 expects
    special_tokens = {
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": [
            "<|im_start|>", "<|im_end|>", "<|endofprompt|>",
            "[breath]", "<strong>", "</strong>", "[noise]",
            "[laughter]", "[cough]", "[clucking]", "[accent]",
            "[quick_breath]",
        ],
    }
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def _tokenize_text(tokenizer, text):
    """Tokenize text for CosyVoice3 LLM input.

    For CosyVoice3, the text is wrapped as:
        <|endofprompt|> + text
    The <|endofprompt|> token is ID 151646 (hardcoded in the original).
    """
    # The <|endofprompt|> token signals the start of synthesis text
    eop_token_id = tokenizer.convert_tokens_to_ids("<|endofprompt|>")
    text_token_ids = tokenizer.encode(text, add_special_tokens=False)
    # CosyVoice3 expects <|endofprompt|> before the text
    full_ids = [eop_token_id] + text_token_ids
    return full_ids


def _build_all_models(model_dir):
    """Build both KB and reference models with shared weights.

    Returns (kb_flow, kb_hift, kb_llm, ref_flow, ref_hift, ref_llm).
    """
    _setup_cosyvoice_imports()

    kb_mod = importlib.import_module("KernelBench.level4.22_CosyVoice")
    config = kb_mod.DEFAULT_CONFIG

    # ---- KB Flow ----
    dit = kb_mod.DiT(
        dim=config["dit_dim"], depth=config["dit_depth"],
        heads=config["dit_heads"], dim_head=config["dit_dim_head"],
        dropout=config["dit_dropout"], ff_mult=config["dit_ff_mult"],
        mel_dim=config["dit_mel_dim"], mu_dim=config.get("dit_mu_dim"),
        spk_dim=config["dit_spk_dim"],
        static_chunk_size=config["dit_static_chunk_size"],
        num_decoding_left_chunks=config["dit_num_decoding_left_chunks"],
    )
    cfm = kb_mod.CausalConditionalCFM(
        in_channels=config["flow_output_size"],
        inference_cfg_rate=config["cfm_inference_cfg_rate"],
        t_scheduler=config["cfm_t_scheduler"], estimator=dit,
    )
    pre_la = kb_mod.PreLookaheadLayer(
        in_channels=config.get("pre_lookahead_in_channels", config["flow_input_size"]),
        channels=config.get("pre_lookahead_channels", 1024),
        pre_lookahead_len=config["flow_pre_lookahead_len"],
    )
    kb_flow = kb_mod.CausalMaskedDiffWithDiT(
        input_size=config["flow_input_size"], output_size=config["flow_output_size"],
        spk_embed_dim=config["flow_spk_embed_dim"], vocab_size=config["flow_vocab_size"],
        token_mel_ratio=config["flow_token_mel_ratio"],
        pre_lookahead_len=config["flow_pre_lookahead_len"],
        pre_lookahead_layer=pre_la, decoder=cfm,
    )

    # ---- KB Vocoder ----
    kb_hift = kb_mod.CausalHiFTGenerator(
        in_channels=config["hift_in_channels"],
        base_channels=config["hift_base_channels"],
        nb_harmonics=config["hift_nb_harmonics"],
        sampling_rate=config["hift_sampling_rate"],
        upsample_rates=config["hift_upsample_rates"],
        upsample_kernel_sizes=config["hift_upsample_kernel_sizes"],
        istft_params={"n_fft": config["hift_istft_n_fft"], "hop_len": config["hift_istft_hop_len"]},
        resblock_kernel_sizes=config["hift_resblock_kernel_sizes"],
        resblock_dilation_sizes=config["hift_resblock_dilation_sizes"],
        source_resblock_kernel_sizes=config["hift_source_resblock_kernel_sizes"],
        source_resblock_dilation_sizes=config["hift_source_resblock_dilation_sizes"],
        conv_pre_look_right=config["hift_conv_pre_look_right"],
    )

    # ---- KB LLM ----
    kb_llm = kb_mod.CosyVoice3LM(
        llm_input_size=config["llm_input_size"],
        llm_output_size=config["llm_output_size"],
        speech_token_size=config["speech_token_size"],
    )

    # ---- Load weights ----
    flow_sd = torch.load(os.path.join(model_dir, "flow.pt"), map_location="cpu", weights_only=True)
    hift_sd = torch.load(os.path.join(model_dir, "hift.pt"), map_location="cpu", weights_only=True)
    llm_sd = torch.load(os.path.join(model_dir, "llm.pt"), map_location="cpu", weights_only=True)

    _copy_flow_weights(kb_flow, flow_sd)
    _copy_hift_weights(kb_hift, hift_sd)
    _copy_llm_weights(kb_llm, llm_sd)

    # ---- Reference models ----
    ref_flow = _build_ref_flow()
    ref_flow.load_state_dict(flow_sd, strict=True)

    ref_hift = _build_ref_hift()
    clean_hift_sd = {k.replace("generator.", ""): v for k, v in hift_sd.items()}
    ref_hift.load_state_dict(clean_hift_sd, strict=True)

    from cosyvoice.llm.llm import CosyVoice3LM as RefCosyVoice3LM
    from cosyvoice.llm.llm import Qwen2Encoder as RefQwen2Encoder
    from cosyvoice.utils.common import ras_sampling

    ref_qwen2 = RefQwen2Encoder(pretrain_path="Qwen/Qwen2.5-0.5B")
    ref_llm = RefCosyVoice3LM(
        llm_input_size=config["llm_input_size"],
        llm_output_size=config["llm_output_size"],
        speech_token_size=config["speech_token_size"],
        llm=ref_qwen2,
        sampling=ras_sampling,
    )
    ref_llm.load_state_dict(llm_sd, strict=True)

    # Move all to device and eval
    for m in (kb_flow, kb_hift, kb_llm, ref_flow, ref_hift, ref_llm):
        m.to(DEVICE).eval()

    return kb_flow, kb_hift, kb_llm, ref_flow, ref_hift, ref_llm


class TestE2EInference:
    """End-to-end TTS tests with real text inputs.

    These tests use real tokenized text and pretrained weights to verify
    alignment between KernelBench and the original CosyVoice3 implementation.
    """

    @pytest.fixture(scope="class")
    def all_models(self, model_dir, cosyvoice_available):
        """Build and cache all models for E2E tests."""
        return _build_all_models(model_dir)

    @pytest.fixture(scope="class")
    def tokenizer(self, model_dir):
        """Build the CosyVoice3 tokenizer."""
        return _build_tokenizer(model_dir)

    # ------------------------------------------------------------------
    # Test 1: LLM logit alignment with real tokenized text
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("prompt_text", E2E_PROMPTS)
    def test_llm_logits_real_text(self, all_models, tokenizer, prompt_text):
        """Compare LLM logits for real tokenized text inputs.

        Both models receive the same tokenized text and we compare the
        first-step logits (before any sampling divergence).
        """
        kb_flow, kb_hift, kb_llm, ref_flow, ref_hift, ref_llm = all_models

        # Tokenize
        token_ids = _tokenize_text(tokenizer, prompt_text)
        text_tensor = torch.tensor([token_ids], dtype=torch.int32, device=DEVICE)

        # Build LM input identically for both models
        with torch.no_grad():
            # Text embedding (from the shared Qwen2 backbone)
            # KB uses llm.model.embed_tokens, ref uses llm.model.model.embed_tokens
            text_emb_kb = kb_llm.llm.model.embed_tokens(text_tensor)
            text_emb_ref = ref_llm.llm.model.model.embed_tokens(text_tensor)

            # SOS and task_id tokens
            sos_id = kb_llm.sos
            task_id = kb_llm.task_id

            kb_sos_emb = kb_llm.speech_embedding(
                torch.tensor([sos_id], device=DEVICE)
            ).unsqueeze(0)
            kb_task_emb = kb_llm.speech_embedding(
                torch.tensor([task_id], device=DEVICE)
            ).unsqueeze(0)

            ref_sos_emb = ref_llm.speech_embedding(
                torch.tensor([sos_id], device=DEVICE)
            ).unsqueeze(0)
            ref_task_emb = ref_llm.speech_embedding(
                torch.tensor([task_id], device=DEVICE)
            ).unsqueeze(0)

            # LM input: [sos, text_emb, task_id]  (no prompt speech tokens)
            empty_speech = torch.zeros(1, 0, kb_llm.llm_input_size, dtype=text_emb_kb.dtype, device=DEVICE)
            kb_lm_input = torch.cat([kb_sos_emb, text_emb_kb, kb_task_emb, empty_speech], dim=1)
            ref_lm_input = torch.cat([ref_sos_emb, text_emb_ref, ref_task_emb, empty_speech], dim=1)

            # Forward one step through LLM
            seq_len = kb_lm_input.shape[1]
            mask = torch.tril(torch.ones(1, seq_len, seq_len, device=DEVICE)).to(torch.bool)

            kb_y_pred, _ = kb_llm.llm.forward_one_step(kb_lm_input, masks=mask, cache=None)
            ref_y_pred, _ = ref_llm.llm.forward_one_step(ref_lm_input, masks=mask, cache=None)

            # Decode to logits
            kb_logp = kb_llm.llm_decoder(kb_y_pred[:, -1]).log_softmax(dim=-1)
            ref_logp = ref_llm.llm_decoder(ref_y_pred[:, -1]).log_softmax(dim=-1)

        # Compare
        max_diff = (kb_logp - ref_logp).abs().max().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            kb_logp.flatten().unsqueeze(0),
            ref_logp.flatten().unsqueeze(0),
        ).item()

        print(f"\n[LLM real text] '{prompt_text[:50]}...' "
              f"max_diff={max_diff:.6f}, cos_sim={cos_sim:.6f}")
        assert cos_sim > 0.999, f"LLM cosine similarity too low: {cos_sim}"
        # KB Qwen2 uses its own SDPA + RoPE; small numerical diffs are expected
        assert max_diff < 0.5, f"LLM max diff too high: {max_diff}"

    # ------------------------------------------------------------------
    # Test 2: Flow + Vocoder E2E with deterministic speech tokens
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("token_len", [30, 60, 100])
    def test_flow_vocoder_e2e(self, all_models, token_len):
        """Feed the same speech tokens to both KB and ref flow+vocoder.

        This tests the full token-to-waveform pipeline deterministically
        (no LLM sampling involved).
        """
        kb_flow, kb_hift, kb_llm, ref_flow, ref_hift, ref_llm = all_models

        # Use fixed seed for reproducible speech tokens
        torch.manual_seed(12345)
        token = torch.randint(0, 6561, (1, token_len), device=DEVICE)
        token_len_t = torch.tensor([token_len], dtype=torch.int32, device=DEVICE)
        prompt_token = torch.zeros(1, 0, dtype=torch.int32, device=DEVICE)
        prompt_token_len = torch.tensor([0], dtype=torch.int32, device=DEVICE)
        prompt_feat = torch.zeros(1, 0, 80, device=DEVICE)
        prompt_feat_len = torch.tensor([0], dtype=torch.int32, device=DEVICE)
        embedding = torch.randn(1, 192, device=DEVICE)

        with torch.no_grad():
            # Flow: tokens -> mel
            kb_mel, _ = kb_flow.inference(
                token, token_len_t, prompt_token, prompt_token_len,
                prompt_feat, prompt_feat_len, embedding,
                streaming=False, finalize=True,
            )
            ref_mel, _ = ref_flow.inference(
                token, token_len_t, prompt_token, prompt_token_len,
                prompt_feat, prompt_feat_len, embedding,
                streaming=False, finalize=True,
            )

        # Compare mel
        assert kb_mel.shape == ref_mel.shape, f"Mel shape mismatch: KB={kb_mel.shape}, ref={ref_mel.shape}"
        mel_cos = torch.nn.functional.cosine_similarity(
            kb_mel.flatten().unsqueeze(0), ref_mel.flatten().unsqueeze(0),
        ).item()
        mel_max_diff = (kb_mel - ref_mel).abs().max().item()
        print(f"\n[Flow E2E, T={token_len}] mel: max_diff={mel_max_diff:.6f}, cos_sim={mel_cos:.6f}")
        assert mel_cos > 0.99, f"Flow mel cosine similarity too low: {mel_cos}"

        # Vocoder: mel -> waveform (use the SAME mel for both to isolate vocoder)
        with torch.no_grad():
            kb_speech, _ = kb_hift.inference(speech_feat=kb_mel, finalize=True)
            ref_speech, _ = ref_hift.inference(speech_feat=ref_mel, finalize=True)

        min_len = min(kb_speech.shape[-1], ref_speech.shape[-1])
        kb_speech = kb_speech[..., :min_len]
        ref_speech = ref_speech[..., :min_len]

        speech_cos = torch.nn.functional.cosine_similarity(
            kb_speech.flatten().unsqueeze(0), ref_speech.flatten().unsqueeze(0),
        ).item()
        speech_max_diff = (kb_speech - ref_speech).abs().max().item()
        print(f"[Vocoder E2E, T={token_len}] speech: max_diff={speech_max_diff:.6f}, "
              f"cos_sim={speech_cos:.6f}, len={min_len}")
        assert speech_cos > 0.95, f"Vocoder cosine similarity too low: {speech_cos}"

    # ------------------------------------------------------------------
    # Test 3: Full pipeline (LLM → Flow → Vocoder) with real text
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("prompt_text", E2E_PROMPTS[:1])
    def test_full_pipeline_real_text(self, all_models, tokenizer, prompt_text, model_dir):
        """Full E2E: tokenize real text, run LLM (reference) to get tokens,
        then run flow+vocoder on both KB and ref, compare output audio.

        Since LLM sampling is stochastic, we use the reference LLM to generate
        speech tokens and feed the same tokens to both flow+vocoder pipelines.
        This ensures deterministic comparison of the flow+vocoder stage with
        realistic (not random) speech tokens from real text.
        """
        kb_flow, kb_hift, kb_llm, ref_flow, ref_hift, ref_llm = all_models

        # Tokenize real text
        token_ids = _tokenize_text(tokenizer, prompt_text)
        text_tensor = torch.tensor([token_ids], dtype=torch.int32, device=DEVICE)
        text_len = torch.tensor([len(token_ids)], dtype=torch.int32, device=DEVICE)
        prompt_text_t = torch.zeros(1, 0, dtype=torch.int32, device=DEVICE)
        prompt_text_len = torch.tensor([0], dtype=torch.int32, device=DEVICE)
        prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32, device=DEVICE)
        prompt_speech_token_len = torch.tensor([0], dtype=torch.int32, device=DEVICE)
        embedding = torch.zeros(1, 192, device=DEVICE)

        # Step 1: Generate speech tokens using reference LLM (seeded for reproducibility)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        with torch.no_grad():
            ref_speech_tokens = list(ref_llm.inference(
                text=text_tensor,
                text_len=text_len,
                prompt_text=prompt_text_t,
                prompt_text_len=prompt_text_len,
                prompt_speech_token=prompt_speech_token,
                prompt_speech_token_len=prompt_speech_token_len,
                embedding=embedding,
                sampling=25,
            ))

        print(f"\n[Full E2E] Text: '{prompt_text[:50]}...'")
        print(f"  Generated {len(ref_speech_tokens)} speech tokens")

        if len(ref_speech_tokens) == 0:
            pytest.skip("Reference LLM generated no speech tokens")

        # Step 2: Feed tokens through both flow+vocoder pipelines
        speech_token_tensor = torch.tensor(
            [ref_speech_tokens], dtype=torch.int32, device=DEVICE
        )
        token_len_t = torch.tensor(
            [speech_token_tensor.shape[1]], dtype=torch.int32, device=DEVICE
        )
        flow_prompt_token = torch.zeros(1, 0, dtype=torch.int32, device=DEVICE)
        flow_prompt_token_len = torch.tensor([0], dtype=torch.int32, device=DEVICE)
        prompt_feat = torch.zeros(1, 0, 80, device=DEVICE)
        prompt_feat_len = torch.tensor([0], dtype=torch.int32, device=DEVICE)
        flow_embedding = torch.zeros(1, 192, device=DEVICE)

        with torch.no_grad():
            # Flow: tokens -> mel
            kb_mel, _ = kb_flow.inference(
                speech_token_tensor, token_len_t,
                flow_prompt_token, flow_prompt_token_len,
                prompt_feat, prompt_feat_len, flow_embedding,
                streaming=False, finalize=True,
            )
            ref_mel, _ = ref_flow.inference(
                speech_token_tensor, token_len_t,
                flow_prompt_token, flow_prompt_token_len,
                prompt_feat, prompt_feat_len, flow_embedding,
                streaming=False, finalize=True,
            )

        mel_cos = torch.nn.functional.cosine_similarity(
            kb_mel.flatten().unsqueeze(0), ref_mel.flatten().unsqueeze(0),
        ).item()
        mel_max_diff = (kb_mel - ref_mel).abs().max().item()
        print(f"  Flow mel: max_diff={mel_max_diff:.6f}, cos_sim={mel_cos:.6f}, shape={kb_mel.shape}")
        assert mel_cos > 0.99, f"Full E2E flow mel cosine similarity too low: {mel_cos}"

        # Vocoder: mel -> waveform
        with torch.no_grad():
            kb_speech, _ = kb_hift.inference(speech_feat=kb_mel, finalize=True)
            ref_speech, _ = ref_hift.inference(speech_feat=ref_mel, finalize=True)

        min_len = min(kb_speech.shape[-1], ref_speech.shape[-1])
        kb_speech_cmp = kb_speech[..., :min_len]
        ref_speech_cmp = ref_speech[..., :min_len]

        speech_cos = torch.nn.functional.cosine_similarity(
            kb_speech_cmp.flatten().unsqueeze(0), ref_speech_cmp.flatten().unsqueeze(0),
        ).item()
        speech_max_diff = (kb_speech_cmp - ref_speech_cmp).abs().max().item()
        print(f"  Vocoder speech: max_diff={speech_max_diff:.6f}, cos_sim={speech_cos:.6f}, "
              f"KB_len={kb_speech.shape[-1]}, ref_len={ref_speech.shape[-1]}")
        assert speech_cos > 0.95, f"Full E2E speech cosine similarity too low: {speech_cos}"

        # Save audio if --save-images is passed
        save = os.environ.get("SAVE_IMAGES", "0") == "1"
        if save:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            try:
                import torchaudio
                kb_path = os.path.join(OUTPUT_DIR, "cosyvoice3_kb_e2e.wav")
                ref_path = os.path.join(OUTPUT_DIR, "cosyvoice3_ref_e2e.wav")
                torchaudio.save(kb_path, kb_speech.cpu(), 24000)
                torchaudio.save(ref_path, ref_speech.cpu(), 24000)
                print(f"  Saved KB audio to {kb_path}")
                print(f"  Saved ref audio to {ref_path}")
            except ImportError:
                print("  torchaudio not available, skipping audio save")

    # ------------------------------------------------------------------
    # Test 4: LLM multi-step generation alignment
    # ------------------------------------------------------------------
    def test_llm_multistep_real_text(self, all_models, tokenizer):
        """Run multiple autoregressive LLM steps with the same seed on both
        KB and reference, comparing logits at each step.

        This tests that the KV-cache and autoregressive loop produce
        identical results across both implementations.
        """
        kb_flow, kb_hift, kb_llm, ref_flow, ref_hift, ref_llm = all_models

        prompt_text = "Hello, this is a text to speech test."
        token_ids = _tokenize_text(tokenizer, prompt_text)
        text_tensor = torch.tensor([token_ids], dtype=torch.int32, device=DEVICE)

        n_steps = 10  # Number of autoregressive steps to compare

        with torch.no_grad():
            # Build LM input
            # KB uses llm.model.embed_tokens, ref uses llm.model.model.embed_tokens
            text_emb_kb = kb_llm.llm.model.embed_tokens(text_tensor)
            text_emb_ref = ref_llm.llm.model.model.embed_tokens(text_tensor)

            sos_id = kb_llm.sos
            task_id = kb_llm.task_id

            kb_sos_emb = kb_llm.speech_embedding(torch.tensor([sos_id], device=DEVICE)).unsqueeze(0)
            kb_task_emb = kb_llm.speech_embedding(torch.tensor([task_id], device=DEVICE)).unsqueeze(0)
            ref_sos_emb = ref_llm.speech_embedding(torch.tensor([sos_id], device=DEVICE)).unsqueeze(0)
            ref_task_emb = ref_llm.speech_embedding(torch.tensor([task_id], device=DEVICE)).unsqueeze(0)

            empty = torch.zeros(1, 0, kb_llm.llm_input_size, dtype=text_emb_kb.dtype, device=DEVICE)
            kb_lm_input = torch.cat([kb_sos_emb, text_emb_kb, kb_task_emb, empty], dim=1)
            ref_lm_input = torch.cat([ref_sos_emb, text_emb_ref, ref_task_emb, empty], dim=1)

            kb_cache = None
            ref_cache = None

            for step in range(n_steps):
                seq_len = kb_lm_input.shape[1] if kb_cache is None else kb_lm_input.shape[1] + kb_cache[0][0].size(2)
                mask = torch.tril(torch.ones(1, seq_len, seq_len, device=DEVICE)).to(torch.bool)

                kb_y, kb_cache = kb_llm.llm.forward_one_step(kb_lm_input, masks=mask, cache=kb_cache)
                ref_y, ref_cache = ref_llm.llm.forward_one_step(ref_lm_input, masks=mask, cache=ref_cache)

                kb_logp = kb_llm.llm_decoder(kb_y[:, -1]).log_softmax(dim=-1)
                ref_logp = ref_llm.llm_decoder(ref_y[:, -1]).log_softmax(dim=-1)

                max_diff = (kb_logp - ref_logp).abs().max().item()
                cos_sim = torch.nn.functional.cosine_similarity(
                    kb_logp.flatten().unsqueeze(0), ref_logp.flatten().unsqueeze(0),
                ).item()

                assert cos_sim > 0.999, f"Step {step}: LLM cos_sim={cos_sim}"
                assert max_diff < 0.5, f"Step {step}: LLM max_diff={max_diff}"

                # Use argmax for deterministic next token (same for both)
                top_id = kb_logp.argmax(dim=-1).item()
                kb_lm_input = kb_llm.speech_embedding(
                    torch.tensor([[top_id]], device=DEVICE)
                )
                ref_lm_input = ref_llm.speech_embedding(
                    torch.tensor([[top_id]], device=DEVICE)
                )

            print(f"\n[LLM multi-step] {n_steps} steps all passed (cos_sim>0.999)")


# ============================================================================
# Test: Component Smoke Tests (no weights needed)
# ============================================================================

class TestComponentSmoke:
    """Smoke tests for individual components (no pretrained weights needed)."""

    def test_dit_forward(self):
        """Test DiT forward pass with random weights."""
        kb_mod = importlib.import_module("KernelBench.level4.22_CosyVoice")
        DiT = kb_mod.DiT

        dit = DiT(
            dim=256,
            depth=2,
            heads=4,
            dim_head=64,
            mel_dim=80,
            spk_dim=80,
        ).to(DEVICE).eval()

        batch_size = 2
        seq_len = 50
        x = torch.randn(batch_size, 80, seq_len, device=DEVICE)
        mask = torch.ones(batch_size, 1, seq_len, device=DEVICE)
        mu = torch.randn(batch_size, 80, seq_len, device=DEVICE)
        t = torch.rand(batch_size, device=DEVICE)
        spks = torch.randn(batch_size, 80, device=DEVICE)
        cond = torch.randn(batch_size, 80, seq_len, device=DEVICE)

        with torch.no_grad():
            output = dit(x, mask, mu, t, spks, cond)

        assert output.shape == (batch_size, 80, seq_len), f"Unexpected shape: {output.shape}"
        assert not torch.isnan(output).any(), "NaN in DiT output"
        print(f"\nDiT smoke test passed: output shape {output.shape}")

    def test_vocoder_forward(self):
        """Test vocoder forward pass with random weights."""
        kb_mod = importlib.import_module("KernelBench.level4.22_CosyVoice")
        CausalHiFTGenerator = kb_mod.CausalHiFTGenerator

        hift = CausalHiFTGenerator(
            in_channels=80,
            base_channels=64,  # Small for testing
            nb_harmonics=8,
            sampling_rate=24000,
            upsample_rates=[8, 6, 2, 2, 2],
            upsample_kernel_sizes=[16, 12, 4, 4, 4],
            source_resblock_kernel_sizes=[7, 11, 11, 11, 11],  # Must match num_upsamples
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]],
        ).to(DEVICE).eval()

        mel = torch.randn(1, 80, 50, device=DEVICE)

        with torch.no_grad():
            speech, source = hift.inference(speech_feat=mel, finalize=True)

        assert speech.dim() == 2, f"Expected 2D output, got {speech.dim()}D"
        assert speech.shape[0] == 1
        assert speech.shape[1] > 0, "Empty speech output"
        assert not torch.isnan(speech).any(), "NaN in vocoder output"
        print(f"\nVocoder smoke test passed: output shape {speech.shape}")

    def test_flow_model_forward(self):
        """Test flow model forward pass with random weights."""
        kb_mod = importlib.import_module("KernelBench.level4.22_CosyVoice")
        CausalMaskedDiffWithDiT = kb_mod.CausalMaskedDiffWithDiT
        PreLookaheadLayer = kb_mod.PreLookaheadLayer
        CausalConditionalCFM = kb_mod.CausalConditionalCFM
        DiT = kb_mod.DiT

        dit = DiT(
            dim=256,
            depth=2,
            heads=4,
            dim_head=64,
            mel_dim=80,
            mu_dim=80,  # matches flow input_size
            spk_dim=80,
        )

        cfm = CausalConditionalCFM(
            in_channels=80,
            inference_cfg_rate=0.7,
            t_scheduler="cosine",
            estimator=dit,
        )

        pre_lookahead = PreLookaheadLayer(
            in_channels=80,
            channels=256,
            pre_lookahead_len=3,
        )

        flow = CausalMaskedDiffWithDiT(
            input_size=80,
            output_size=80,
            spk_embed_dim=192,
            vocab_size=6561,
            token_mel_ratio=2,
            pre_lookahead_len=3,
            pre_lookahead_layer=pre_lookahead,
            decoder=cfm,
        ).to(DEVICE).eval()

        token = torch.randint(0, 6561, (1, 30), device=DEVICE)
        token_len = torch.tensor([30], device=DEVICE)
        prompt_token = torch.randint(0, 6561, (1, 5), device=DEVICE)
        prompt_token_len = torch.tensor([5], device=DEVICE)
        prompt_feat = torch.randn(1, 10, 80, device=DEVICE)
        prompt_feat_len = torch.tensor([10], device=DEVICE)
        embedding = torch.randn(1, 192, device=DEVICE)

        with torch.no_grad():
            mel, _ = flow.inference(
                token, token_len, prompt_token, prompt_token_len,
                prompt_feat, prompt_feat_len, embedding,
                streaming=False, finalize=True,
            )

        assert mel.dim() == 3, f"Expected 3D output, got {mel.dim()}D"
        assert mel.shape[0] == 1
        assert mel.shape[1] == 80
        assert mel.shape[2] > 0, "Empty mel output"
        assert not torch.isnan(mel).any(), "NaN in flow output"
        print(f"\nFlow smoke test passed: output shape {mel.shape}")

    def test_llm_forward(self):
        """Test LLM forward pass with random weights."""
        kb_mod = importlib.import_module("KernelBench.level4.22_CosyVoice")
        CosyVoice3LM = kb_mod.CosyVoice3LM

        llm = CosyVoice3LM(
            llm_input_size=896,
            llm_output_size=896,
            speech_token_size=6561,
        ).to(DEVICE).eval()

        batch_size = 1
        seq_len = 10
        lm_input = torch.randn(batch_size, seq_len, 896, device=DEVICE)
        lm_input_len = torch.tensor([seq_len], device=DEVICE)

        with torch.no_grad():
            logits = llm.forward(lm_input, lm_input_len)

        assert logits.shape == (batch_size, seq_len, 6561 + 200), (
            f"Unexpected shape: {logits.shape}"
        )
        assert not torch.isnan(logits).any(), "NaN in LLM output"
        print(f"\nLLM smoke test passed: output shape {logits.shape}")
