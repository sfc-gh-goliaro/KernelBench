"""
Test speedup and KL divergence of KernelBench Llama 3.1 vs HuggingFace transformers.

This test measures:
1. End-to-end speedup (prefill + generation) of KernelBench vs transformers
2. KL divergence between logit distributions from both implementations
3. Comparison across different attention backends (sdpa, flash, flashinfer, eager)

Usage:
    # Basic test with 8B model (4 layers for fast testing)
    pytest tests/test_speedup_kl_divergence.py -v -s --model-name meta-llama/Llama-3.1-8B-Instruct --max-layers 4
    
    # Test specific attention backends
    pytest tests/test_speedup_kl_divergence.py -v -s --model-name meta-llama/Llama-3.1-8B-Instruct --attn-backends sdpa,flash
    
    # Run with plots saved to file
    pytest tests/test_speedup_kl_divergence.py -v -s --model-name meta-llama/Llama-3.1-8B-Instruct --save-plots
    
    # Full model test
    pytest tests/test_speedup_kl_divergence.py -v -s --model-name meta-llama/Llama-3.1-8B-Instruct

Requires HuggingFace authentication with access to the specified model.
"""

import pytest
import torch
import torch.nn.functional as F
import time
import sys
import os
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import json

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

from transformers import AutoTokenizer

# Import shared utilities from test_hf_alignment
from test_hf_alignment import (
    load_models,
    DEVICE,
    DTYPE,
    create_kb_model_from_hf_config,
    copy_weights,
    get_implementation_module,
)


# ============================================================================
# Configuration
# ============================================================================

# Test prompts with varying lengths
TEST_PROMPTS = [
    # Short prompt
    "What is the capital of France?",
    
    # Medium prompt
    "Explain the concept of machine learning to a 10-year-old. Use simple words and give an example that a child would understand.",
    
    # Long prompt with context
    """You are a helpful AI assistant. A user asks you the following question:

"I'm planning a road trip from San Francisco to Los Angeles. What are some interesting stops I should make along the way? I'm interested in nature, good food, and historical sites."

Please provide a detailed response with at least 5 recommendations.""",
]

# Number of tokens to generate for each prompt
GENERATION_TOKENS = [20, 50, 100]

# Number of warmup iterations before timing
WARMUP_ITERATIONS = 2

# Number of timing iterations for more accurate measurements
TIMING_ITERATIONS = 3


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    prompt: str
    prompt_len: int
    gen_tokens: int
    attn_backend: str
    hf_prefill_time_ms: float
    kb_prefill_time_ms: float
    hf_total_time_ms: float
    kb_total_time_ms: float
    prefill_speedup: float
    total_speedup: float
    prefill_kl_div: float
    generation_kl_div: float
    avg_kl_div: float
    tokens_match: bool


@dataclass 
class BackendComparisonResults:
    """Results comparing multiple backends."""
    model_name: str
    num_layers: int
    backends: List[str]
    results: Dict[str, List[BenchmarkResult]] = field(default_factory=dict)
    
    def add_result(self, backend: str, result: BenchmarkResult):
        if backend not in self.results:
            self.results[backend] = []
        self.results[backend].append(result)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for each backend."""
        summary = {}
        for backend, results in self.results.items():
            if not results:
                continue
            summary[backend] = {
                "avg_prefill_speedup": sum(r.prefill_speedup for r in results) / len(results),
                "avg_total_speedup": sum(r.total_speedup for r in results) / len(results),
                "avg_kl_div": sum(r.avg_kl_div for r in results) / len(results),
                "max_kl_div": max(r.avg_kl_div for r in results),
                "tokens_match_rate": sum(1 for r in results if r.tokens_match) / len(results),
            }
        return summary


def compute_kl_divergence(
    logits_a: torch.Tensor, 
    logits_b: torch.Tensor,
    temperature: float = 1.0
) -> float:
    """
    Compute KL divergence between two logit distributions.
    
    KL(P || Q) where P is derived from logits_a and Q from logits_b.
    
    Args:
        logits_a: Reference logits (batch, seq_len, vocab_size) or (batch, vocab_size)
        logits_b: Comparison logits (same shape as logits_a)
        temperature: Temperature for softmax (default 1.0)
        
    Returns:
        Mean KL divergence across all positions
    """
    # Flatten to (N, vocab_size) for easier computation
    if logits_a.dim() == 3:
        logits_a = logits_a.reshape(-1, logits_a.size(-1))
        logits_b = logits_b.reshape(-1, logits_b.size(-1))
    
    # Apply temperature
    logits_a = logits_a / temperature
    logits_b = logits_b / temperature
    
    # Compute log probabilities
    log_probs_a = F.log_softmax(logits_a, dim=-1)
    log_probs_b = F.log_softmax(logits_b, dim=-1)
    probs_a = F.softmax(logits_a, dim=-1)
    
    # KL(P || Q) = sum(P * (log P - log Q))
    kl_div = (probs_a * (log_probs_a - log_probs_b)).sum(dim=-1)
    
    return kl_div.mean().item()


def compute_symmetric_kl_divergence(
    logits_a: torch.Tensor, 
    logits_b: torch.Tensor,
    temperature: float = 1.0
) -> float:
    """
    Compute symmetric KL divergence (Jensen-Shannon style, but using average of two KLs).
    
    This is more robust as it doesn't depend on which distribution is reference.
    
    Args:
        logits_a: First logits tensor
        logits_b: Second logits tensor
        temperature: Temperature for softmax
        
    Returns:
        Symmetric KL divergence: (KL(P||Q) + KL(Q||P)) / 2
    """
    kl_ab = compute_kl_divergence(logits_a, logits_b, temperature)
    kl_ba = compute_kl_divergence(logits_b, logits_a, temperature)
    return (kl_ab + kl_ba) / 2


@torch.no_grad()
def benchmark_prefill(
    hf_model,
    kb_model,
    input_ids: torch.Tensor,
    block_table: torch.Tensor,
    warmup: int = WARMUP_ITERATIONS,
    iterations: int = TIMING_ITERATIONS,
) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    """
    Benchmark prefill (prompt processing) phase.
    
    Returns:
        (hf_time_ms, kb_time_ms, hf_logits, kb_logits)
    """
    # Warmup
    for _ in range(warmup):
        _ = hf_model(input_ids=input_ids, use_cache=False)
        kb_model.reset_cache()
        _ = kb_model.generate(input_ids, max_new_tokens=0, block_table=block_table, return_logits=True)
    
    torch.cuda.synchronize()
    
    # Time HuggingFace
    start = time.perf_counter()
    for _ in range(iterations):
        hf_out = hf_model(input_ids=input_ids, use_cache=False)
        torch.cuda.synchronize()
    hf_time_ms = (time.perf_counter() - start) * 1000 / iterations
    hf_logits = hf_out.logits
    
    # Time KernelBench
    start = time.perf_counter()
    for _ in range(iterations):
        kb_model.reset_cache()
        _, kb_logits_list = kb_model.generate(
            input_ids, 
            max_new_tokens=0, 
            block_table=block_table,
            return_logits=True
        )
        torch.cuda.synchronize()
    kb_time_ms = (time.perf_counter() - start) * 1000 / iterations
    kb_logits = kb_logits_list[0]
    
    return hf_time_ms, kb_time_ms, hf_logits, kb_logits


@torch.no_grad()
def benchmark_generation(
    hf_model,
    kb_model,
    input_ids: torch.Tensor,
    block_table: torch.Tensor,
    max_new_tokens: int,
    warmup: int = WARMUP_ITERATIONS,
    iterations: int = TIMING_ITERATIONS,
) -> Tuple[float, float, float, float, List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Benchmark full generation (prefill + decode).
    
    Returns:
        (hf_prefill_ms, hf_total_ms, kb_prefill_ms, kb_total_ms, 
         hf_all_logits, kb_all_logits, hf_tokens, kb_tokens)
    """
    prompt_len = input_ids.size(1)
    
    # Warmup
    for _ in range(warmup):
        # HF warmup
        hf_out = hf_model(input_ids=input_ids, use_cache=True)
        hf_past = hf_out.past_key_values
        hf_next = hf_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for _ in range(min(5, max_new_tokens - 1)):
            hf_out = hf_model(input_ids=hf_next, past_key_values=hf_past, use_cache=True)
            hf_past = hf_out.past_key_values
            hf_next = hf_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        # KB warmup
        kb_model.reset_cache()
        _ = kb_model.generate(input_ids, max_new_tokens=min(5, max_new_tokens), 
                              block_table=block_table, return_logits=True)
    
    torch.cuda.synchronize()
    
    # Time HuggingFace generation
    hf_generated = []
    hf_all_logits = []
    
    torch.cuda.synchronize()
    start_total = time.perf_counter()
    
    for iter_idx in range(iterations):
        iter_generated = []
        iter_logits = []
        
        # Prefill
        start_prefill = time.perf_counter()
        hf_out = hf_model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()
        if iter_idx == 0:
            hf_prefill_time = time.perf_counter() - start_prefill
        
        hf_past = hf_out.past_key_values
        iter_logits.append(hf_out.logits)
        hf_next = hf_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        iter_generated.append(hf_next)
        
        # Decode
        for _ in range(max_new_tokens - 1):
            hf_out = hf_model(input_ids=hf_next, past_key_values=hf_past, use_cache=True)
            hf_past = hf_out.past_key_values
            iter_logits.append(hf_out.logits)
            hf_next = hf_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            iter_generated.append(hf_next)
        
        torch.cuda.synchronize()
        
        if iter_idx == iterations - 1:
            hf_generated = iter_generated
            hf_all_logits = iter_logits
    
    hf_total_time = time.perf_counter() - start_total
    hf_prefill_ms = hf_prefill_time * 1000
    hf_total_ms = hf_total_time * 1000 / iterations
    hf_tokens = torch.cat(hf_generated, dim=1)
    
    # Time KernelBench generation
    torch.cuda.synchronize()
    start_total = time.perf_counter()
    
    for iter_idx in range(iterations):
        kb_model.reset_cache()
        
        # Use generate with return_logits to get all logits
        start_prefill = time.perf_counter()
        kb_full_seq, kb_logits_list = kb_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            block_table=block_table,
            return_logits=True
        )
        torch.cuda.synchronize()
        
        if iter_idx == 0:
            # Approximate prefill time (first iteration)
            pass
    
    kb_total_time = time.perf_counter() - start_total
    kb_total_ms = kb_total_time * 1000 / iterations
    
    # For KB prefill time, we'll use a separate prefill-only call
    kb_model.reset_cache()
    torch.cuda.synchronize()
    start = time.perf_counter()
    _, prefill_logits = kb_model.generate(
        input_ids, max_new_tokens=0, block_table=block_table, return_logits=True
    )
    torch.cuda.synchronize()
    kb_prefill_ms = (time.perf_counter() - start) * 1000
    
    kb_tokens = kb_full_seq[:, prompt_len:]
    kb_all_logits = kb_logits_list
    
    return (hf_prefill_ms, hf_total_ms, kb_prefill_ms, kb_total_ms,
            hf_all_logits, kb_all_logits, hf_tokens, kb_tokens)


def get_available_kb_backends():
    """Get available attention backends from KernelBench."""
    try:
        from KernelBench.level1.attention._10_GroupedQueryAttentionMultiBackend import get_available_backends
        return get_available_backends()
    except ImportError:
        return ["sdpa", "eager"]


def create_kb_model_with_backend(model_name: str, config_dict: dict, backend: str):
    """Create a KernelBench model with a specific attention backend."""
    import importlib
    module_path = get_implementation_module(model_name)
    kb_module = importlib.import_module(module_path)
    config_dict = config_dict.copy()
    config_dict['attn_backend'] = backend
    return kb_module.Model(**config_dict), kb_module


# ============================================================================
# Plotting Functions
# ============================================================================

def create_comparison_plots(
    results: BackendComparisonResults,
    output_dir: str = ".",
    show_plots: bool = False,
) -> List[str]:
    """
    Create comparison plots for speedup and KL divergence across backends.
    
    Args:
        results: BackendComparisonResults containing benchmark data
        output_dir: Directory to save plots
        show_plots: Whether to display plots interactively
        
    Returns:
        List of saved plot file paths
    """
    try:
        import matplotlib
        if not show_plots:
            matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  Warning: matplotlib not available, skipping plots")
        return []
    
    saved_files = []
    summary = results.get_summary()
    
    if not summary:
        print("  No results to plot")
        return []
    
    backends = list(summary.keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'KernelBench vs HuggingFace Comparison\n{results.model_name} ({results.num_layers} layers)', 
                 fontsize=14, fontweight='bold')
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(backends)))
    
    # 1. Prefill Speedup by Backend
    ax1 = axes[0, 0]
    prefill_speedups = [summary[b]['avg_prefill_speedup'] for b in backends]
    bars1 = ax1.bar(backends, prefill_speedups, color=colors)
    ax1.axhline(y=1.0, color='red', linestyle='--', label='HF baseline')
    ax1.set_ylabel('Speedup (x)')
    ax1.set_title('Prefill Speedup by Backend')
    ax1.set_ylim(bottom=0)
    for bar, val in zip(bars1, prefill_speedups):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=10)
    
    # 2. Total Speedup by Backend
    ax2 = axes[0, 1]
    total_speedups = [summary[b]['avg_total_speedup'] for b in backends]
    bars2 = ax2.bar(backends, total_speedups, color=colors)
    ax2.axhline(y=1.0, color='red', linestyle='--', label='HF baseline')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Total (Prefill + Decode) Speedup by Backend')
    ax2.set_ylim(bottom=0)
    for bar, val in zip(bars2, total_speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=10)
    
    # 3. KL Divergence by Backend
    ax3 = axes[1, 0]
    avg_kl = [summary[b]['avg_kl_div'] for b in backends]
    max_kl = [summary[b]['max_kl_div'] for b in backends]
    x = np.arange(len(backends))
    width = 0.35
    bars3a = ax3.bar(x - width/2, avg_kl, width, label='Avg KL Div', color=colors)
    bars3b = ax3.bar(x + width/2, max_kl, width, label='Max KL Div', color=colors, alpha=0.5)
    ax3.set_ylabel('KL Divergence')
    ax3.set_title('KL Divergence by Backend (lower is better)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(backends)
    ax3.legend()
    ax3.set_yscale('log')
    
    # 4. Token Match Rate by Backend
    ax4 = axes[1, 1]
    match_rates = [summary[b]['tokens_match_rate'] * 100 for b in backends]
    bars4 = ax4.bar(backends, match_rates, color=colors)
    ax4.axhline(y=100.0, color='green', linestyle='--', alpha=0.5)
    ax4.set_ylabel('Match Rate (%)')
    ax4.set_title('Token Match Rate by Backend')
    ax4.set_ylim(0, 105)
    for bar, val in zip(bars4, match_rates):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'backend_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    saved_files.append(plot_path)
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Create detailed per-prompt plots
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 10))
    fig2.suptitle(f'Detailed Results by Prompt\n{results.model_name}', fontsize=14, fontweight='bold')
    
    # Group results by prompt length
    prompt_lens = sorted(set(r.prompt_len for backend_results in results.results.values() 
                             for r in backend_results))
    
    # Speedup by prompt length
    ax5 = axes2[0]
    x = np.arange(len(prompt_lens))
    width = 0.8 / len(backends)
    for i, backend in enumerate(backends):
        backend_results = results.results.get(backend, [])
        speedups_by_len = []
        for plen in prompt_lens:
            matching = [r.total_speedup for r in backend_results if r.prompt_len == plen]
            speedups_by_len.append(np.mean(matching) if matching else 0)
        ax5.bar(x + i * width, speedups_by_len, width, label=backend, color=colors[i])
    ax5.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Prompt Length (tokens)')
    ax5.set_ylabel('Total Speedup (x)')
    ax5.set_title('Speedup by Prompt Length')
    ax5.set_xticks(x + width * (len(backends) - 1) / 2)
    ax5.set_xticklabels(prompt_lens)
    ax5.legend()
    
    # KL divergence by prompt length
    ax6 = axes2[1]
    for i, backend in enumerate(backends):
        backend_results = results.results.get(backend, [])
        kl_by_len = []
        for plen in prompt_lens:
            matching = [r.avg_kl_div for r in backend_results if r.prompt_len == plen]
            kl_by_len.append(np.mean(matching) if matching else 0)
        ax6.bar(x + i * width, kl_by_len, width, label=backend, color=colors[i])
    ax6.set_xlabel('Prompt Length (tokens)')
    ax6.set_ylabel('Avg KL Divergence')
    ax6.set_title('KL Divergence by Prompt Length')
    ax6.set_xticks(x + width * (len(backends) - 1) / 2)
    ax6.set_xticklabels(prompt_lens)
    ax6.legend()
    ax6.set_yscale('log')
    
    plt.tight_layout()
    
    plot_path2 = os.path.join(output_dir, 'detailed_results.png')
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    saved_files.append(plot_path2)
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return saved_files


def print_results_table(results: BackendComparisonResults):
    """Print a formatted table of results."""
    print("\n" + "="*100)
    print("DETAILED RESULTS TABLE")
    print("="*100)
    
    header = f"{'Backend':<12} {'Prompt Len':<12} {'Gen Tokens':<12} {'Prefill↑':<12} {'Total↑':<12} {'KL Div':<12} {'Match':<8}"
    print(header)
    print("-"*100)
    
    for backend, backend_results in sorted(results.results.items()):
        for r in backend_results:
            row = f"{backend:<12} {r.prompt_len:<12} {r.gen_tokens:<12} {r.prefill_speedup:<12.2f} {r.total_speedup:<12.2f} {r.avg_kl_div:<12.2e} {'Yes' if r.tokens_match else 'No':<8}"
            print(row)
    
    print("\n" + "="*100)
    print("SUMMARY BY BACKEND")
    print("="*100)
    
    summary = results.get_summary()
    header2 = f"{'Backend':<12} {'Prefill Speedup':<18} {'Total Speedup':<18} {'Avg KL Div':<15} {'Match Rate':<12}"
    print(header2)
    print("-"*100)
    
    for backend, stats in sorted(summary.items()):
        row = f"{backend:<12} {stats['avg_prefill_speedup']:<18.2f}x {stats['avg_total_speedup']:<18.2f}x {stats['avg_kl_div']:<15.2e} {stats['tokens_match_rate']*100:<12.0f}%"
        print(row)


# ============================================================================
# Model Fixture
# ============================================================================

@pytest.fixture(scope="module")
def loaded_models(request):
    """
    Load models based on command-line parameters.
    
    This fixture loads both the HuggingFace model and the corresponding
    KernelBench implementation for comparison testing.
    """
    model_name = request.config.getoption("--model-name")
    max_layers = request.config.getoption("--max-layers")
    try:
        return load_models(model_name, max_layers), model_name, max_layers
    except ValueError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not load {model_name}: {e}")


@pytest.fixture(scope="module")
def test_backends(request):
    """Get list of attention backends to test."""
    backends_str = request.config.getoption("--attn-backends")
    if backends_str:
        requested = [b.strip() for b in backends_str.split(",")]
        available = get_available_kb_backends()
        valid = [b for b in requested if b in available]
        if not valid:
            pytest.skip(f"None of the requested backends {requested} are available. Available: {available}")
        return valid
    else:
        # Default: test all available backends
        return get_available_kb_backends()


@pytest.fixture(scope="module")
def plot_config(request):
    """Get plotting configuration."""
    return {
        "save_plots": request.config.getoption("--save-plots"),
        "plot_dir": request.config.getoption("--plot-dir"),
    }


# ============================================================================
# Main Tests
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_speedup_and_kl(loaded_models, test_backends):
    """
    Test prefill phase speedup and KL divergence across attention backends.
    """
    (hf_model, kb_model, tokenizer, kb_config, kb_module), model_name, max_layers = loaded_models
    
    layers_info = f" ({max_layers} layers)" if max_layers else ""
    print("\n" + "="*80)
    print(f"PREFILL SPEEDUP AND KL DIVERGENCE TEST")
    print(f"Model: {model_name}{layers_info}")
    print(f"Testing backends: {test_backends}")
    print("="*80)
    
    all_results = {}
    
    for backend in test_backends:
        print(f"\n--- Testing backend: {backend} ---")
        
        # Set backend
        try:
            kb_model.set_attn_backend(backend)
        except ValueError as e:
            print(f"  Skipping backend {backend}: {e}")
            continue
        
        results = []
        
        for prompt in TEST_PROMPTS:
            encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            input_ids = encoded['input_ids']
            prompt_len = input_ids.size(1)
            
            # Allocate block table
            max_blocks = (prompt_len + kb_config['block_size'] - 1) // kb_config['block_size'] + 10
            block_table = torch.arange(max_blocks, device=DEVICE, dtype=torch.long).unsqueeze(0)
            
            # Benchmark
            hf_time, kb_time, hf_logits, kb_logits = benchmark_prefill(
                hf_model, kb_model, input_ids, block_table
            )
            
            # Compute KL divergence
            kl_div = compute_symmetric_kl_divergence(hf_logits, kb_logits)
            
            # Compute speedup
            speedup = hf_time / kb_time if kb_time > 0 else float('inf')
            
            # Check top token match
            hf_top = hf_logits[:, -1, :].argmax(dim=-1)
            kb_top = kb_logits[:, -1, :].argmax(dim=-1)
            tokens_match = (hf_top == kb_top).all().item()
            
            results.append({
                'prompt_len': prompt_len,
                'hf_time_ms': hf_time,
                'kb_time_ms': kb_time,
                'speedup': speedup,
                'kl_div': kl_div,
                'tokens_match': tokens_match,
            })
            
            print(f"\n  Prompt length: {prompt_len} tokens")
            print(f"    HF time: {hf_time:.2f} ms, KB time: {kb_time:.2f} ms")
            print(f"    Speedup: {speedup:.2f}x, KL div: {kl_div:.2e}, Match: {tokens_match}")
        
        all_results[backend] = results
        
        # Summary for this backend
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        avg_kl = sum(r['kl_div'] for r in results) / len(results)
        print(f"\n  Backend {backend} summary: avg speedup={avg_speedup:.2f}x, avg KL={avg_kl:.2e}")
    
    print("\n" + "="*80)
    print("PREFILL SUMMARY ACROSS BACKENDS")
    print("="*80)
    
    for backend, results in all_results.items():
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        avg_kl = sum(r['kl_div'] for r in results) / len(results)
        all_match = all(r['tokens_match'] for r in results)
        print(f"  {backend}: speedup={avg_speedup:.2f}x, KL={avg_kl:.2e}, all_match={all_match}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_generation_speedup_and_kl_multibackend(loaded_models, test_backends, plot_config):
    """
    Test end-to-end generation speedup and KL divergence across attention backends.
    
    This is the main comprehensive test that compares all backends.
    """
    (hf_model, kb_model, tokenizer, kb_config, kb_module), model_name, max_layers = loaded_models
    
    layers_info = f" ({max_layers} layers)" if max_layers else ""
    print("\n" + "="*80)
    print(f"GENERATION SPEEDUP AND KL DIVERGENCE TEST (MULTI-BACKEND)")
    print(f"Model: {model_name}{layers_info}")
    print(f"Testing backends: {test_backends}")
    print("="*80)
    
    # Create results container
    comparison_results = BackendComparisonResults(
        model_name=model_name,
        num_layers=max_layers or kb_config.get('num_layers', -1),
        backends=test_backends,
    )
    
    for backend in test_backends:
        print(f"\n{'='*40}")
        print(f"TESTING BACKEND: {backend.upper()}")
        print(f"{'='*40}")
        
        # Set backend
        try:
            kb_model.set_attn_backend(backend)
        except ValueError as e:
            print(f"  Skipping backend {backend}: {e}")
            continue
        
        for prompt, gen_tokens in zip(TEST_PROMPTS, GENERATION_TOKENS):
            encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            input_ids = encoded['input_ids']
            prompt_len = input_ids.size(1)
            
            # Allocate block table
            max_seq = prompt_len + gen_tokens + 10
            max_blocks = (max_seq + kb_config['block_size'] - 1) // kb_config['block_size']
            block_table = torch.arange(max_blocks, device=DEVICE, dtype=torch.long).unsqueeze(0)
            
            # Benchmark
            (hf_prefill_ms, hf_total_ms, kb_prefill_ms, kb_total_ms,
             hf_all_logits, kb_all_logits, hf_tokens, kb_tokens) = benchmark_generation(
                hf_model, kb_model, input_ids, block_table, gen_tokens
            )
            
            # Compute KL divergence for prefill
            prefill_kl = compute_symmetric_kl_divergence(
                hf_all_logits[0][:, -1:, :],
                kb_all_logits[0][:, -1:, :]
            )
            
            # Compute average KL divergence for generation steps
            gen_kl_divs = []
            min_steps = min(len(hf_all_logits) - 1, len(kb_all_logits) - 1)
            for i in range(min_steps):
                hf_step_logits = hf_all_logits[i + 1]
                kb_step_logits = kb_all_logits[i + 1]
                step_kl = compute_symmetric_kl_divergence(hf_step_logits, kb_step_logits)
                gen_kl_divs.append(step_kl)
            
            avg_gen_kl = sum(gen_kl_divs) / len(gen_kl_divs) if gen_kl_divs else 0.0
            overall_kl = (prefill_kl + avg_gen_kl) / 2
            
            # Compute speedups
            prefill_speedup = hf_prefill_ms / kb_prefill_ms if kb_prefill_ms > 0 else float('inf')
            total_speedup = hf_total_ms / kb_total_ms if kb_total_ms > 0 else float('inf')
            
            # Check token match
            min_tokens = min(hf_tokens.size(1), kb_tokens.size(1))
            tokens_match = (hf_tokens[0, :min_tokens] == kb_tokens[0, :min_tokens]).all().item()
            
            result = BenchmarkResult(
                prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt,
                prompt_len=prompt_len,
                gen_tokens=gen_tokens,
                attn_backend=backend,
                hf_prefill_time_ms=hf_prefill_ms,
                kb_prefill_time_ms=kb_prefill_ms,
                hf_total_time_ms=hf_total_ms,
                kb_total_time_ms=kb_total_ms,
                prefill_speedup=prefill_speedup,
                total_speedup=total_speedup,
                prefill_kl_div=prefill_kl,
                generation_kl_div=avg_gen_kl,
                avg_kl_div=overall_kl,
                tokens_match=tokens_match,
            )
            comparison_results.add_result(backend, result)
            
            print(f"\n  Prompt len: {prompt_len}, Gen: {gen_tokens} tokens")
            print(f"    Prefill: HF={hf_prefill_ms:.1f}ms, KB={kb_prefill_ms:.1f}ms, Speedup={prefill_speedup:.2f}x")
            print(f"    Total: HF={hf_total_ms:.1f}ms, KB={kb_total_ms:.1f}ms, Speedup={total_speedup:.2f}x")
            print(f"    KL Div: prefill={prefill_kl:.2e}, gen={avg_gen_kl:.2e}, overall={overall_kl:.2e}")
            print(f"    Tokens match: {tokens_match}")
    
    # Print detailed results table
    print_results_table(comparison_results)
    
    # Create plots if requested
    if plot_config["save_plots"]:
        print("\n" + "="*80)
        print("CREATING PLOTS")
        print("="*80)
        plot_files = create_comparison_plots(
            comparison_results,
            output_dir=plot_config["plot_dir"],
            show_plots=False,
        )
        if plot_files:
            print(f"  Saved plots to: {plot_files}")
    
    # Assertions
    summary = comparison_results.get_summary()
    for backend, stats in summary.items():
        assert stats['avg_kl_div'] < 0.1, \
            f"Backend {backend}: Average KL divergence {stats['avg_kl_div']:.2e} too high"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_throughput_multibackend(loaded_models, test_backends):
    """
    Test throughput (tokens per second) across backends.
    """
    (hf_model, kb_model, tokenizer, kb_config, kb_module), model_name, max_layers = loaded_models
    
    layers_info = f" ({max_layers} layers)" if max_layers else ""
    print("\n" + "="*80)
    print(f"THROUGHPUT TEST (MULTI-BACKEND)")
    print(f"Model: {model_name}{layers_info}")
    print(f"Testing backends: {test_backends}")
    print("="*80)
    
    # Use medium-length prompt
    prompt = TEST_PROMPTS[1]
    gen_tokens = 100
    iterations = 3
    
    encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = encoded['input_ids']
    prompt_len = input_ids.size(1)
    
    # Allocate block table
    max_seq = prompt_len + gen_tokens + 10
    max_blocks = (max_seq + kb_config['block_size'] - 1) // kb_config['block_size']
    block_table = torch.arange(max_blocks, device=DEVICE, dtype=torch.long).unsqueeze(0)
    
    # HuggingFace throughput (baseline)
    torch.cuda.synchronize()
    hf_start = time.perf_counter()
    
    for _ in range(iterations):
        hf_out = hf_model(input_ids=input_ids, use_cache=True)
        hf_past = hf_out.past_key_values
        hf_next = hf_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        for _ in range(gen_tokens - 1):
            hf_out = hf_model(input_ids=hf_next, past_key_values=hf_past, use_cache=True)
            hf_past = hf_out.past_key_values
            hf_next = hf_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        torch.cuda.synchronize()
    
    hf_total_time = time.perf_counter() - hf_start
    hf_total_tokens = (prompt_len + gen_tokens) * iterations
    hf_throughput = hf_total_tokens / hf_total_time
    
    print(f"\n  Configuration: prompt_len={prompt_len}, gen_tokens={gen_tokens}")
    print(f"\n  HuggingFace baseline: {hf_throughput:.1f} tokens/sec")
    
    results = {"hf": hf_throughput}
    
    for backend in test_backends:
        try:
            kb_model.set_attn_backend(backend)
        except ValueError as e:
            print(f"  Skipping backend {backend}: {e}")
            continue
        
        torch.cuda.synchronize()
        kb_start = time.perf_counter()
        
        for _ in range(iterations):
            kb_model.reset_cache()
            _ = kb_model.generate(
                input_ids,
                max_new_tokens=gen_tokens,
                block_table=block_table,
                return_logits=False
            )
            torch.cuda.synchronize()
        
        kb_total_time = time.perf_counter() - kb_start
        kb_throughput = hf_total_tokens / kb_total_time
        speedup = kb_throughput / hf_throughput
        
        results[backend] = kb_throughput
        print(f"  KB ({backend}): {kb_throughput:.1f} tokens/sec (speedup: {speedup:.2f}x)")
    
    # Summary
    print("\n" + "-"*80)
    print("THROUGHPUT SUMMARY")
    print("-"*80)
    print(f"  {'Backend':<15} {'Throughput':<20} {'Speedup':<15}")
    print("-"*80)
    print(f"  {'HuggingFace':<15} {hf_throughput:<20.1f} {'1.00x (baseline)':<15}")
    for backend in test_backends:
        if backend in results:
            speedup = results[backend] / hf_throughput
            print(f"  {f'KB ({backend})':<15} {results[backend]:<20.1f} {speedup:.2f}x")


if __name__ == "__main__":
    # Forward command-line arguments to pytest
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
