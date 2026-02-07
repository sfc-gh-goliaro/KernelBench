"""
Pytest configuration and shared fixtures for KernelBench tests.
"""

import pytest


def pytest_addoption(parser):
    """Add command-line options for model selection and testing."""
    parser.addoption(
        "--model-name",
        action="store",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name to test (default: meta-llama/Llama-3.1-8B-Instruct)"
    )
    parser.addoption(
        "--max-layers",
        action="store",
        type=int,
        default=None,
        help="Number of transformer layers to use (default: all layers). "
             "Use fewer layers for faster testing or to fit larger models on smaller GPUs."
    )
    parser.addoption(
        "--attn-backends",
        action="store",
        default=None,
        help="Comma-separated list of attention backends to test (e.g., 'sdpa,flash,eager'). "
             "If not specified, tests all available backends."
    )
    parser.addoption(
        "--save-plots",
        action="store_true",
        default=False,
        help="Save comparison plots to files (for test_speedup_kl_divergence.py)"
    )
    parser.addoption(
        "--plot-dir",
        action="store",
        default=".",
        help="Directory to save plots (default: current directory)"
    )
