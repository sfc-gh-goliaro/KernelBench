"""
Pytest configuration and shared fixtures for KernelBench tests.
"""

import pytest


def pytest_addoption(parser):
    """Add command-line options for model selection."""
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
        "--save-images",
        action="store_true",
        default=False,
        help="Save generated images to tests/outputs/ for visual inspection."
    )
