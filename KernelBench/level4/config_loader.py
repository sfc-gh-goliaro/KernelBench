"""
Configuration Loader for Level4 Models

This module provides utilities for loading model configurations from:
- HuggingFace Hub using transformers.AutoConfig
- Hardcoded fallback values from Table 5
- GitHub reference implementations
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Hardcoded Configurations from Table 5
# ============================================================================

HARDCODED_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Dense Decoder - Llama-3.1
    "meta-llama/Llama-3.1-8B": {
        "hidden_size": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 14336,
        "num_layers": 32,
        "vocab_size": 128256,
        "max_seq_len": 131072,
        "rope_theta": 500000,
        "rms_norm_eps": 1e-5,
    },
    "meta-llama/Llama-3.1-70B": {
        "hidden_size": 8192,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 28672,
        "num_layers": 80,
        "vocab_size": 128256,
        "max_seq_len": 131072,
        "rope_theta": 500000,
        "rms_norm_eps": 1e-5,
    },
    
    # Dense Decoder - Falcon (MQA + ALiBi)
    "tiiuae/falcon-7b": {
        "hidden_size": 4544,
        "num_heads": 71,
        "num_kv_heads": 1,
        "head_dim": 64,
        "intermediate_size": 18176,
        "num_layers": 32,
        "vocab_size": 65024,
        "max_seq_len": 2048,
        "alibi": True,
    },
    "tiiuae/falcon-40b": {
        "hidden_size": 8192,
        "num_heads": 128,
        "num_kv_heads": 8,
        "head_dim": 64,
        "intermediate_size": 32768,
        "num_layers": 60,
        "vocab_size": 65024,
        "max_seq_len": 2048,
        "alibi": True,
    },
    
    # Dense Decoder - Mistral (Sliding Window)
    "mistralai/Mistral-7B-v0.3": {
        "hidden_size": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 14336,
        "num_layers": 32,
        "vocab_size": 32768,
        "max_seq_len": 32768,
        "sliding_window": 4096,
        "rope_theta": 10000,
    },
    "mistralai/Mistral-Nemo-12B": {
        "hidden_size": 5120,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 14336,
        "num_layers": 40,
        "vocab_size": 131072,
        "max_seq_len": 131072,
        "sliding_window": None,
    },
    
    # MoE Models
    "deepseek-ai/DeepSeek-V2-Lite": {
        "hidden_size": 2048,
        "num_heads": 16,
        "kv_lora_rank": 512,
        "qk_nope_dim": 128,
        "qk_rope_dim": 64,
        "v_head_dim": 128,
        "intermediate_size": 10944,
        "num_layers": 27,
        "vocab_size": 102400,
        "num_experts": 64,
        "num_experts_per_tok": 6,
        "num_shared_experts": 2,
    },
    "mistralai/Mixtral-8x7B-v0.1": {
        "hidden_size": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 14336,
        "num_layers": 32,
        "vocab_size": 32000,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "sliding_window": 4096,
        "rope_theta": 1000000,
    },
    
    # SSM - Mamba-2
    "state-spaces/mamba2-1.3b": {
        "d_model": 2048,
        "d_state": 128,
        "d_conv": 4,
        "expand": 2,
        "d_inner": 4096,
        "headdim": 64,
        "ngroups": 8,
        "num_layers": 48,
        "vocab_size": 50280,
    },
    "state-spaces/mamba2-2.7b": {
        "d_model": 2560,
        "d_state": 128,
        "d_conv": 4,
        "expand": 2,
        "d_inner": 5120,
        "headdim": 64,
        "ngroups": 8,
        "num_layers": 64,
        "vocab_size": 50280,
    },
    
    # Linear Attention - RWKV-6
    "RWKV/rwkv-6-1.6b": {
        "hidden_size": 2048,
        "num_heads": 32,
        "head_size": 64,
        "intermediate_size": 7168,
        "num_layers": 24,
        "vocab_size": 65536,
        "time_mix_extra_dim": 64,
        "time_decay_extra_dim": 128,
    },
    "RWKV/rwkv-6-7b": {
        "hidden_size": 4096,
        "num_heads": 64,
        "head_size": 64,
        "intermediate_size": 14336,
        "num_layers": 32,
        "vocab_size": 65536,
        "time_mix_extra_dim": 64,
        "time_decay_extra_dim": 128,
    },
    
    # Linear Attention - GLA
    "fla-hub/gla-1.3b": {
        "hidden_size": 2048,
        "num_heads": 8,
        "head_dim": 256,
        "expand_k": 1.0,
        "expand_v": 2.0,
        "intermediate_size": 5632,
        "num_layers": 24,
        "vocab_size": 32000,
        "gate_logit_normalizer": 16,
    },
    "fla-hub/gla-7b": {
        "hidden_size": 4096,
        "num_heads": 16,
        "head_dim": 256,
        "expand_k": 1.0,
        "expand_v": 2.0,
        "intermediate_size": 11264,
        "num_layers": 32,
        "vocab_size": 32000,
        "gate_logit_normalizer": 16,
    },
    
    # Linear Attention - RetNet
    "microsoft/retnet-1.3b": {
        "hidden_size": 2048,
        "num_heads": 8,
        "head_dim": 256,
        "expand_k": 1.0,
        "expand_v": 2.0,
        "intermediate_size": 4096,
        "num_layers": 24,
        "vocab_size": 32000,
        "value_dim": 4096,
    },
    "microsoft/retnet-6.7b": {
        "hidden_size": 4096,
        "num_heads": 16,
        "head_dim": 256,
        "expand_k": 1.0,
        "expand_v": 2.0,
        "intermediate_size": 8192,
        "num_layers": 32,
        "vocab_size": 32000,
        "value_dim": 8192,
    },
    
    # Encoder-Decoder - T5
    "google-t5/t5-base": {
        "hidden_size": 768,
        "num_heads": 12,
        "head_dim": 64,
        "intermediate_size": 3072,
        "num_encoder_layers": 12,
        "num_decoder_layers": 12,
        "vocab_size": 32128,
        "max_seq_len": 512,
        "relative_attention_num_buckets": 32,
    },
    "google-t5/t5-large": {
        "hidden_size": 1024,
        "num_heads": 16,
        "head_dim": 64,
        "intermediate_size": 4096,
        "num_encoder_layers": 24,
        "num_decoder_layers": 24,
        "vocab_size": 32128,
        "max_seq_len": 512,
        "relative_attention_num_buckets": 32,
    },
    "google-t5/t5-3b": {
        "hidden_size": 1024,
        "num_heads": 32,
        "head_dim": 128,
        "intermediate_size": 16384,
        "num_encoder_layers": 24,
        "num_decoder_layers": 24,
        "vocab_size": 32128,
        "max_seq_len": 512,
        "relative_attention_num_buckets": 32,
    },
    
    # Vision Transformer - Swin-v2
    "microsoft/swinv2-tiny-patch4-window8-256": {
        "hidden_size": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": 8,
        "patch_size": 4,
        "image_size": 256,
        "mlp_ratio": 4.0,
    },
    "microsoft/swinv2-base-patch4-window12-384": {
        "hidden_size": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
        "window_size": 12,
        "patch_size": 4,
        "image_size": 384,
        "mlp_ratio": 4.0,
    },
    "microsoft/swinv2-large-patch4-window12-384": {
        "hidden_size": 192,
        "depths": [2, 2, 18, 2],
        "num_heads": [6, 12, 24, 48],
        "window_size": 12,
        "patch_size": 4,
        "image_size": 384,
        "mlp_ratio": 4.0,
    },
    
    # Vision-Language - Qwen2-VL
    "Qwen/Qwen2-VL-2B-Instruct": {
        "hidden_size": 1536,
        "num_heads": 12,
        "num_kv_heads": 2,
        "head_dim": 128,
        "intermediate_size": 8960,
        "num_layers": 28,
        "vocab_size": 151936,
        "vision_hidden": 1280,
        "vision_heads": 16,
        "vision_layers": 32,
        "patch_size": 14,
        "temporal_patch_size": 2,
    },
    "Qwen/Qwen2-VL-7B-Instruct": {
        "hidden_size": 3584,
        "num_heads": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
        "intermediate_size": 18944,
        "num_layers": 28,
        "vocab_size": 152064,
        "vision_hidden": 1280,
        "vision_heads": 16,
        "vision_layers": 32,
        "patch_size": 14,
        "temporal_patch_size": 2,
    },
    "Qwen/Qwen2-VL-72B-Instruct": {
        "hidden_size": 8192,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 29568,
        "num_layers": 80,
        "vocab_size": 152064,
        "vision_hidden": 1280,
        "vision_heads": 16,
        "vision_layers": 32,
        "patch_size": 14,
        "temporal_patch_size": 2,
    },
    
    # Audio-Language - Whisper
    "openai/whisper-small": {
        "hidden_size": 768,
        "num_heads": 12,
        "head_dim": 64,
        "intermediate_size": 3072,
        "num_encoder_layers": 12,
        "num_decoder_layers": 12,
        "vocab_size": 51865,
        "max_seq_len": 448,
        "n_mels": 80,
        "n_audio_ctx": 1500,
    },
    "openai/whisper-medium": {
        "hidden_size": 1024,
        "num_heads": 16,
        "head_dim": 64,
        "intermediate_size": 4096,
        "num_encoder_layers": 24,
        "num_decoder_layers": 24,
        "vocab_size": 51865,
        "max_seq_len": 448,
        "n_mels": 80,
        "n_audio_ctx": 1500,
    },
    "openai/whisper-large-v3": {
        "hidden_size": 1280,
        "num_heads": 20,
        "head_dim": 64,
        "intermediate_size": 5120,
        "num_encoder_layers": 32,
        "num_decoder_layers": 32,
        "vocab_size": 51866,
        "max_seq_len": 448,
        "n_mels": 128,
        "n_audio_ctx": 1500,
    },
    
    # Diffusion - Stable Diffusion
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "unet_channels": [320, 640, 1280],
        "attention_heads": [5, 10, 20],
        "transformer_layers": [1, 2, 10],
        "cross_attention_dim": 2048,
        "vae_latent_channels": 4,
        "vae_out_channels": 3,
        "image_size": 1024,
    },
    "stabilityai/stable-diffusion-3-medium": {
        "hidden_size": 1536,
        "num_heads": 24,
        "head_dim": 64,
        "num_layers": 24,
        "patch_size": 2,
        "in_channels": 16,
        "pooled_projection_dim": 2048,
        "joint_attention_dim": 4096,
    },
    
    # CNN/Mobile - EfficientNet
    "google/efficientnet-b0": {
        "width_mult": 1.0,
        "depth_mult": 1.0,
        "image_size": 224,
        "dropout": 0.2,
        "num_classes": 1000,
        "stem_channels": 32,
    },
    "google/efficientnet-b4": {
        "width_mult": 1.4,
        "depth_mult": 1.8,
        "image_size": 380,
        "dropout": 0.4,
        "num_classes": 1000,
        "stem_channels": 48,
    },
    "google/efficientnet-b7": {
        "width_mult": 2.0,
        "depth_mult": 3.1,
        "image_size": 600,
        "dropout": 0.5,
        "num_classes": 1000,
        "stem_channels": 64,
    },
    
    # Classic CNN - ResNet
    "microsoft/resnet-50": {
        "layers": [3, 4, 6, 3],
        "channels": [64, 128, 256, 512],
        "expansion": 4,
        "image_size": 224,
        "num_classes": 1000,
        "stem_channels": 64,
    },
    "microsoft/resnet-101": {
        "layers": [3, 4, 23, 3],
        "channels": [64, 128, 256, 512],
        "expansion": 4,
        "image_size": 224,
        "num_classes": 1000,
        "stem_channels": 64,
    },
    "microsoft/resnet-152": {
        "layers": [3, 8, 36, 3],
        "channels": [64, 128, 256, 512],
        "expansion": 4,
        "image_size": 224,
        "num_classes": 1000,
        "stem_channels": 64,
    },
    
    # TTS - CosyVoice
    "FunAudioLLM/CosyVoice-300M": {
        "llm_hidden": 1024,
        "llm_heads": 16,
        "llm_layers": 14,
        "flow_hidden": 512,
        "flow_heads": 8,
        "flow_layers": 6,
        "mel_bins": 80,
        "speech_token_size": 4096,
        "spk_embed_dim": 192,
    },
    "FunAudioLLM/CosyVoice2-0.5B": {
        "llm_hidden": 1280,
        "llm_heads": 20,
        "llm_layers": 18,
        "flow_hidden": 768,
        "flow_heads": 12,
        "flow_layers": 8,
        "mel_bins": 128,
        "speech_token_size": 6561,
        "spk_embed_dim": 256,
    },
    
    # Object Detection - YOLOv8
    "ultralytics/yolov8s": {
        "depth_mult": 0.33,
        "width_mult": 0.50,
        "base_channels": 64,
        "csp_e": 0.5,
        "num_classes": 80,
        "image_size": 640,
        "reg_max": 16,
    },
    "ultralytics/yolov8m": {
        "depth_mult": 0.67,
        "width_mult": 0.75,
        "base_channels": 96,
        "csp_e": 0.67,
        "num_classes": 80,
        "image_size": 640,
        "reg_max": 16,
    },
    "ultralytics/yolov8x": {
        "depth_mult": 1.0,
        "width_mult": 1.25,
        "base_channels": 160,
        "csp_e": 0.5,
        "num_classes": 80,
        "image_size": 640,
        "reg_max": 16,
    },
    
    # Object Detection - Deformable-DETR
    "SenseTime/deformable-detr-r50": {
        "hidden_dim": 256,
        "num_heads": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 1024,
        "num_queries": 300,
        "num_feature_levels": 4,
        "num_points": 4,
        "backbone": "resnet50",
    },
    "SenseTime/deformable-detr-r101": {
        "hidden_dim": 256,
        "num_heads": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 1024,
        "num_queries": 300,
        "num_feature_levels": 4,
        "num_points": 4,
        "backbone": "resnet101",
    },
    
    # Recommendation - DCN-v2
    "dcnv2-small": {
        "num_sparse_features": 26,
        "embedding_dim": 16,
        "cross_layers": 2,
        "cross_low_rank": 32,
        "deep_dims": [256, 128, 64],
        "num_dense_features": 13,
    },
    "dcnv2-base": {
        "num_sparse_features": 26,
        "embedding_dim": 16,
        "cross_layers": 3,
        "cross_low_rank": 64,
        "deep_dims": [512, 256, 128],
        "num_dense_features": 13,
    },
    "dcnv2-large": {
        "num_sparse_features": 26,
        "embedding_dim": 32,
        "cross_layers": 4,
        "cross_low_rank": 128,
        "deep_dims": [1024, 512, 256, 128],
        "num_dense_features": 13,
    },
    
    # Embedding - BGE-M3
    "BAAI/bge-m3": {
        "hidden_size": 1024,
        "num_heads": 16,
        "head_dim": 64,
        "intermediate_size": 4096,
        "num_layers": 24,
        "vocab_size": 250002,
        "max_seq_len": 8192,
        "pooling": "cls",
    },
    "Alibaba-NLP/gte-Qwen2-7B-instruct": {
        "hidden_size": 3584,
        "num_heads": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
        "intermediate_size": 18944,
        "num_layers": 28,
        "vocab_size": 152064,
        "max_seq_len": 131072,
        "pooling": "last_token",
    },
    
    # Speculative - EAGLE-3
    "EAGLE3-Llama3-8B": {
        "draft_hidden": 4096,
        "draft_layers": 1,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "tree_max_depth": 6,
        "tree_max_width": 64,
        "target_model": "meta-llama/Llama-3-8B",
    },
    "EAGLE3-Llama3-70B": {
        "draft_hidden": 8192,
        "draft_layers": 1,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "tree_max_depth": 6,
        "tree_max_width": 64,
        "target_model": "meta-llama/Llama-3-70B",
    },
    
    # Speculative - Lookahead
    "lookahead-W4-N5": {
        "window_size": 4,
        "ngram_size": 5,
        "max_verify_length": 20,
        "cache_size": 8192,
    },
    "lookahead-W7-N7": {
        "window_size": 7,
        "ngram_size": 7,
        "max_verify_length": 49,
        "cache_size": 16384,
    },
    
    # Neural Rendering - 3DGS
    "3dgs-1m": {
        "num_gaussians": 1000000,
        "sh_degree": 3,
        "sh_coeffs": 16,
        "position_dim": 3,
        "scale_dim": 3,
        "rotation_dim": 4,
        "opacity_dim": 1,
        "features_dim": 48,
        "image_size": 1024,
    },
    "3dgs-5m": {
        "num_gaussians": 5000000,
        "sh_degree": 3,
        "sh_coeffs": 16,
        "position_dim": 3,
        "scale_dim": 3,
        "rotation_dim": 4,
        "opacity_dim": 1,
        "features_dim": 48,
        "image_size": 2048,
    },
    
    # Neural Rendering - InstantNGP
    "instantngp-base": {
        "hash_table_size": 2**19,
        "num_levels": 16,
        "feature_per_level": 2,
        "base_resolution": 16,
        "max_resolution": 2048,
        "mlp_hidden": 64,
        "mlp_layers": 2,
        "density_mlp_layers": 1,
        "color_mlp_layers": 3,
    },
    "instantngp-large": {
        "hash_table_size": 2**22,
        "num_levels": 24,
        "feature_per_level": 4,
        "base_resolution": 16,
        "max_resolution": 8192,
        "mlp_hidden": 128,
        "mlp_layers": 4,
        "density_mlp_layers": 2,
        "color_mlp_layers": 4,
    },
    
    # Recommendation - DLRMv2
    "dlrmv2-small": {
        "num_dense_features": 13,
        "num_sparse_features": 26,
        "sparse_feature_size": 16,
        "vocab_sizes": [1000] * 26,
        "bot_mlp_sizes": [13, 512, 256, 16],
        "top_mlp_sizes": [367, 256, 1],
        "interaction_op": "dot",
    },
    "dlrmv2-base": {
        "num_dense_features": 13,
        "num_sparse_features": 26,
        "sparse_feature_size": 64,
        "vocab_sizes": [
            40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63,
            40000000, 3067956, 405282, 10, 2209, 11938, 155, 4, 976,
            14, 40000000, 40000000, 40000000, 590152, 12973, 108, 36,
        ],
        "bot_mlp_sizes": [13, 512, 256, 64],
        "top_mlp_sizes": [415, 512, 256, 1],
        "interaction_op": "dot",
    },
    "dlrmv2-large": {
        "num_dense_features": 13,
        "num_sparse_features": 26,
        "sparse_feature_size": 128,
        "vocab_sizes": [
            40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63,
            40000000, 3067956, 405282, 10, 2209, 11938, 155, 4, 976,
            14, 40000000, 40000000, 40000000, 590152, 12973, 108, 36,
        ],
        "bot_mlp_sizes": [13, 512, 256, 128],
        "top_mlp_sizes": [479, 1024, 512, 256, 1],
        "interaction_op": "dot",
    },
    
    # Recommendation - LightGCN
    "lightgcn-movielens-small": {
        "num_nodes": 3000,
        "embedding_dim": 64,
        "num_layers": 3,
    },
    "lightgcn-gowalla": {
        "num_nodes": 107092,
        "embedding_dim": 64,
        "num_layers": 3,
    },
    "lightgcn-amazon-book": {
        "num_nodes": 105283,
        "embedding_dim": 64,
        "num_layers": 3,
    },
}


def load_hf_config(model_name_or_path: str) -> Dict[str, Any]:
    """
    Load configuration from HuggingFace Hub.
    
    Args:
        model_name_or_path: HuggingFace model ID or local path
        
    Returns:
        Configuration dictionary
    """
    # First try to load from HuggingFace
    try:
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        config_dict = hf_config.to_dict()
        
        # Normalize common field names
        config_dict = _normalize_config(config_dict)
        logger.info(f"Loaded config from HuggingFace: {model_name_or_path}")
        return config_dict
        
    except Exception as e:
        logger.warning(f"Failed to load from HuggingFace: {e}")
    
    # Fallback to hardcoded configs
    if model_name_or_path in HARDCODED_CONFIGS:
        logger.info(f"Using hardcoded config for: {model_name_or_path}")
        return HARDCODED_CONFIGS[model_name_or_path].copy()
    
    # Try partial matching
    for key, config in HARDCODED_CONFIGS.items():
        if model_name_or_path.lower() in key.lower() or key.lower() in model_name_or_path.lower():
            logger.info(f"Using hardcoded config (partial match): {key}")
            return config.copy()
    
    raise ValueError(f"Could not find config for: {model_name_or_path}")


def _normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize HuggingFace config field names to our standard format.
    """
    # Common mappings
    mappings = {
        "d_model": "hidden_size",
        "n_heads": "num_heads",
        "n_layers": "num_layers",
        "n_embd": "hidden_size",
        "n_head": "num_heads",
        "n_layer": "num_layers",
        "dim": "hidden_size",
        "n_kv_heads": "num_kv_heads",
        "ffn_dim": "intermediate_size",
        "d_ff": "intermediate_size",
        "max_position_embeddings": "max_seq_len",
    }
    
    normalized = {}
    for key, value in config.items():
        new_key = mappings.get(key, key)
        normalized[new_key] = value
    
    return normalized


def get_available_configs() -> List[str]:
    """Get list of all available hardcoded configurations."""
    return list(HARDCODED_CONFIGS.keys())


def get_config_info(model_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration info for a model without loading full config."""
    if model_name in HARDCODED_CONFIGS:
        return HARDCODED_CONFIGS[model_name]
    return None
