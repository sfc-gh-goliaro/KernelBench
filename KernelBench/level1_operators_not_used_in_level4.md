# Level1 Operators Not Used in Level4

This document tracks level1 operators that are currently not being imported or used by any level4 model implementations.

## Summary

- **Total level1 operators:** 233
- **Used in level4:** 37
- **Not used in level4:** 196

---

## Operators Used in Level4

For reference, these 37 operators ARE currently imported and used in level4 models:

| Category | Operators Used |
|----------|---------------|
| activations | `_1_ReLU`, `_3_Sigmoid`, `_4_Tanh`, `_5_Softmax`, `_7_Swish`, `_8_GELU`, `_11_Softplus` |
| attention | `_3_GroupedQueryAttention`, `_4_MultiQueryAttention`, `_5_MultiHeadLatentAttention`, `_6_ALiBi` |
| convolutions | `_1_Conv2d_Standard` |
| embeddings | `_1_RotaryEmbedding`, `_2_Embedding`, `_3_SinusoidalPosEmbed`, `_4_RelativePositionBias` |
| matmul | `_1_MatMul`, `_10_Linear` |
| moe | `_3_FusedMoE`, `_6_SharedFusedMoE` |
| normalization | `_1_BatchNorm`, `_3_GroupNorm`, `_4_RMSNorm`, `_6_LayerNorm`, `_7_RMSNormGated` |
| pooling | `_2_MaxPool2d`, `_7_AdaptiveAvgPool2d`, `_9_MeanPooling` |
| ssm | `_1_MambaCausalConv1d`, `_2_MambaCausalConv1dStep`, `_3_Mamba2SSDChunkedScan`, `_4_Mamba2StateUpdateStep`, `_5_Mamba1SelectiveScan`, `_6_Mamba1SSMStep`, `_7_RWKV6NaiveRecurrent`, `_8_RWKV6TokenShift`, `_9_RWKV6LerpLinear` |

---

## Operators NOT Used in Level4

### activations/ (11 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_2_LeakyReLU.py` | LeakyReLU | Leaky ReLU activation |
| `_4_Tanh.py` | Tanh | Hyperbolic tangent activation |
| `_6_LogSoftmax.py` | LogSoftmax | Log-softmax activation |
| `_9_SELU.py` | SELU | Scaled Exponential Linear Unit |
| `_10_HardSigmoid.py` | HardSigmoid | Hard sigmoid activation |
| `_12_Softsign.py` | Softsign | Softsign activation |
| `_13_ELU.py` | ELU | Exponential Linear Unit |
| `_14_HardTanh.py` | HardTanh | Hard tanh activation |
| `_15_HardSwish.py` | HardSwish | Hard swish activation |
| `_16_SiluAndMul.py` | SiluAndMul | Fused SiLU + multiply |
| `_17_GeluAndMul.py` | GeluAndMul | Fused GELU + multiply |

### attention/ (7 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_ScaledDotProductAttention.py` | ScaledDotProductAttention | Basic scaled dot-product attention |
| `_2_MultiHeadAttention_Causal.py` | MultiHeadAttention_Causal | Multi-head attention with causal mask |
| `_3_GroupedQueryAttention.py` | GroupedQueryAttention | GQA used in Llama-2/3 |
| `_4_MultiQueryAttention.py` | MultiQueryAttention | MQA used in Falcon |
| `_5_MultiHeadLatentAttention.py` | MultiHeadLatentAttention | MLA used in DeepSeek-V2 |
| `_8_ShiftedWindowAttention.py` | ShiftedWindowAttention | Shifted window attention for Swin |
| `_9_ChunkedLocalAttention.py` | ChunkedLocalAttention | Chunked local attention |

### audio/ (5 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_MelSpectrogram.py` | MelSpectrogram | Mel spectrogram computation |
| `_2_AudioConvEncoder.py` | AudioConvEncoder | Audio convolutional encoder |
| `_3_Vocoder.py` | Vocoder | Vocoder for audio synthesis |
| `_4_STFT.py` | STFT | Short-time Fourier transform |
| `_5_AudioFeatureExtractor.py` | AudioFeatureExtractor | Audio feature extraction |

### communication/ (7 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_AllReduce.py` | AllReduce | Distributed all-reduce |
| `_2_AllGather.py` | AllGather | Distributed all-gather |
| `_3_ReduceScatter.py` | ReduceScatter | Distributed reduce-scatter |
| `_4_AllToAll.py` | AllToAll | Distributed all-to-all |
| `_5_SendRecv.py` | SendRecv | Point-to-point send/receive |
| `_6_TensorParallelAllGather.py` | TensorParallelAllGather | TP all-gather |
| `_7_ExpertParallelAllToAll.py` | ExpertParallelAllToAll | Expert parallel all-to-all |

### convolutions/ (34 operators not used)

| File | Operator Name |
|------|---------------|
| `_2_Conv3d_Standard.py` | Conv3d_Standard |
| `_3_Conv2d_AsymInput.py` | Conv2d_AsymInput |
| `_4_Conv2d_AsymKernel.py` | Conv2d_AsymKernel |
| `_5_Conv3d_AsymInput.py` | Conv3d_AsymInput |
| `_6_Conv3d_AsymKernel.py` | Conv3d_AsymKernel |
| `_7_Conv2d_SquareAsym.py` | Conv2d_SquareAsym |
| `_8_Conv2d_Square.py` | Conv2d_Square |
| `_9_Conv3d_AsymBoth.py` | Conv3d_AsymBoth |
| `_10_Conv1d_Standard.py` | Conv1d_Standard |
| `_11_Conv1d_DilatedStrided.py` | Conv1d_DilatedStrided |
| `_12_Conv2d_DilatedPadded.py` | Conv2d_DilatedPadded |
| `_13_ConvTranspose2d_Square.py` | ConvTranspose2d_Square |
| `_14_ConvTranspose3d_Asym.py` | ConvTranspose3d_Asym |
| `_15_ConvTranspose3d_Square.py` | ConvTranspose3d_Square |
| `_16_ConvTranspose1d.py` | ConvTranspose1d |
| `_17_ConvTranspose2d_AsymKernel.py` | ConvTranspose2d_AsymKernel |
| `_18_ConvTranspose3d_AsymKernel.py` | ConvTranspose3d_AsymKernel |
| `_19_ConvTranspose2d_AsymBoth.py` | ConvTranspose2d_AsymBoth |
| `_20_ConvTranspose3d_AsymInput.py` | ConvTranspose3d_AsymInput |
| `_21_ConvTranspose2d_AsymInput.py` | ConvTranspose2d_AsymInput |
| `_22_ConvTranspose3d_StridedGrouped.py` | ConvTranspose3d_StridedGrouped |
| `_23_ConvTranspose3d_StridedPadded.py` | ConvTranspose3d_StridedPadded |
| `_24_ConvTranspose1d_Dilated.py` | ConvTranspose1d_Dilated |
| `_25_ConvTranspose2d_Full.py` | ConvTranspose2d_Full |
| `_26_ConvTranspose3d_Full.py` | ConvTranspose3d_Full |
| `_27_ConvTranspose2d_Padded.py` | ConvTranspose2d_Padded |
| `_28_ConvTranspose1d_Full.py` | ConvTranspose1d_Full |
| `_29_ConvTranspose2d_DilatedStrided.py` | ConvTranspose2d_DilatedStrided |
| `_30_DepthwiseConv2d_Square.py` | DepthwiseConv2d_Square |
| `_31_DepthwiseConv2d_AsymKernel.py` | DepthwiseConv2d_AsymKernel |
| `_32_DepthwiseConv2d_AsymInput.py` | DepthwiseConv2d_AsymInput |
| `_33_DepthwiseConv2d_AsymBoth.py` | DepthwiseConv2d_AsymBoth |
| `_34_DepthwiseSeparable.py` | DepthwiseSeparable |
| `_35_PointwiseConv2d.py` | PointwiseConv2d |

### detection/ (12 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_CSPBlock.py` | CSPBlock | Cross Stage Partial block |
| `_2_C2f.py` | C2f | C2f block for YOLOv8 |
| `_3_SPPF.py` | SPPF | Spatial Pyramid Pooling Fast |
| `_4_FPN_Concat.py` | FPN_Concat | FPN concatenation |
| `_5_PAN_Upsample.py` | PAN_Upsample | PAN upsampling |
| `_6_DetectionHead.py` | DetectionHead | Detection head |
| `_7_AnchorFreeHead.py` | AnchorFreeHead | Anchor-free detection head |
| `_8_NMS.py` | NMS | Non-maximum suppression |
| `_9_DeformableAttention.py` | DeformableAttention | Deformable attention |
| `_10_ObjectQueries.py` | ObjectQueries | Object queries for DETR |
| `_11_HungarianMatcher.py` | HungarianMatcher | Hungarian matching |
| `_12_PositionalEncoding2D.py` | PositionalEncoding2D | 2D positional encoding |

### diffusion/ (6 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_AdaLN.py` | AdaLN | Adaptive layer normalization |
| `_2_AdaLN_Zero.py` | AdaLN_Zero | AdaLN-Zero for DiT |
| `_3_TimestepEmbedding.py` | TimestepEmbedding | Timestep embedding |
| `_4_CFG_Scale.py` | CFG_Scale | Classifier-free guidance scaling |
| `_5_UNet_ResBlock.py` | UNet_ResBlock | UNet residual block |
| `_6_SkipConnection.py` | SkipConnection | Skip connection |

### embeddings/ (1 operator not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_2_Embedding.py` | Embedding | Standard embedding lookup |

### loss/ (8 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_MSELoss.py` | MSELoss | Mean squared error loss |
| `_2_CrossEntropyLoss.py` | CrossEntropyLoss | Cross-entropy loss |
| `_3_HuberLoss.py` | HuberLoss | Huber loss |
| `_4_KLDivLoss.py` | KLDivLoss | KL divergence loss |
| `_5_TripletMarginLoss.py` | TripletMarginLoss | Triplet margin loss |
| `_6_HingeLoss.py` | HingeLoss | Hinge loss |
| `_7_ContrastiveLoss.py` | ContrastiveLoss | Contrastive loss |
| `_8_L1Loss.py` | L1Loss | L1 loss |

### matmul/ (8 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_2_BatchedMatMul.py` | BatchedMatMul | Batched matrix multiplication |
| `_3_DiagonalMatMul.py` | DiagonalMatMul | Diagonal matrix multiplication |
| `_4_SymmetricMatMul.py` | SymmetricMatMul | Symmetric matrix multiplication |
| `_5_UpperTriangularMatMul.py` | UpperTriangularMatMul | Upper triangular matmul |
| `_6_LowerTriangularMatMul.py` | LowerTriangularMatMul | Lower triangular matmul |
| `_7_TransposedA.py` | TransposedA | Matmul with transposed A |
| `_8_TransposedB.py` | TransposedB | Matmul with transposed B |
| `_9_TransposedBoth.py` | TransposedBoth | Matmul with both transposed |

### mobile/ (5 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_InvertedResidual.py` | InvertedResidual | MobileNet inverted residual |
| `_2_SqueezeExcitation.py` | SqueezeExcitation | SE block |
| `_3_SqueezeExcitation_HardSig.py` | SqueezeExcitation_HardSig | SE with hard sigmoid |
| `_4_StochasticDepth.py` | StochasticDepth | Stochastic depth |
| `_5_MBConv.py` | MBConv | Mobile inverted bottleneck conv |

### model_merging/ (8 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_WeightAverage.py` | WeightAverage | Weight averaging |
| `_2_SLERP.py` | SLERP | Spherical linear interpolation |
| `_3_TIES.py` | TIES | TIES merging |
| `_4_DARE.py` | DARE | DARE merging |
| `_5_TaskArithmetic.py` | TaskArithmetic | Task arithmetic |
| `_6_FisherWeighted.py` | FisherWeighted | Fisher-weighted merging |
| `_7_TaskVector.py` | TaskVector | Task vector computation |
| `_8_ModelSoups.py` | ModelSoups | Model soups |

### moe/ (5 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_2_Expert_Dispatch.py` | Expert_Dispatch | Expert dispatching |
| `_3_FusedMoE.py` | FusedMoE | Fused MoE computation |
| `_4_Expert_Combine.py` | Expert_Combine | Expert output combination |
| `_5_Aux_Loss_LoadBalance.py` | Aux_Loss_LoadBalance | Load balancing aux loss |
| `_6_SharedFusedMoE.py` | SharedFusedMoE | Shared expert fused MoE |

### normalization/ (2 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_2_InstanceNorm.py` | InstanceNorm | Instance normalization |
| `_5_VectorNorm.py` | VectorNorm | Vector normalization |

### optimizers/ (5 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_Adam.py` | Adam | Adam optimizer step |
| `_2_AdamW.py` | AdamW | AdamW optimizer step |
| `_3_SGD.py` | SGD | SGD optimizer step |
| `_4_LAMB.py` | LAMB | LAMB optimizer step |
| `_5_Adafactor.py` | Adafactor | Adafactor optimizer step |

### peft/ (4 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_LoRA_Linear.py` | LoRA_Linear | LoRA linear layer |
| `_2_QLoRA_Linear.py` | QLoRA_Linear | QLoRA linear layer |
| `_3_DoRA_Linear.py` | DoRA_Linear | DoRA linear layer |
| `_4_AdapterLayer.py` | AdapterLayer | Adapter layer |

### pooling/ (7 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_MaxPool1d.py` | MaxPool1d | 1D max pooling |
| `_3_MaxPool3d.py` | MaxPool3d | 3D max pooling |
| `_4_AvgPool1d.py` | AvgPool1d | 1D average pooling |
| `_5_AvgPool2d.py` | AvgPool2d | 2D average pooling |
| `_6_AvgPool3d.py` | AvgPool3d | 3D average pooling |
| `_8_GlobalAveragePooling.py` | GlobalAveragePooling | Global average pooling |
| `_10_LastTokenPooling.py` | LastTokenPooling | Last token pooling |

### quantization/ (7 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_W4A16_GEMM.py` | W4A16_GEMM | W4A16 quantized GEMM |
| `_2_FP8_GEMM.py` | FP8_GEMM | FP8 GEMM |
| `_3_KVCache_Quantize.py` | KVCache_Quantize | KV cache quantization |
| `_4_Dynamic_Quantize.py` | Dynamic_Quantize | Dynamic quantization |
| `_5_AWQQuantize.py` | AWQQuantize | AWQ quantization |
| `_6_GPTQDequant.py` | GPTQDequant | GPTQ dequantization |
| `_7_SmoothQuantScale.py` | SmoothQuantScale | SmoothQuant scaling |

### recommendation/ (10 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_SparseEmbedding.py` | SparseEmbedding | Sparse embedding lookup |
| `_2_FeatureCrossing.py` | FeatureCrossing | Feature crossing |
| `_3_FactorizationMachine.py` | FactorizationMachine | Factorization machine |
| `_4_SelfAttn_Features.py` | SelfAttn_Features | Self-attention for features |
| `_5_WideComponent.py` | WideComponent | Wide component |
| `_6_DeepComponent.py` | DeepComponent | Deep component |
| `_7_CrossNetwork.py` | CrossNetwork | Cross network |
| `_8_HashEmbedding.py` | HashEmbedding | Hash embedding |
| `_9_DotProduct.py` | DotProduct | Dot product |
| `_10_CosineSimilarity.py` | CosineSimilarity | Cosine similarity |

### reductions/ (4 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_Sum.py` | Sum | Sum reduction |
| `_2_MinMax.py` | MinMax | Min/max reduction |
| `_3_ArgMinMax.py` | ArgMinMax | Argmin/argmax |
| `_4_Scan.py` | Scan | Parallel scan |

### rendering/ (8 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_GaussianSplatting.py` | GaussianSplatting | 3DGS rendering |
| `_2_NeRFMLP.py` | NeRFMLP | NeRF MLP |
| `_3_VolumeRendering.py` | VolumeRendering | Volume rendering |
| `_4_RayMarching.py` | RayMarching | Ray marching |
| `_5_SHCoefficients.py` | SHCoefficients | Spherical harmonics |
| `_6_PositionalEncoding3D.py` | PositionalEncoding3D | 3D positional encoding |
| `_7_PoissonReconstruction.py` | PoissonReconstruction | Poisson reconstruction |
| `_8_TileRasterization.py` | TileRasterization | Tile-based rasterization |

### sampling/ (3 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_UnifiedSampler.py` | UnifiedSampler | Unified sampling |
| `_2_RepetitionPenalty.py` | RepetitionPenalty | Repetition penalty |
| `_3_PresenceFrequencyPenalty.py` | PresenceFrequencyPenalty | Presence/frequency penalty |

### speculative/ (9 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_TreeAttention.py` | TreeAttention | Tree attention for speculative decoding |
| `_2_VerificationSampling.py` | VerificationSampling | Verification sampling |
| `_3_DraftHead.py` | DraftHead | Draft head for speculative decoding |
| `_4_FeatureFusion.py` | FeatureFusion | Feature fusion |
| `_5_NGramPool.py` | NGramPool | N-gram pooling |
| `_6_JacobiIteration.py` | JacobiIteration | Jacobi iteration |
| `_7_EarlyExit.py` | EarlyExit | Early exit |
| `_8_TreeExpansion.py` | TreeExpansion | Tree expansion |
| `_9_TreePruning.py` | TreePruning | Tree pruning |

### ssm/ (2 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_10_GLA_Recurrence.py` | GLA_Recurrence | GLA recurrence |
| `_11_Retention.py` | Retention | RetNet retention |

### tts/ (8 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_SpeechTokenizer.py` | SpeechTokenizer | Speech tokenization |
| `_2_FlowMatching.py` | FlowMatching | Flow matching |
| `_3_DurationPredictor.py` | DurationPredictor | Duration prediction |
| `_4_PitchPredictor.py` | PitchPredictor | Pitch prediction |
| `_5_DualARDecoder.py` | DualARDecoder | Dual autoregressive decoder |
| `_6_VocoderDecoder.py` | VocoderDecoder | Vocoder decoder |
| `_7_ZeroShotVoiceClone.py` | ZeroShotVoiceClone | Zero-shot voice cloning |
| `_8_EmotionControl.py` | EmotionControl | Emotion control |

### upsampling/ (2 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_PixelShuffle.py` | PixelShuffle | Pixel shuffle upsampling |
| `_2_Interpolate.py` | Interpolate | Interpolation upsampling |

### vae/ (5 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_Reparameterize.py` | Reparameterize | Reparameterization trick |
| `_2_KLDiv_Gaussian.py` | KLDiv_Gaussian | KL divergence for Gaussians |
| `_3_VectorQuantize.py` | VectorQuantize | Vector quantization |
| `_4_GumbelSoftmax.py` | GumbelSoftmax | Gumbel softmax |
| `_5_ReconstructionLoss.py` | ReconstructionLoss | Reconstruction loss |

### vision/ (8 operators not used)

| File | Operator Name | Description |
|------|---------------|-------------|
| `_1_PatchEmbed2D.py` | PatchEmbed2D | 2D patch embedding |
| `_2_PatchEmbed3D.py` | PatchEmbed3D | 3D patch embedding |
| `_3_PatchMerging.py` | PatchMerging | Patch merging |
| `_4_CLS_Pooling.py` | CLS_Pooling | CLS token pooling |
| `_5_VisionProjection.py` | VisionProjection | Vision projection |
| `_6_RoPE_3D.py` | RoPE_3D | 3D rotary position embedding |
| `_7_Resampler.py` | Resampler | Vision resampler |
| `_8_ResidualBlock.py` | ResidualBlock | Residual block |

---

## Notes

- Many level4 models implement their own attention and MLP layers using `nn.Linear` directly rather than importing level1 attention operators. This is because the level1 attention operators are more "complete" (include Q/K/V projections) while level4 models often need custom implementations.

- The Mamba SSM operators are used by `5_Mamba2.py` and `28_Mamba1.py`. The RWKV6 SSM operators are used by `6_RWKV6.py`. The GLA and RetNet SSM operators are not yet used because `7_GLA.py` and `8_RetNet.py` implement their core computations inline.

- The rendering operators (`GaussianSplatting`, `NeRFMLP`, etc.) are not used by `26_3DGS.py` and `27_InstantNGP.py` which implement their rendering logic inline.

- Future work could involve refactoring level4 models to use more level1 operators to better separate concerns and enable more granular benchmarking.
