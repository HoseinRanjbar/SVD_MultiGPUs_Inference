# SVD_MultiGPUs_Inference

# Multi-GPU Inference for Stable Video Diffusion Models

## Overview
This project accelerates video diffusion inference by distributing generation across multiple GPUs.

## Challenge
Splitting a video into GPU-specific segments introduces visible seams at segment boundaries due to missing temporal context. Naively sharing full context at every denoising step leads to heavy synchronization and communication overhead.

## Solution
We optimize context exchange by sharing **only** the critical temporal-attention tensors during the U-Net’s denoising loop. This targeted approach dramatically reduces inter-GPU communication while preserving smooth, coherent video outputs.

