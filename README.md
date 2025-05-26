# Multi-GPU Inference for Stable Video Diffusion Models

*Thanks to [ControlNeXt](https://github.com/dvlab-research/ControlNeXt/tree/main/ControlNeXt-SVD-v2) and [Video-Infinity](https://github.com/Yuanshi9815/Video-Infinity) for their implementations that laid the groundwork for this project.*

## Overview
This project accelerates video diffusion inference by distributing generation across multiple GPUs.

## Challenge
Splitting a video into GPU-specific segments introduces visible seams at segment boundaries due to missing temporal context. Below is an example illustrating this issue. Naively sharing full context at every denoising step leads to heavy synchronization and communication overhead.

<img 
  src="https://github.com/user-attachments/assets/2e2d6faa-c2a0-4675-885e-3348c573a039" 
  alt="Multi-GPU-WO-Exchange" 
  width="600" 
  height="600" 
/>
## Solution
We optimize context exchange by sharing **only** the critical temporal-attention tensors during the U-Net’s denoising loop. This targeted approach dramatically reduces inter-GPU communication while preserving smooth, coherent video outputs.

![Picture1](https://github.com/user-attachments/assets/01fb576b-21e8-4ef8-a5d6-7619ae1a0276)



