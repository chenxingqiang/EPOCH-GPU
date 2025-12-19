---
title: Prerequisites
linktitle: Prerequisites
toc: true
type: docs
date: "2024-12-19T00:00:00Z"
draft: false
menu:
  gpu:
    parent: GPU Guide
    weight: 2
---

## Hardware Requirements

### Supported GPUs

EPOCH-GPU supports NVIDIA GPUs with compute capability 7.0 or higher:

| GPU | Compute Capability | Memory | Recommended For |
|-----|-------------------|--------|-----------------|
| Tesla V100 | 7.0 | 16/32 GB | Production runs |
| A100 | 8.0 | 40/80 GB | Large-scale simulations |
| H100 | 9.0 | 80 GB | Latest performance |
| RTX 3090/4090 | 8.6/8.9 | 24 GB | Development/testing |

### Memory Requirements

- Minimum: 8 GB GPU memory
- Recommended: 16+ GB for production simulations
- Memory scales with:
  - Grid size: ~72 bytes per cell (9 field arrays × 8 bytes)
  - Particles: ~56 bytes per particle (7 arrays × 8 bytes)

## Software Requirements

### NVIDIA HPC SDK

The NVIDIA HPC SDK provides the `nvfortran` compiler required for GPU acceleration.

**Installation on Linux:**

```bash
# Download from NVIDIA website or use package manager
# Ubuntu/Debian
wget https://developer.download.nvidia.com/hpc-sdk/24.11/nvhpc-24-11_24.11_amd64.deb
sudo apt install ./nvhpc-24-11_24.11_amd64.deb

# Setup environment
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/lib:$LD_LIBRARY_PATH
```

**Verify installation:**

```bash
nvfortran --version
# Expected: nvfortran 24.11-0 64-bit target on ...
```

### CUDA Toolkit

CUDA 11.0 or later is required (usually bundled with HPC SDK).

```bash
nvcc --version
# Expected: Cuda compilation tools, release 12.x
```

### MPI Library

GPU-aware MPI is recommended for multi-GPU simulations:

```bash
# Check if MPI is available
mpirun --version

# For NVIDIA HPC SDK's MPI:
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/comm_libs/mpi/bin:$PATH
```

## Verification

Run this script to verify your environment:

```bash
#!/bin/bash
echo "=== EPOCH-GPU Environment Check ==="

echo -n "nvfortran: "
nvfortran --version 2>/dev/null | head -1 || echo "NOT FOUND"

echo -n "CUDA: "
nvcc --version 2>/dev/null | grep release || echo "NOT FOUND"

echo -n "MPI: "
mpirun --version 2>/dev/null | head -1 || echo "NOT FOUND"

echo -n "GPU: "
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "NOT FOUND"

echo "==================================="
```

