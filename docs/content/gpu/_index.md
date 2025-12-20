---
title: GPU Acceleration Guide
linktitle: GPU Guide
summary: Complete guide to GPU-accelerated EPOCH simulations
weight: 1

# Page metadata.
date: "2025-12-19T00:00:00Z"
lastmod: "2025-12-19T00:00:00Z"
draft: false
toc: true
type: docs

# Add menu entry to sidebar.
menu:
  gpu:
    name: Overview
    weight: 1
---

> **Verified**: All tests passed on Tesla V100-PCIE-32GB (December 2025)

EPOCH-GPU extends the EPOCH Particle-In-Cell code with NVIDIA GPU acceleration, 
enabling high-performance plasma physics simulations on modern GPU hardware.

## Key Features

- **Full Dimensional Support**: GPU acceleration for 1D, 2D, and 3D simulations
- **NVIDIA GPU Optimized**: Supports V100, A100, H100 and other CUDA-capable GPUs
- **High Performance**: Up to 5+ billion particles/second on a single V100
- **OpenACC + CUDA Fortran**: Hybrid approach for maximum performance
- **Drop-in Replacement**: Compatible with existing EPOCH input decks

## Performance Highlights

| Dimension | Kernel | Configuration | Throughput |
|-----------|--------|---------------|------------|
| 1D | Particle Push | 1M particles | **5.98B particles/sec** |
| 2D | Particle Push | 1M particles | **5.30B particles/sec** |
| 2D | Field Update | 256×256 grid | **1.87B cells/sec** |
| 3D | Particle Push | 500K particles | **4.30B particles/sec** |
| 3D | Field Update | 64³ grid | **2.63B cells/sec** |

## Repository Links

- **GitHub**: [chenxingqiang/EPOCH-GPU](https://github.com/chenxingqiang/EPOCH-GPU)
- **PyPI**: [epoch-gpu](https://pypi.org/project/epoch-gpu/)
- **Upstream EPOCH**: [Warwick-Plasma/epoch](https://github.com/Warwick-Plasma/epoch)

## Getting Started

1. [Prerequisites]({{< relref "prerequisites.md" >}})
2. [Building EPOCH-GPU]({{< relref "building.md" >}})
3. [Running GPU Simulations]({{< relref "running.md" >}})
4. [**Tutorial: GPU Particle Simulations**]({{< relref "tutorial.md" >}})
5. [GPU Architecture]({{< relref "architecture.md" >}})
6. [Performance Tuning]({{< relref "performance.md" >}})
7. [Multi-GPU Support]({{< relref "multi_gpu.md" >}})
8. [Troubleshooting]({{< relref "troubleshooting.md" >}})
9. [Testing & Validation]({{< relref "testing.md" >}})

