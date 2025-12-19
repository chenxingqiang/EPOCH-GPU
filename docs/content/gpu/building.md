---
title: Building EPOCH-GPU
linktitle: Building
toc: true
type: docs
date: "2024-12-19T00:00:00Z"
draft: false
menu:
  gpu:
    parent: GPU Guide
    weight: 3
---

# Building EPOCH-GPU

## Clone the Repository

```bash
# Using SSH (recommended)
git clone --recursive git@github.com:chenxingqiang/EPOCH-GPU.git
cd EPOCH-GPU

# Or using HTTPS
git clone --recursive https://github.com/chenxingqiang/EPOCH-GPU.git
cd EPOCH-GPU
```

The `--recursive` flag is essential to download the SDF submodule.

## Build Commands

### Building 1D GPU Version

```bash
cd epoch1d
make -f Makefile.gpu GPU_CC=cc70
```

### Building 2D GPU Version

```bash
cd epoch2d
make -f Makefile.gpu GPU_CC=cc70
```

### Building 3D GPU Version

```bash
cd epoch3d
make -f Makefile.gpu GPU_CC=cc70
```

## GPU Architecture Options

Set `GPU_CC` according to your GPU:

| GPU | Compute Capability | Flag |
|-----|-------------------|------|
| Tesla V100 | 7.0 | `GPU_CC=cc70` |
| Tesla A100 | 8.0 | `GPU_CC=cc80` |
| Tesla H100 | 9.0 | `GPU_CC=cc90` |
| RTX 3090 | 8.6 | `GPU_CC=cc86` |
| RTX 4090 | 8.9 | `GPU_CC=cc89` |

## Build Options

```bash
# Full rebuild
make -f Makefile.gpu clean
make -f Makefile.gpu GPU_CC=cc70

# Debug build
make -f Makefile.gpu GPU_CC=cc70 MODE=debug

# Verbose compilation
make -f Makefile.gpu GPU_CC=cc70 VERBOSE=1
```

## Verify Build

After successful compilation, binaries are located in:

```
epoch1d/bin/epoch1d_gpu
epoch2d/bin/epoch2d_gpu
epoch3d/bin/epoch3d_gpu
```

Test the binary:

```bash
./bin/epoch2d_gpu --version
```

## Building GPU Tests

To compile and run GPU kernel tests:

```bash
cd epoch2d/src/gpu

# Compile test modules
for f in gpu_config particle_data_gpu fields_gpu particles_gpu; do
  nvfortran -c -O3 -r8 -acc -gpu=cc70 -cuda -DGPU_ENABLED $f.F90
done

# Compile and link test program
nvfortran -O3 -r8 -acc -gpu=cc70 -cuda -DGPU_ENABLED \
  -o test_gpu_2d \
  gpu_config.o particle_data_gpu.o fields_gpu.o particles_gpu.o \
  test_gpu_2d.F90

# Run test
./test_gpu_2d
```

## Troubleshooting Build Issues

### Missing nvfortran

```
error: nvfortran: command not found
```

**Solution**: Add NVIDIA HPC SDK to PATH:
```bash
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/bin:$PATH
```

### Wrong GPU Architecture

```
NVFORTRAN-W-0155-Unrecognized target architecture
```

**Solution**: Use correct `GPU_CC` for your GPU. Check with:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

### SDF Submodule Missing

```
fatal: Unable to read MODULE file sdf.mod
```

**Solution**: Initialize submodules:
```bash
git submodule update --init --recursive
```

