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

## Step-by-Step Build Guide

### Complete Build Process

```bash
# 1. Verify environment
nvfortran --version
nvidia-smi

# 2. Clone repository
git clone --recursive https://github.com/chenxingqiang/EPOCH-GPU.git
cd EPOCH-GPU

# 3. Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Example output: 7.0 (for V100)

# 4. Build SDF library first (if needed)
cd SDF/FORTRAN
make COMPILER=nvidia
cd ../..

# 5. Build EPOCH-GPU (choose your dimension)
cd epoch2d
make -f Makefile.gpu GPU_CC=cc70 -j4

# 6. Verify binary
ls -la bin/epoch2d_gpu
./bin/epoch2d_gpu --version 2>/dev/null || echo "Binary created successfully"
```

### Build All Dimensions

```bash
#!/bin/bash
# build_all_gpu.sh

GPU_CC=${1:-cc70}
echo "Building with GPU_CC=$GPU_CC"

for dim in 1d 2d 3d; do
    echo "=== Building epoch$dim ==="
    cd epoch$dim
    make -f Makefile.gpu clean
    make -f Makefile.gpu GPU_CC=$GPU_CC -j$(nproc)
    if [ -f bin/epoch${dim}_gpu ]; then
        echo "✓ epoch$dim build successful"
    else
        echo "✗ epoch$dim build failed"
        exit 1
    fi
    cd ..
done

echo "=== All builds completed ==="
```

Run with:
```bash
chmod +x build_all_gpu.sh
./build_all_gpu.sh cc70  # For V100
./build_all_gpu.sh cc80  # For A100
```

## Advanced Build Options

### Makefile Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `GPU_CC` | GPU compute capability | cc70 | `GPU_CC=cc80` |
| `MODE` | Build mode | release | `MODE=debug` |
| `VERBOSE` | Show commands | 0 | `VERBOSE=1` |
| `DEFINE` | Preprocessor flags | - | `DEFINE="-DSINGLE"` |
| `MPIF90` | MPI wrapper | mpif90 | `MPIF90=mpiifort` |

### Build Configurations

```bash
# Production build (optimized)
make -f Makefile.gpu GPU_CC=cc70 MODE=release

# Debug build (with symbols, no optimization)
make -f Makefile.gpu GPU_CC=cc70 MODE=debug

# Single precision (saves memory)
make -f Makefile.gpu GPU_CC=cc70 DEFINE="-DSINGLE"

# With particle IDs
make -f Makefile.gpu GPU_CC=cc70 DEFINE="-DPARTICLE_ID"

# Verbose output (see all commands)
make -f Makefile.gpu GPU_CC=cc70 VERBOSE=1

# Parallel build (faster)
make -f Makefile.gpu GPU_CC=cc70 -j8
```

### Cross-Compilation for Different GPUs

```bash
# Build for multiple GPU architectures
# This creates a binary that works on V100, A100, and H100

# Method 1: Build separate binaries
make -f Makefile.gpu GPU_CC=cc70 -o bin/epoch2d_v100
make -f Makefile.gpu clean
make -f Makefile.gpu GPU_CC=cc80 -o bin/epoch2d_a100

# Method 2: Fat binary (if supported)
# Edit Makefile.gpu to add: -gpu=cc70,cc80,cc90
```

## Troubleshooting Build Issues

### Issue 1: "nvfortran: command not found"

**Cause**: NVIDIA HPC SDK not in PATH

**Solutions**:

```bash
# Find where nvfortran is installed
find /opt -name "nvfortran" 2>/dev/null
# Or
locate nvfortran

# Add to PATH (adjust path as needed)
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/bin:$PATH

# Make permanent
echo 'export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Issue 2: "Unrecognized target architecture"

**Error message**:
```
NVFORTRAN-W-0155-Unrecognized target architecture
```

**Cause**: Invalid GPU_CC value

**Solution**:
```bash
# Check your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Example: 7.0

# Use the correct flag
# For 7.0 -> cc70
# For 8.0 -> cc80
# For 8.6 -> cc86
make -f Makefile.gpu GPU_CC=cc70
```

### Issue 3: "Unable to read MODULE file sdf.mod"

**Cause**: SDF library not built or submodule missing

**Solutions**:

```bash
# Solution 1: Initialize submodules
git submodule update --init --recursive

# Solution 2: Build SDF manually
cd SDF/FORTRAN
make COMPILER=nvidia
cd ../..

# Solution 3: Clean and rebuild
make -f Makefile.gpu cleanall
make -f Makefile.gpu GPU_CC=cc70
```

### Issue 4: "undefined reference to `cudaXXX`"

**Cause**: CUDA libraries not linked

**Solution**:
```bash
# Check CUDA library path
ls /usr/local/cuda/lib64/libcudart.so

# Set library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Or if using HPC SDK's CUDA
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/lib64:$LD_LIBRARY_PATH
```

### Issue 5: "Out of memory" during compilation

**Cause**: nvfortran uses a lot of memory for optimization

**Solutions**:

```bash
# Solution 1: Reduce parallel jobs
make -f Makefile.gpu GPU_CC=cc70 -j1

# Solution 2: Reduce optimization
make -f Makefile.gpu GPU_CC=cc70 OPTIMIZATION=-O1

# Solution 3: Add swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Issue 6: MPI-related errors

**Error message**:
```
mpif90: command not found
```

**Solutions**:

```bash
# Solution 1: Use HPC SDK's MPI
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/comm_libs/mpi/bin:$PATH

# Solution 2: Install OpenMPI
sudo apt install openmpi-bin libopenmpi-dev

# Solution 3: Specify MPI wrapper
make -f Makefile.gpu GPU_CC=cc70 MPIF90=mpifort
```

### Issue 7: "Syntax error" in GPU code

**Cause**: Old compiler version or missing flags

**Solution**:
```bash
# Check compiler version
nvfortran --version
# Should be 21.x or later

# Ensure GPU flags are set
nvfortran -acc -gpu=cc70 -cuda -c file.F90
```

### Issue 8: Build succeeds but binary doesn't run

**Check runtime requirements**:
```bash
# Check linked libraries
ldd bin/epoch2d_gpu

# Check for missing libraries
ldd bin/epoch2d_gpu | grep "not found"

# Fix missing CUDA libraries
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/lib64:$LD_LIBRARY_PATH
```

## Build Verification

### Quick Verification

```bash
# Check binary exists and is executable
ls -la bin/epoch2d_gpu

# Check library dependencies
ldd bin/epoch2d_gpu | head -10

# Quick GPU test (should print GPU info)
./bin/epoch2d_gpu 2>&1 | head -20
```

### Full Verification with GPU Tests

```bash
# Build and run GPU kernel tests
cd epoch2d/src/gpu

# Compile modules
for f in gpu_config particle_data_gpu fields_gpu particles_gpu; do
    echo "Compiling $f..."
    nvfortran -c -O3 -r8 -acc -gpu=cc70 -cuda -DGPU_ENABLED $f.F90
done

# Compile test program
nvfortran -O3 -r8 -acc -gpu=cc70 -cuda -DGPU_ENABLED \
    -o test_gpu_2d \
    gpu_config.o particle_data_gpu.o fields_gpu.o particles_gpu.o \
    test_gpu_2d.F90

# Run test
./test_gpu_2d

# Expected output:
# ============================================
#    EPOCH2D GPU Integration Test
# ============================================
# GPU Acceleration: ENABLED
# Device: Tesla V100-PCIE-32GB
# ...
# ============================================
#    EPOCH2D GPU Test PASSED!
# ============================================
```

## Build Times Reference

| Dimension | First Build | Incremental | Clean Build |
|-----------|-------------|-------------|-------------|
| epoch1d | ~2 min | ~30 sec | ~2 min |
| epoch2d | ~5 min | ~1 min | ~5 min |
| epoch3d | ~8 min | ~2 min | ~8 min |

*Tested on 8-core CPU with nvfortran 24.11*

## Next Steps

After successful build:
1. [Run GPU simulations]({{< relref "running.md" >}})
2. [Try the tutorial]({{< relref "tutorial.md" >}})
3. [Optimize performance]({{< relref "performance.md" >}})

