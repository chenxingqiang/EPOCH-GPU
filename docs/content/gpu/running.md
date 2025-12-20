---
title: Running GPU Simulations
linktitle: Running
toc: true
type: docs
date: "2025-12-19T00:00:00Z"
draft: false
menu:
  gpu:
    parent: GPU Guide
    weight: 4
---

## Basic Usage

EPOCH-GPU uses the same input deck format as standard EPOCH:

```bash
# Create working directory
mkdir my_simulation
cd my_simulation

# Copy input deck
cp /path/to/input.deck .

# Run with GPU
echo . | mpirun -n 1 /path/to/epoch2d_gpu
```

## Input Deck Compatibility

GPU-accelerated EPOCH is fully compatible with existing input decks:

```fortran
begin:control
  nx = 1024
  ny = 1024
  
  x_min = 0
  x_max = 10.0e-6
  y_min = 0
  y_max = 10.0e-6
  
  t_end = 100.0e-15
end:control

begin:species
  name = electron
  charge = -1.0
  mass = 1.0
  nparticles = 1000000
end:species
```

## GPU-Specific Environment Variables

```bash
# Select specific GPU (for multi-GPU systems)
export CUDA_VISIBLE_DEVICES=0

# Enable GPU debugging
export NVCOMPILER_ACC_DEBUG=1

# Monitor GPU memory
export NVCOMPILER_ACC_NOTIFY=1
```

## Running GPU Tests

### Fortran Integration Tests

```bash
# Test 1D kernels
cd epoch1d/src/gpu && ./test_gpu_1d

# Test 2D kernels
cd epoch2d/src/gpu && ./test_gpu_2d

# Test 3D kernels
cd epoch3d/src/gpu && ./test_gpu_3d
```

### Python Test Framework

```bash
# Run all GPU tests
python run_gpu_tests.py

# Run specific dimension
python run_gpu_tests.py 2d

# Verbose output
python run_gpu_tests.py -v
```

## Monitoring GPU Usage

During simulation, monitor GPU with:

```bash
# Real-time GPU stats
watch -n 1 nvidia-smi

# Detailed profiling
nsys profile ./epoch2d_gpu
```

## Example Output

```
============================================
   EPOCH2D GPU Integration Test
============================================
GPU Acceleration: ENABLED (OpenACC)
Device: Tesla V100-SXM2-32GB
Memory: 32510 MB

Test Configuration:
  Grid: 256 x 256
  Particles: 1000000

Test 1: 2D Field Update GPU
  100 iterations: 3.5 ms
  Per iteration: 0.035 ms
  Cells/sec: 1.87E+09

Test 2: 2D Particle Push GPU
  100 iterations: 19.0 ms
  Per iteration: 0.19 ms
  Particles/sec: 5.26E+09

============================================
   EPOCH2D GPU Test PASSED!
============================================
```

## Performance Tips

1. **Maximize GPU Utilization**: Use large particle counts (>100K)
2. **Batch Operations**: Let particles accumulate before processing
3. **Memory Management**: Monitor GPU memory with `nvidia-smi`
4. **Grid Size**: Powers of 2 often perform best

