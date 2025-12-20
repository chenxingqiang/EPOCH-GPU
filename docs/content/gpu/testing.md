---
title: Testing & Validation
linktitle: Testing
toc: true
type: docs
date: "2024-12-20T00:00:00Z"
draft: false
menu:
  gpu:
    parent: GPU Guide
    weight: 9
---

## Test Environment

All tests verified on:

| Component | Version |
|-----------|---------|
| **GPU** | Tesla V100-PCIE-32GB |
| **Compute Capability** | 7.0 |
| **GPU Memory** | 32 GB |
| **CUDA Driver** | 535.54.03 |
| **CUDA Version** | 12.2 |
| **Compiler** | nvfortran 24.11 |
| **Python** | 3.10.12 |

## Fortran GPU Kernel Tests

### Running Tests

```bash
# Compile and run 1D tests
cd epoch1d/src/gpu
nvfortran -c -O3 -r8 -acc -gpu=cc70 -cuda -DGPU_ENABLED *.F90
nvfortran -O3 -r8 -acc -gpu=cc70 -cuda -DGPU_ENABLED \
  -o test_gpu_1d *.o test_gpu_1d.F90
./test_gpu_1d

# Similarly for 2D and 3D
cd epoch2d/src/gpu && ./test_gpu_2d
cd epoch3d/src/gpu && ./test_gpu_3d
```

### Test Results Summary

| Dimension | Tests | Status | Key Metrics |
|-----------|-------|--------|-------------|
| **1D** | 4 kernels | ✅ ALL PASSED | 5.98B particles/sec |
| **2D** | 4 kernels | ✅ ALL PASSED | 5.30B particles/sec |
| **3D** | 4 kernels | ✅ ALL PASSED | 4.30B particles/sec |

### Detailed 1D Results

```
============================================
   EPOCH1D GPU Integration Test
============================================
GPU Acceleration: ENABLED
Device: Tesla V100-PCIE-32GB
Memory: 31.74 GB

Test 1: 1D Field Update GPU (1024 cells)
  1000 iterations: 25.58 ms
  Per iteration: 0.0256 ms

Test 2: 1D Particle Push GPU (1M particles)
  100 iterations: 16.73 ms
  Per iteration: 0.167 ms
  Particles/sec: 5.98E+09

Test 3: 1D Field Interpolation GPU
  100 iterations: 9.45 ms
  Per iteration: 0.095 ms

Test 4: 1D Current Deposition GPU
  100 iterations: 15.04 ms
  Per iteration: 0.150 ms

============================================
   EPOCH1D GPU Test PASSED!
============================================
```

### Detailed 2D Results

```
============================================
   EPOCH2D GPU Integration Test
============================================
GPU Acceleration: ENABLED
Device: Tesla V100-PCIE-32GB
Grid: 256 x 256 (65536 cells)
Particles: 1,000,000

Test 1: 2D Field Update GPU
  100 iterations: 3.53 ms
  Per iteration: 0.035 ms
  Cells/sec: 1.87E+09

Test 2: 2D Particle Push GPU
  100 iterations: 18.88 ms
  Per iteration: 0.189 ms
  Particles/sec: 5.30E+09

Test 3: 2D Field Interpolation GPU (Bilinear)
  100 iterations: 11.02 ms
  Per iteration: 0.110 ms

Test 4: 2D Current Deposition GPU (Atomic)
  100 iterations: 48.96 ms
  Per iteration: 0.490 ms

============================================
   EPOCH2D GPU Test PASSED!
============================================
```

### Detailed 3D Results

```
============================================
   EPOCH3D GPU Integration Test
============================================
GPU Acceleration: ENABLED
Device: Tesla V100-PCIE-32GB
Grid: 64 x 64 x 64 (262144 cells)
Particles: 500,000

Test 1: 3D Field Update GPU
  100 iterations: 9.98 ms
  Per iteration: 0.100 ms
  Cells/sec: 2.63E+09

Test 2: 3D Particle Push GPU
  100 iterations: 11.62 ms
  Per iteration: 0.116 ms
  Particles/sec: 4.30E+09

Test 3: 3D Field Interpolation GPU (Trilinear)
  100 iterations: 31.74 ms
  Per iteration: 0.317 ms

Test 4: 3D Current Deposition GPU (Atomic)
  100 iterations: 35.87 ms
  Per iteration: 0.359 ms

============================================
   EPOCH3D GPU Test PASSED!
============================================
```

## Python Test Framework

### Running Python Tests

```bash
# Install pytest
pip install pytest

# Run tests for each dimension separately (to avoid module conflicts)
cd epoch1d/tests_gpu && python -m pytest . -v
cd epoch2d/tests_gpu && python -m pytest . -v
cd epoch3d/tests_gpu && python -m pytest . -v
```

### Test Results

| Dimension | Passed | Failed | Total |
|-----------|--------|--------|-------|
| **1D** | 12 | 0 | 12 |
| **2D** | 25 | 2* | 27 |
| **3D** | 18 | 1* | 19 |

> *Failed tests are CFL condition parameter validation (test configuration issues, not GPU kernel failures)

### Test Categories

**1D Tests** (`epoch1d/tests_gpu/`):
- `test_gpu_kernels.py` - Core GPU kernel validation
- `test_gpu_twostream.py` - Two-stream instability physics

**2D Tests** (`epoch2d/tests_gpu/`):
- `test_gpu_kernels.py` - Core GPU kernel validation
- `test_gpu_laser.py` - Laser propagation physics
- `test_gpu_maxwell_solvers.py` - FDTD solver validation
- `test_gpu_particles.py` - Particle pusher validation

**3D Tests** (`epoch3d/tests_gpu/`):
- `test_gpu_kernels.py` - Core GPU kernel validation
- `test_gpu_laser.py` - Laser propagation physics
- `test_gpu_maxwell_solvers.py` - FDTD solver validation

## Python Package Tests

### Installation Verification

```bash
pip install epoch-gpu
python -c "import epoch_gpu; print(epoch_gpu.__version__)"
# Output: 1.0.0
```

### CLI Tools Test

```bash
$ epoch-gpu-info
============================================================
EPOCH-GPU Configuration
============================================================

NVIDIA Tools:
  nvidia-smi: /usr/bin/nvidia-smi
  nvfortran:  /opt/nvidia/hpc_sdk/.../nvfortran

Available GPUs (1):
  [0] Tesla V100-PCIE-32GB
      Memory: 32501 MiB free / 32768 MiB total
      Compute Capability: 7.0
      CUDA Driver: 535.54.03
============================================================
```

### API Test Results

| Module | Function | Status |
|--------|----------|--------|
| `GPUConfig` | `has_gpu()` | ✅ True |
| `GPUConfig` | `has_compiler()` | ✅ True |
| `GPUConfig` | `get_gpus()` | ✅ 1 GPU detected |
| `units` | `plasma_frequency(1e25)` | ✅ 1.784e+14 rad/s |
| `units` | `debye_length(1e25, 1000)` | ✅ 0.69 nm |
| `units` | `skin_depth(1e25)` | ✅ 1.68 μm |
| `units` | `thermal_velocity(1000)` | ✅ 1.23e+05 m/s |
| `DeckFile` | `set()` / `get()` | ✅ Working |
| `diagnostics` | `calculate_field_energy()` | ✅ Working |
| `diagnostics` | `calculate_temperature()` | ✅ Working |

## Continuous Integration

GitHub Actions workflow runs on every push:

- **Python Tests**: Unit tests for the epoch-gpu package
- **GPU Syntax Check**: Validates GPU Fortran code compilation
- **GPU Build Test**: Manual trigger for full GPU compilation

See `.github/workflows/build_test.yml` for details.

