---
title: Performance Tuning
linktitle: Performance
toc: true
type: docs
date: "2024-12-19T00:00:00Z"
draft: false
menu:
  gpu:
    parent: GPU Guide
    weight: 6
---

# GPU Performance Tuning

## Benchmark Results (Tesla V100-32GB)

### 1D Performance

| Kernel | Configuration | Time/Iteration | Throughput |
|--------|---------------|----------------|------------|
| Field Update | 1024 cells | 0.027 ms | - |
| Particle Push | 1M particles | 0.17 ms | **5.9B particles/sec** |
| Field Interpolation | 1M particles | 0.097 ms | - |
| Current Deposition | 1M particles | 0.15 ms | - |

### 2D Performance

| Kernel | Configuration | Time/Iteration | Throughput |
|--------|---------------|----------------|------------|
| Field Update | 256×256 | 0.035 ms | 1.9B cells/sec |
| Particle Push | 1M particles | 0.19 ms | **5.3B particles/sec** |
| Field Interpolation | 1M particles | 0.11 ms | - |
| Current Deposition | 1M particles | 0.49 ms | - |

### 3D Performance

| Kernel | Configuration | Time/Iteration | Throughput |
|--------|---------------|----------------|------------|
| Field Update | 64³ | 0.096 ms | 2.7B cells/sec |
| Particle Push | 500K particles | 0.11 ms | **4.4B particles/sec** |
| Field Interpolation | 500K particles | 0.35 ms | - |
| Current Deposition | 500K particles | 0.38 ms | - |

## Optimization Strategies

### 1. Particle Count

GPUs excel with large particle counts:

```
Particles    Time/Push    Efficiency
10,000       0.05 ms      ~5% GPU utilization
100,000      0.08 ms      ~25% GPU utilization
1,000,000    0.19 ms      ~90% GPU utilization
10,000,000   1.8 ms       ~95% GPU utilization
```

**Recommendation**: Use at least 100K particles per species.

### 2. Grid Size

Optimal grid sizes are powers of 2:

```fortran
! Good: powers of 2
nx = 256
ny = 256

! Also good: multiples of 32
nx = 320
ny = 320
```

### 3. Memory Bandwidth

V100 theoretical bandwidth: 900 GB/s

Field solver memory analysis:
- 6 reads (E, B fields) + 3 reads (J) + 6 writes = 15 arrays
- Per cell: 15 × 8 bytes = 120 bytes
- For 256² grid: ~7.9 MB per update
- Achieved: ~200-400 GB/s (22-44% efficiency)

### 4. Occupancy

Check GPU occupancy with:

```bash
nvprof --metrics achieved_occupancy ./epoch2d_gpu
```

Target: >50% occupancy for compute-bound kernels.

## Profiling Tools

### NVIDIA Nsight Systems

```bash
nsys profile -o report ./epoch2d_gpu
nsys-ui report.qdrep
```

### NVIDIA Nsight Compute

```bash
ncu --set full -o kernel_analysis ./epoch2d_gpu
ncu-ui kernel_analysis.ncu-rep
```

### Environment Variables

```bash
# Show kernel launch info
export NVCOMPILER_ACC_NOTIFY=1

# Detailed debugging
export NVCOMPILER_ACC_DEBUG=1

# Memory allocation tracking
export NVCOMPILER_ACC_TIME=1
```

## Common Bottlenecks

### 1. Current Deposition (Atomics)

Current deposition uses atomic operations which can serialize:

```fortran
!$ACC ATOMIC
jx(ix, iy) = jx(ix, iy) + j_contrib
```

**Mitigation**: Use histogram binning or thread-local accumulation.

### 2. Data Transfer

Minimize host-device transfers:

```fortran
! Bad: transfer every step
!$ACC UPDATE DEVICE(particles)
CALL push_particles_gpu(...)
!$ACC UPDATE HOST(particles)

! Good: keep data on device
!$ACC DATA COPYIN(particles) COPYOUT(particles)
DO step = 1, nsteps
  CALL push_particles_gpu(...)
END DO
!$ACC END DATA
```

### 3. Memory Allocation

Pre-allocate with capacity buffer:

```fortran
soa%capacity = INT(n_particles * 1.2_num, i8)  ! 20% buffer
```

## Scaling Analysis

Expected scaling for larger problems:

| Grid Size | Field Update | Memory |
|-----------|--------------|--------|
| 256² | 0.035 ms | ~5 MB |
| 512² | 0.12 ms | ~19 MB |
| 1024² | 0.45 ms | ~75 MB |
| 2048² | 1.8 ms | ~300 MB |

| Particles | Push Time | Memory |
|-----------|-----------|--------|
| 1M | 0.19 ms | ~56 MB |
| 10M | 1.8 ms | ~560 MB |
| 100M | 18 ms | ~5.6 GB |

