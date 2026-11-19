---
title: Performance Tuning
linktitle: Performance
toc: true
type: docs
date: "2026-10-22T00:00:00Z"
draft: false
menu:
  gpu:
    parent: GPU Guide
    weight: 6
---

## Benchmark Results (Tesla V100-PCIE-32GB)

> **Verified**: December 2025 | CUDA 12.2 | nvfortran 24.11

### 1D Performance

| Kernel | Configuration | Time/Iteration | Throughput |
|--------|---------------|----------------|------------|
| Field Update | 1024 cells × 1000 iter | 0.0256 ms | 39.1M cells/sec |
| Particle Push | 1M particles × 100 iter | 0.167 ms | **5.98B particles/sec** |
| Field Interpolation | 1M particles × 100 iter | 0.095 ms | - |
| Current Deposition | 1M particles × 100 iter | 0.150 ms | - |

### 2D Performance

| Kernel | Configuration | Time/Iteration | Throughput |
|--------|---------------|----------------|------------|
| Field Update | 256×256 × 100 iter | 0.035 ms | 1.87B cells/sec |
| Particle Push | 1M particles × 100 iter | 0.189 ms | **5.30B particles/sec** |
| Field Interpolation | 1M particles × 100 iter | 0.110 ms | - |
| Current Deposition | 1M particles × 100 iter | 0.490 ms | - |

### 3D Performance

| Kernel | Configuration | Time/Iteration | Throughput |
|--------|---------------|----------------|------------|
| Field Update | 64×64×64 × 100 iter | 0.100 ms | 2.63B cells/sec |
| Particle Push | 500K particles × 100 iter | 0.116 ms | **4.30B particles/sec** |
| Field Interpolation | 500K particles × 100 iter | 0.317 ms | - |
| Current Deposition | 500K particles × 100 iter | 0.359 ms | - |

These rows are kernel microbenchmarks. The arrays already reside on the device, and the timer covers one kernel, not a full EPOCH time step. A full step also updates the fields, finishes the current, migrates particles, and writes output.

## Comparison with the CPU code

This repository does not contain a measured wall-clock ratio of a full GPU run against a full CPU run. The Python check `test_speedup_over_cpu` divides a GPU cell rate by a hardcoded `1e7` cells/s. That constant is not a timing of EPOCH on a CPU, so it is not a speedup.

What the V100 timings do show is the particle kernels once the data is on the device:

| 2D kernel, 1M particles | Time per call | Particle rate |
|-------------------------|---------------|---------------|
| Field interpolation | 0.110 ms | 9.1 billion/s |
| Boris push | 0.189 ms | 5.30 billion/s |
| Esirkepov current deposition | 0.490 ms | 2.04 billion/s |
| Three kernels together | 0.789 ms | 1.27 billion/s |

Deposition is the slowest of the three. It uses `!$ACC ATOMIC UPDATE` on FP64 current arrays (`epoch2d/src/gpu/particles_gpu.F90`), so its rate is set by atomic throughput rather than by streaming bandwidth.

On the 1D, 2D, and 3D GPU builds, a second-order Yee step with no CPML keeps the particle positions on the device. Old positions for Esirkepov are copied by a device kernel, and the gathered field arrays are allocated once. The Yee stencil itself runs on the device. Boundary conditions, lasers, and `current_finish` still run on the host, so each half-step copies the electric or magnetic field across PCIe, and the current is copied back after deposition and again after `current_finish`. Particle output copies the particles only on a dump step.

That host work is still in the timed step. Moving-window runs, injectors, load balancing, collisions, ionisation, and migration refresh the device arrays from the host lists, because those operators edit the host particle list. A static periodic deck does not take that path.

`gpu_init_all` selects `local_rank mod device_count`. One process per GPU can also be pinned with `CUDA_VISIBLE_DEVICES`.

### Measured on an RTX 5090

One MPI rank, `nvfortran` 25.9, `-O3 -gpu=cc120`, double precision. The deck is a periodic electron-ion plasma (`n = 1e22 m^-3`, 100 eV, 16 electron and 4 ion particles per cell). The CPU binary is the same source built without the GPU kernels. Times below are the gap from iteration 10 to iteration 20, so they leave out deck setup and the final copy off the device.

| Grid | Particles | GPU | CPU, one rank |
|------|-----------|-----|----------------|
| 256×256 | 1.31 million | 3 ms/step | 161 ms/step |
| 1024×1024 | 21.0 million | 32 ms/step | not run |

On the 256×256 deck the final `Ex` is finite on both binaries and of the same order (a few MV/m). The cell values are not identical: the GPU deposit uses the linear Esirkepov weights plus the EPOCH triangle normalization, and the two runs load different random particles. The 1D, 2D, and 3D particle kernels use SI `c = 299792458`. The same resident Yee path was compiled with `-gpu=cc120` and run on this RTX 5090: a 1024-cell 1D plasma (40,960 particles) finished 30 steps in 0.09 s of core time, and a 48³ plasma (1.11 million particles) took about 10 ms/step between iteration 10 and iteration 20. Those two runs were not compared with a CPU binary.

1D has a separate limit: the GPU result differs from the CPU result by about a 1.7× amplitude ratio. Use 2D or 3D when comparing physics. Those two are the dimensions marked as physically checked.

## GeForce RTX 3090, 4090, and 5090

The GPU modules fix the working precision at `REAL64`, and `Makefile.gpu` always passes `-r8`. `constants.F90` sets `num = KIND(1.d0)` with no `SINGLE` branch. `make DEFINE="-DSINGLE"` does not build a single-precision GPU binary. Consumer cards therefore run this code at full double precision.

GeForce FP64 units are there so FP64 programs run correctly. NVIDIA states the rate as 1/64 of the FP32 rate for GA102 ([Ampere GA102 whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)), AD102 ([Ada whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)), and GB202 ([Blackwell GeForce whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)). The same Blackwell table lists the non-tensor peaks and the DRAM bandwidth used below. The V100 PCIe numbers are from the [V100 datasheet](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet.pdf) (7 TFLOPS FP64, 900 GB/s). The H100 SXM numbers are from the [H100 product page](https://www.nvidia.com/en-us/data-center/h100/) (34 TFLOPS FP64, 3.35 TB/s, 80 GB). Tensor-core FP64 (67 TFLOPS on H100 SXM) does not apply to the Boris push or to Esirkepov deposition.

| GPU | FP64 peak | Memory | DRAM bandwidth | FP64 per byte of DRAM |
|-----|-----------|--------|----------------|------------------------|
| V100 PCIe 32 GB | 7 TFLOPS | 32 GB HBM2 | 900 GB/s | 7.8 FLOP/byte |
| RTX 3090 | 35.6 / 64 = 0.556 TFLOPS | 24 GB GDDR6X | 936 GB/s | 0.59 FLOP/byte |
| RTX 4090 | 82.6 / 64 = 1.29 TFLOPS | 24 GB GDDR6X | 1008 GB/s | 1.28 FLOP/byte |
| RTX 5090 | 104.8 / 64 = 1.64 TFLOPS | 32 GB GDDR7 | 1792 GB/s | 0.91 FLOP/byte |
| H100 SXM | 34 TFLOPS | 80 GB HBM3 | 3350 GB/s | 10.1 FLOP/byte |

The 2D push reads 11 FP64 values and writes 5 per particle (positions, momenta, and the six field components already gathered to the particle): 128 bytes. The loop body is about 90 add/multiply operations plus 3 square roots and 4 divisions. At roughly 100 arithmetic operations per particle the arithmetic intensity is about 0.8 FLOP/byte.

On the V100 that intensity sits far below 7.8 FLOP/byte, and the measured 5.30 billion particles/s corresponds to about 680 GB/s, most of the 900 GB/s HBM2 peak. The push is bandwidth-bound on that card. Scaling that measured rate by published bandwidth gives an upper bound, not a benchmark:

| GPU | Bandwidth upper bound on the 2D push | FP64 upper bound at 100 FLOP/particle |
|-----|----------------------------------------|----------------------------------------|
| V100 PCIe | 5.30 billion/s, measured | 70 billion/s |
| RTX 3090 | 5.5 billion/s | 5.6 billion/s |
| RTX 4090 | 5.9 billion/s | 13 billion/s |
| RTX 5090 | 10.6 billion/s | 16 billion/s |
| H100 SXM | 19.7 billion/s | 340 billion/s |

Reading the two columns together:

- An RTX 4090 push should land near a V100. Its DRAM bandwidth is close to 900 GB/s, and 1.29 TFLOPS of FP64 is enough for this kernel.
- An RTX 3090 has the same DRAM bandwidth class and about 0.56 TFLOPS of FP64. The push sits on that FP64 ceiling. Expect the same order as a V100 for streaming updates, with less spare FP64 for deposition.
- An RTX 5090 has about twice the V100's DRAM bandwidth (1792 / 900). If the kernel stays bandwidth-bound, the push upper bound is about 11 billion particles/s. The FP64 ceiling at 100 FLOP/particle is 16 billion/s, so the two limits are close; square roots, divisions, and atomics can pull a real run down onto the FP64 units.
- An H100 SXM stays bandwidth-bound. Its DRAM bound is about 20 billion particles/s, roughly twice the 5090 bound, in a single 80 GB allocation. The FP64 unit rate is 34 / 1.64 ≈ 21 times the 5090. H100 is the faster double-precision device for this code.

Claims that an RTX 5090 beats an H100 come from graphics and from low-precision tensor math (the Blackwell table lists 1676 / 3352 FP4 tensor TFLOPS with sparsity). Those units are idle in this solver.

Current deposition is the kernel to watch on GeForce. It is already 2.6× slower than the push on the V100, and each deposit issues many FP64 atomics. Atomic throughput tracks the FP64 pipes, which are the 1/64 resource. A 3090 or 5090 can be deposition-limited on a problem that was bandwidth-limited on a V100.

The field kernels are the other way round: a Yee update is a handful of FLOPs per cell and is bandwidth-bound everywhere. The published 256×256 field timing is a few megabytes and fits in the RTX 4090 L2 (72 MB) and the RTX 5090 L2 (96 MB) reported in those whitepapers. Rank cards on a grid large enough to miss L2.

### Which card is worth using

No RTX 3090, 4090, or 5090 timing has been taken in this repository. The build flags that match the hardware are `GPU_CC=cc86` (3090), `GPU_CC=cc89` (4090), and `GPU_CC=cc120` (5090, compute capability 12.0 on the [RTX 5090 page](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)). A 5090 build needs an `nvfortran` that accepts `cc120`, and that combination is unverified here.

For a workstation whose alternative is a desktop CPU, any of the three is a useful double-precision accelerator for 2D problems that fit in one card: 24 GB on the 3090 and 4090, 32 GB on the 5090. The memory formula in the prerequisites guide is the one to use. The 4090 is the card whose build is already documented and whose FP64 rate clears the push. The 5090 is the one whose 32 GB and 1792 GB/s raise the ceiling, and whose GDDR7 DRAM has on-die single-bit ECC always enabled ([Blackwell whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf), v1.1 note). That ECC covers the DRAM die. It is a narrower guarantee than HBM ECC on a V100 or H100.

Card power from the same Blackwell comparison table is 350 W (3090), 450 W (4090), and 575 W (5090).

## Several RTX cards

Four 32 GB cards sum to 128 GB of aggregate memory. Four 24 GB cards (3090 or 4090) sum to 96 GB. Either total is a set of separate address spaces. EPOCH already decomposes the domain with MPI, so each rank must fit its own patch, its particles, and the ghost cells in one GPU. A 2D patch splits cleanly that way; a single 3D array larger than one card does not become one allocation.

`gpu_init_all` maps the local MPI rank onto `local_rank mod device_count`. Halo exchange still uses the host MPI path: `mpi_gpu.F90` is not linked into `epoch2d_gpu`. One rank per card is the supported layout. `CUDA_VISIBLE_DEVICES` can pin a process to a single device; the local-rank map covers the case where every rank sees every card.
The production binary does not do this yet:

- `gpu_init_all` calls `gpu_init(0)`. Every MPI rank selects device 0.
- `epoch2d/src/gpu/mpi_gpu.F90` is not in `GPU_SRCFILES` in `epoch2d/Makefile.gpu`. The multi-GPU helpers are compiled only by the standalone GPU makefile under `epoch2d/src/gpu/`.

Binding ranks with `CUDA_VISIBLE_DEVICES` is the available workaround, one visible GPU per process. Halo exchange then uses the normal host MPI path, because the GPU-aware exchange is not linked.

GeForce interconnects follow the same split. The Ampere whitepaper describes 2-way NVLink on the RTX 3090 and states that 3-way and 4-way SLI are unsupported. The [RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) and [RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) pages list NVLink as not available. A four-card node communicates over PCIe. For a large 2D problem the halo is a thin strip, so PCIe can carry it once device binding works. Four 5090s also draw 4 × 575 W from the GPUs alone.

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

