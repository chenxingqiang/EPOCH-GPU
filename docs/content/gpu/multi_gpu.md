---
title: Multi-GPU Support
linktitle: Multi-GPU
toc: true
type: docs
date: "2025-12-19T00:00:00Z"
draft: false
menu:
  gpu:
    parent: GPU Guide
    weight: 8
---

EPOCH-GPU supports multi-GPU simulations using GPU-aware MPI for efficient 
inter-GPU communication.

## Features

- **CUDA-aware MPI**: Direct GPU-to-GPU communication when available
- **Automatic GPU assignment**: Based on local MPI rank
- **Fallback mode**: Staged transfers when CUDA-aware MPI is unavailable
- **Hybrid parallelism**: MPI + GPU for large-scale simulations

## Environment Variables

```bash
# Enable/disable GPU-aware MPI
export EPOCH_GPU_AWARE_MPI=1

# Set GPUs per node
export EPOCH_GPUS_PER_NODE=4

# Limit visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## Running Multi-GPU Simulations

### Single Node, Multiple GPUs

```bash
# 4 GPUs on one node
mpirun -n 4 ./epoch2d_gpu
```

### Multiple Nodes

```bash
# 2 nodes × 4 GPUs = 8 total GPUs
mpirun -n 8 --map-by ppr:4:node ./epoch2d_gpu
```

### SLURM Example

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00

module load nvhpc cuda

srun ./epoch2d_gpu
```

## GPU-aware MPI Detection

EPOCH-GPU automatically detects CUDA-aware MPI support:

1. **OpenMPI**: Checks `OMPI_MCA_mpi_cuda_support`
2. **MVAPICH**: Checks `MV2_USE_CUDA`
3. **Manual**: Set `EPOCH_GPU_AWARE_MPI=1`

## Module Architecture

```fortran
MODULE mpi_gpu
  ! GPU-aware MPI communication
  SUBROUTINE exchange_field_x_gpu(...)  ! X boundary exchange
  SUBROUTINE exchange_field_y_gpu(...)  ! Y boundary exchange
  SUBROUTINE exchange_fields_gpu(...)   ! All fields
END MODULE

MODULE epoch_main_gpu
  ! Main loop integration
  SUBROUTINE epoch_gpu_main_init(...)   ! Initialize
  SUBROUTINE epoch_gpu_time_step()      ! One time step
  SUBROUTINE epoch_gpu_main_finalize()  ! Cleanup
END MODULE
```

## Communication Flow

```
┌─────────────────────────────────────────────────────────────┐
│                  GPU-aware MPI Exchange                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GPU 0              GPU 1              GPU 2              │
│  ┌─────┐           ┌─────┐           ┌─────┐              │
│  │Field│◄─────────►│Field│◄─────────►│Field│              │
│  │Data │  Direct   │Data │  Direct   │Data │              │
│  └─────┘  GPU-GPU  └─────┘  GPU-GPU  └─────┘              │
│                                                              │
│  When CUDA-aware MPI is available:                          │
│  - No host-device transfers needed                          │
│  - Direct RDMA between GPUs                                 │
│                                                              │
│  Fallback mode:                                              │
│  - GPU → Host → MPI → Host → GPU                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Performance Considerations

### GPU Assignment

Each MPI rank is assigned to a GPU based on local rank:

```
GPU ID = local_rank % gpus_per_node
```

### Communication Overlap

Future optimization: overlap computation with communication:

```fortran
! Async field update (interior)
!$ACC PARALLEL LOOP ASYNC(1)
  ! Update interior cells
!$ACC END PARALLEL LOOP

! Start boundary exchange
CALL exchange_fields_gpu(...) ! Uses stream 2

! Wait for both
!$ACC WAIT
```

## Testing Multi-GPU

```bash
# Compile multi-GPU test
cd epoch2d/src/gpu
nvfortran -O3 -r8 -acc -gpu=cc70 -cuda -DGPU_ENABLED \
  -o test_multi_gpu \
  gpu_config.o fields_gpu.o particles_gpu.o mpi_gpu.o epoch_main_gpu.o \
  test_multi_gpu.F90 -lmpi

# Run with 4 GPUs
mpirun -n 4 ./test_multi_gpu
```

## Troubleshooting

### GPUs not detected

```bash
# Check GPU visibility
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
```

### Wrong GPU assignment

```bash
# Force specific GPUs
export CUDA_VISIBLE_DEVICES=0,1
mpirun -n 2 ./epoch2d_gpu
```

### CUDA-aware MPI issues

```bash
# Disable GPU-aware MPI
export EPOCH_GPU_AWARE_MPI=0
```

