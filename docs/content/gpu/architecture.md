---
title: GPU Architecture
linktitle: Architecture
toc: true
type: docs
date: "2025-12-19T00:00:00Z"
draft: false
menu:
  gpu:
    parent: GPU Guide
    weight: 5
---

## Directory Structure

```
epoch{1,2,3}d/src/gpu/
├── gpu_config.F90          # GPU device management & initialization
├── particle_data_gpu.F90   # SoA particle structures (dimension-specific)
├── fields_gpu.F90          # FDTD field solver kernels
├── particles_gpu.F90       # Particle pusher, interpolation, deposition
├── particle_converter.F90  # AoS ↔ SoA conversion utilities
├── epoch_gpu_interface.F90 # High-level integration layer
├── test_gpu_{1,2,3}d.F90   # GPU kernel integration tests
└── test_converter.F90      # Particle converter unit tests
```

**Note**: All three dimensions (1D, 2D, 3D) now have complete GPU integration with unified interfaces.

## Module Descriptions

### gpu_config.F90

Manages GPU device initialization and configuration:

```fortran
MODULE gpu_config
  LOGICAL, PARAMETER :: GPU_ENABLED = .TRUE.
  INTEGER :: gpu_device_id = 0
  
CONTAINS
  SUBROUTINE gpu_init(device_id)
  SUBROUTINE gpu_finalize()
  SUBROUTINE gpu_print_device_info()
END MODULE
```

### particle_data_gpu.F90

Defines Structure of Arrays (SoA) for GPU-optimized particle storage:

```fortran
TYPE :: particle_soa
  REAL(num), ALLOCATABLE :: pos_x(:)
  REAL(num), ALLOCATABLE :: pos_y(:)  ! 2D/3D only
  REAL(num), ALLOCATABLE :: pos_z(:)  ! 3D only
  REAL(num), ALLOCATABLE :: px(:), py(:), pz(:)
  REAL(num), ALLOCATABLE :: weight(:)
  INTEGER(i8) :: count, capacity
END TYPE
```

**Why SoA?** GPU memory access is optimized for coalesced reads. SoA ensures 
adjacent threads access adjacent memory locations.

### fields_gpu.F90

FDTD electromagnetic field solver with OpenACC:

```fortran
SUBROUTINE update_e_field_gpu(ex, ey, ez, bx, by, bz, jx, jy, jz, ...)
  !$ACC PARALLEL LOOP COLLAPSE(2) PRESENT(ex, ey, ez, bx, by, bz, jx, jy, jz)
  DO iy = 1, ny
    DO ix = 1, nx
      ex(ix,iy) = ex(ix,iy) + cny*(bz(ix,iy) - bz(ix,iy-1)) - fac*jx(ix,iy)
      ! ... E and B field updates
    END DO
  END DO
  !$ACC END PARALLEL LOOP
END SUBROUTINE
```

### particles_gpu.F90

Contains three key GPU kernels:

1. **Boris Pusher** - Relativistic particle momentum update
2. **Field Interpolation** - Gather fields to particle positions
3. **Current Deposition** - Scatter particle currents to grid (with atomics)

```fortran
SUBROUTINE push_particles_gpu(n_part, pos_x, pos_y, px, py, pz, ...)
  !$ACC PARALLEL LOOP PRESENT(pos_x, pos_y, px, py, pz, ex_p, ey_p, ...)
  DO ipart = 1, n_part
    ! Boris rotation algorithm
    ! ...
  END DO
  !$ACC END PARALLEL LOOP
END SUBROUTINE

SUBROUTINE deposit_current_gpu(...)
  !$ACC ATOMIC
  jx(ix, iy) = jx(ix, iy) + j_contrib
  !$ACC END ATOMIC
END SUBROUTINE
```

## Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                    Main Time Loop                        │
├─────────────────────────────────────────────────────────┤
│  1. update_b_field_gpu()  ─────────────────────────────►│
│     └── FDTD half-step B update                         │
│                                                          │
│  2. interpolate_fields_to_particles_gpu()  ────────────►│
│     └── Gather E, B to particle positions               │
│                                                          │
│  3. push_particles_gpu()  ─────────────────────────────►│
│     └── Boris algorithm: update p, then x               │
│                                                          │
│  4. deposit_current_gpu()  ────────────────────────────►│
│     └── Scatter J to grid (atomic operations)           │
│                                                          │
│  5. update_e_field_gpu()  ─────────────────────────────►│
│     └── FDTD E update with current                      │
│                                                          │
│  6. update_b_field_gpu()  ─────────────────────────────►│
│     └── FDTD half-step B update                         │
└─────────────────────────────────────────────────────────┘
```

## Memory Management

Data residency is managed with OpenACC directives:

```fortran
! Allocate on device
!$ACC ENTER DATA CREATE(ex, ey, ez, bx, by, bz, jx, jy, jz)

! Copy to device
!$ACC UPDATE DEVICE(ex, ey, ez, bx, by, bz)

! Compute on device
!$ACC PARALLEL LOOP PRESENT(ex, ey, ez, ...)

! Copy results back
!$ACC UPDATE HOST(ex, ey, ez, bx, by, bz)

! Free device memory
!$ACC EXIT DATA DELETE(ex, ey, ez, bx, by, bz, jx, jy, jz)
```

## Algorithm Design Choices

### Current Deposition Method

EPOCH-GPU uses **Esirkepov charge-conserving current deposition** for all dimensions:

```fortran
! Esirkepov method: J from particle displacement (charge-conserving)
! 1. Compute shape functions at old and new positions
! 2. Calculate shape function difference: dw = w_new - w_old
! 3. Deposit Jx via cumulative sum (ensures div(J) = -d(rho)/dt)

! Shape functions at old position
wold(0) = 1 - frac_old
wold(1) = frac_old

! Shape functions at new position (shifted by dcell)
wnew(dcell) = 1 - frac_new
wnew(dcell+1) = frac_new

! Cumulative sum for Jx (Esirkepov)
jx_accum = 0
DO ix = -1, 2
  jx_accum = jx_accum - qw_idt * dw(ix)
  !$ACC ATOMIC UPDATE
  jx(ix_base + ix) = jx(ix_base + ix) + jx_accum
END DO
```

**Why Esirkepov method?**

1. **Charge conservation**: Ensures ∇·J = -∂ρ/∂t exactly (no spurious charge accumulation)
2. **Physical accuracy**: Critical for long-time simulations where charge errors accumulate
3. **Consistency with CPU**: Matches EPOCH's original charge-conserving implementation

**Dimension-specific configurations:**

| Dimension | Indexing | Shape Functions | Normalization |
|-----------|----------|-----------------|---------------|
| **1D** | Cell-centered | Linear (sum=1) | fac = 0.5 |
| **2D** | Edge-centered | Linear (sum=1) | No fac |
| **3D** | Edge-centered | Linear (sum=1) | No fac |

### Unified Interface Design

All three dimensions (1D, 2D, 3D) share the same interface pattern:

| Function | 1D | 2D | 3D |
|----------|----|----|-----|
| `push_particles_gpu` | ✓ | ✓ | ✓ |
| `interpolate_fields_to_particles_gpu` | ✓ | ✓ | ✓ |
| `deposit_current_gpu` | ✓ | ✓ | ✓ |

The dimension-specific implementations are provided via Fortran generic interfaces:

```fortran
! In particles_gpu.F90
INTERFACE push_particles_gpu
  MODULE PROCEDURE push_particles_gpu_1d  ! or _2d, _3d
END INTERFACE
```

### Field Solver Status

Currently, field updates (E and B) remain on CPU:

- **Reason**: Complex boundary conditions and MPI communication patterns
- **Plan**: GPU field solver is functional but disabled in main integration pending further testing

