---
title: GPU Particle Simulation Tutorial
linktitle: Tutorial
toc: true
type: docs
date: "2024-12-20T00:00:00Z"
draft: false
menu:
  gpu:
    parent: GPU Guide
    weight: 10
---

## Introduction

This tutorial walks you through creating and running a GPU-accelerated particle
simulation using EPOCH-GPU. We'll simulate a classic plasma physics problem: the
two-stream instability.

## Prerequisites

Before starting, ensure you have:

- EPOCH-GPU compiled with GPU support
- Access to an NVIDIA GPU (V100, A100, or H100 recommended)
- Basic understanding of PIC simulations

```bash
# Verify GPU is available
nvidia-smi

# Verify EPOCH-GPU is compiled
ls epoch2d/bin/epoch2d_gpu
```

## Tutorial 1: Two-Stream Instability (1D)

The two-stream instability occurs when two counter-streaming electron beams
interact, leading to exponential growth of electrostatic waves.

### Step 1: Create the Input Deck

Create a file `twostream.deck`:

```fortran
begin:control
  # Grid configuration
  nx = 1024
  
  # Domain size (in meters)
  x_min = 0.0
  x_max = 1.0e-4
  
  # Simulation time
  t_end = 5.0e-12
  
  # Output frequency
  nsteps = -1
  dt_multiplier = 0.95
end:control

begin:boundaries
  bc_x_min = periodic
  bc_x_max = periodic
end:boundaries

begin:constant
  # Physical constants
  n0 = 1.0e20          # Plasma density (m^-3)
  v_beam = 1.0e7       # Beam velocity (m/s)
  T_e = 100.0          # Temperature (eV)
  
  # Derived quantities
  omega_pe = sqrt(n0 * qe^2 / (epsilon0 * me))
  lambda_d = sqrt(epsilon0 * T_e * ev / (n0 * qe^2))
end:constant

begin:species
  name = beam_right
  charge = -1.0
  mass = 1.0
  nparticles_per_cell = 100
  
  number_density = n0 / 2.0
  temp_ev = T_e
  drift_x = v_beam
end:species

begin:species
  name = beam_left
  charge = -1.0
  mass = 1.0
  nparticles_per_cell = 100
  
  number_density = n0 / 2.0
  temp_ev = T_e
  drift_x = -v_beam
end:species

begin:species
  name = ions
  charge = 1.0
  mass = 1836.0
  nparticles_per_cell = 50
  
  number_density = n0
  temp_ev = T_e / 100.0
end:species

begin:output
  dt_snapshot = 1.0e-13
  
  # Field outputs
  ex = always
  ey = always
  
  # Particle outputs
  px = always
  py = always
  weight = always
  
  # Derived quantities
  number_density = always + species
  ekbar = always + species
end:output
```

### Step 2: Run the Simulation

```bash
# Create working directory
mkdir twostream_gpu
cd twostream_gpu

# Copy the deck file
cp ../twostream.deck input.deck

# Run with GPU acceleration
echo . | mpirun -n 1 ../epoch1d/bin/epoch1d_gpu
```

### Step 3: Monitor GPU Usage

In another terminal:

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Or use NVIDIA's profiler
nsys profile -o twostream_profile mpirun -n 1 ../epoch1d/bin/epoch1d_gpu
```

### Step 4: Analyze Results

Using the epoch-gpu Python package:

```python
import numpy as np
import matplotlib.pyplot as plt
from epoch_gpu.utils import units

# Load SDF data (using sdf library)
import sdf

# Load field data
data = sdf.read("0010.sdf")
ex = data.Electric_Field_Ex.data
x = data.Electric_Field_Ex.grid.data[0]

# Plot electric field
plt.figure(figsize=(10, 6))
plt.plot(x * 1e6, ex)
plt.xlabel('x (μm)')
plt.ylabel('Ex (V/m)')
plt.title('Two-Stream Instability - Electric Field')
plt.savefig('ex_field.png')

# Calculate plasma frequency
n_e = 1e20
omega_pe = units.plasma_frequency(n_e)
print(f"Plasma frequency: {omega_pe:.3e} rad/s")
print(f"Period: {2*np.pi/omega_pe*1e15:.2f} fs")
```

## Tutorial 2: Laser-Plasma Interaction (2D)

This tutorial demonstrates a 2D laser pulse interacting with a plasma slab.

### Step 1: Create the Input Deck

Create a file `laser_plasma.deck`:

```fortran
begin:control
  nx = 512
  ny = 256
  
  # Domain (10 μm × 5 μm)
  x_min = 0.0
  x_max = 10.0e-6
  y_min = -2.5e-6
  y_max = 2.5e-6
  
  t_end = 100.0e-15
  
  dt_multiplier = 0.95
end:control

begin:boundaries
  bc_x_min = simple_laser
  bc_x_max = open
  bc_y_min = periodic
  bc_y_max = periodic
end:boundaries

begin:constant
  # Laser parameters
  lambda = 800.0e-9        # Wavelength (m)
  omega = 2.0 * pi * c / lambda
  intensity = 1.0e18       # W/cm^2
  
  # Plasma parameters
  n_crit = epsilon0 * me * omega^2 / qe^2
  n_plasma = 0.1 * n_crit  # 10% critical density
  
  # Derived
  a0 = sqrt(intensity * 1e4 * lambda^2 / 1.37e18)
  
  # Geometry
  x_plasma_start = 3.0e-6
  x_plasma_end = 8.0e-6
end:constant

begin:laser
  boundary = x_min
  amp = a0
  omega = omega
  pol_angle = 0.0
  profile = gauss(y, 0, 1.0e-6)
  t_profile = gauss(time, 30.0e-15, 10.0e-15)
end:laser

begin:species
  name = electron
  charge = -1.0
  mass = 1.0
  nparticles_per_cell = 16
  
  number_density = if((x gt x_plasma_start) and (x lt x_plasma_end), n_plasma, 0)
  temp_ev = 10.0
end:species

begin:species
  name = proton
  charge = 1.0
  mass = 1836.0
  nparticles_per_cell = 4
  
  number_density = number_density(electron)
  temp_ev = 1.0
end:species

begin:output
  dt_snapshot = 5.0e-15
  
  # EM fields
  ex = always
  ey = always
  ez = always
  bx = always
  by = always
  bz = always
  
  # Particles
  px = always
  py = always
  pz = always
  weight = always
  
  # Distributions
  dist_fn = always
  dist_fn_px_py = always
  
  # Derived
  number_density = always + species
  ekbar = always + species
  charge_density = always
end:output

begin:dist_fn
  name = dist_electron
  ndims = 2
  
  direction1 = dir_px
  direction2 = dir_py
  
  range1 = (-5.0e-22, 5.0e-22)
  range2 = (-5.0e-22, 5.0e-22)
  
  resolution1 = 200
  resolution2 = 200
  
  include_species:electron
end:dist_fn
```

### Step 2: Run the 2D Simulation

```bash
mkdir laser_plasma
cd laser_plasma
cp ../laser_plasma.deck input.deck

# Run with GPU (2D requires more memory)
echo . | mpirun -n 1 ../epoch2d/bin/epoch2d_gpu
```

### Step 3: Visualize Results

```python
import sdf
import numpy as np
import matplotlib.pyplot as plt

# Load snapshot
data = sdf.read("0010.sdf")

# Get field data
ex = data.Electric_Field_Ex.data
ey = data.Electric_Field_Ey.data
x = data.Electric_Field_Ex.grid.data[0]
y = data.Electric_Field_Ex.grid.data[1]

# Calculate laser intensity
I = (ex**2 + ey**2) * 8.85e-12 * 3e8 / 2

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Electric field Ex
ax = axes[0, 0]
im = ax.pcolormesh(x*1e6, y*1e6, ex.T, cmap='RdBu_r', shading='auto')
ax.set_xlabel('x (μm)')
ax.set_ylabel('y (μm)')
ax.set_title('Ex Field')
plt.colorbar(im, ax=ax, label='V/m')

# Electron density
ne = data.Derived_Number_Density_electron.data
ax = axes[0, 1]
im = ax.pcolormesh(x*1e6, y*1e6, ne.T/1e25, cmap='hot', shading='auto')
ax.set_xlabel('x (μm)')
ax.set_ylabel('y (μm)')
ax.set_title('Electron Density')
plt.colorbar(im, ax=ax, label='×10²⁵ m⁻³')

# Laser intensity
ax = axes[1, 0]
im = ax.pcolormesh(x*1e6, y*1e6, I.T/1e18, cmap='inferno', shading='auto')
ax.set_xlabel('x (μm)')
ax.set_ylabel('y (μm)')
ax.set_title('Intensity')
plt.colorbar(im, ax=ax, label='×10¹⁸ W/cm²')

# Momentum distribution
px = data.dist_fn_dist_electron.data
ax = axes[1, 1]
im = ax.imshow(np.log10(px.T + 1), origin='lower', cmap='viridis')
ax.set_title('px-py Distribution (log)')

plt.tight_layout()
plt.savefig('laser_plasma_snapshot.png', dpi=150)
```

## Tutorial 3: GPU Performance Optimization

### Optimizing Particle Count

GPUs perform best with large numbers of particles:

```fortran
begin:species
  name = electron
  
  # Recommended: at least 100 particles/cell for GPU efficiency
  nparticles_per_cell = 100
  
  # For high-resolution runs: 
  # nparticles_per_cell = 500
end:species
```

### Optimizing Grid Size

Use grid sizes that are multiples of 32:

```fortran
begin:control
  # Good: powers of 2
  nx = 256
  ny = 256
  
  # Also good: multiples of 32
  nx = 320
  ny = 320
  
  # Avoid: odd or prime sizes
  # nx = 257  # Not optimal
end:control
```

### Memory Estimation

Estimate GPU memory requirements:

```python
from epoch_gpu.utils import units

# Grid memory
nx, ny = 512, 512
n_fields = 9  # Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz
grid_memory = nx * ny * n_fields * 8  # bytes (double precision)

# Particle memory
n_particles = 1_000_000
n_arrays = 7  # x, y, px, py, pz, weight, species
particle_memory = n_particles * n_arrays * 8

# Total
total_gb = (grid_memory + particle_memory) / 1e9
print(f"Estimated GPU memory: {total_gb:.2f} GB")
```

### Profiling GPU Kernels

```bash
# Enable verbose output
export NVCOMPILER_ACC_NOTIFY=1

# Run simulation
echo . | mpirun -n 1 ./epoch2d_gpu

# Profile with Nsight
nsys profile -o profile_report ./epoch2d_gpu
nsys-ui profile_report.nsys-rep
```

## Tutorial 4: Multi-GPU Simulations

For large-scale simulations, use multiple GPUs:

### Domain Decomposition

```bash
# 4 GPUs, 2x2 decomposition
mpirun -n 4 ./epoch2d_gpu
```

### SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=epoch_gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=epoch_%j.out

module load nvhpc cuda

# Set GPU-aware MPI
export EPOCH_GPU_AWARE_MPI=1

# Run with 8 GPUs (2 nodes × 4 GPUs)
srun ./epoch2d_gpu
```

### Scaling Test

```python
# Weak scaling analysis
import matplotlib.pyplot as plt

gpus = [1, 2, 4, 8]
time_per_step = [0.189, 0.098, 0.052, 0.028]  # ms
ideal = [time_per_step[0] / g for g in gpus]

plt.figure(figsize=(8, 6))
plt.loglog(gpus, time_per_step, 'o-', label='Measured')
plt.loglog(gpus, ideal, '--', label='Ideal')
plt.xlabel('Number of GPUs')
plt.ylabel('Time per step (ms)')
plt.legend()
plt.title('EPOCH-GPU Weak Scaling')
plt.savefig('weak_scaling.png')
```

## Common Issues and Solutions

### Out of GPU Memory

```fortran
! Reduce grid size or particle count
begin:control
  nx = 256  ! Reduce from 512
  ny = 256
end:control

begin:species
  nparticles_per_cell = 50  ! Reduce from 100
end:species
```

### Slow Current Deposition

This is expected due to atomic operations. Consider:

1. Using larger cells (coarser grid)
2. Reducing particles per cell
3. Future: histogram-based deposition

### NaN Values in Output

Check CFL condition:

```python
from epoch_gpu.utils import units

dx = 1e-7  # m
dt = 1e-16  # s
c = units.C

cfl = c * dt / dx
print(f"CFL number: {cfl}")
assert cfl < 1.0, "CFL condition violated!"
```

## Next Steps

- [Performance Tuning]({{< relref "performance.md" >}}) - Optimize your simulations
- [Multi-GPU Support]({{< relref "multi_gpu.md" >}}) - Scale to multiple GPUs
- [Troubleshooting]({{< relref "troubleshooting.md" >}}) - Debug common issues

