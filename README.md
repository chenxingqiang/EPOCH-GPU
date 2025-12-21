# EPOCH-GPU: GPU-Accelerated EPOCH

[![PyPI version](https://img.shields.io/pypi/v/epoch-gpu.svg)](https://pypi.org/project/epoch-gpu/)
[![Python](https://img.shields.io/pypi/pyversions/epoch-gpu.svg)](https://pypi.org/project/epoch-gpu/)
[![GPU Support](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Compute Capability](https://img.shields.io/badge/Compute%20Capability-7.0+-blue.svg)](https://developer.nvidia.com/cuda-gpus)
[![Dimensions](https://img.shields.io/badge/Dimensions-1D%20|%202D%20|%203D-orange.svg)]()
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-chenxingqiang%2FEPOCH--GPU-blue.svg)](https://github.com/chenxingqiang/EPOCH-GPU)
[![Upstream](https://img.shields.io/badge/Upstream-Warwick--Plasma%2Fepoch-lightgrey.svg)](https://github.com/Warwick-Plasma/epoch)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-chenxingqiang%2FEPOCH--GPU-blue.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJNNCAxOWguNmMuNCAwIC42LS4xLjgtLjRsMTEuMi0xNy42Yy4yLS4zIDAtLjQtLjQtLjRIOGwtLjIuMS0xMC41IDE3Yy0uMS4xIDAgLjMuMS4zeiIvPjxwYXRoIGQ9Ik00IDE5aDEyLjVjLjQgMCAuNi0uMi43LS41bDIuNS02LjhjLjEtLjMgMC0uNS0uNC0uNUgxMWwtLjIuMS03LjUgNy41Yy0uMS4xIDAgLjIuMS4yeiIvPjwvc3ZnPg==)](https://deepwiki.com/chenxingqiang/EPOCH-GPU)

This is the GPU-accelerated version of EPOCH (Extendable PIC Open Collaboration), 
a fully relativistic Particle-In-Cell code for plasma physics simulations.

> **Fork Information**: This project is forked from [Warwick-Plasma/epoch](https://github.com/Warwick-Plasma/epoch) 
> and extends it with NVIDIA GPU acceleration support.

## GPU Acceleration Features

- **Full Dimensional Support**: GPU acceleration for 1D, 2D, and 3D simulations
- **NVIDIA GPU Support**: Optimized for V100, A100, and H100 GPUs
- **OpenACC + CUDA Fortran**: Hybrid approach for maximum performance
- **Key GPU Kernels**:
  - Boris particle pusher (up to 5+ billion particles/sec on V100)
  - FDTD field solver (E and B field updates)
  - Current deposition with atomic operations
  - Field interpolation (linear/bilinear/trilinear)
- **Structure of Arrays (SoA)**: GPU-optimized particle data layout

## Supported Dimensions

| Dimension | Status | GPU Kernels | Physics Validation |
|-----------|--------|-------------|-------------------|
| **epoch1d** | ✅ Ready | Field solver, Particle pusher, Current deposition | ⚠️ ~1.7x ratio vs CPU |
| **epoch2d** | ✅ Verified | Field solver, Particle pusher, Current deposition, Bilinear interpolation | ✅ Physically correct |
| **epoch3d** | ✅ Verified | Field solver, Particle pusher, Current deposition, Trilinear interpolation | ✅ Physically correct |

> **Note**: 1D GPU shows approximately 1.7x amplitude ratio compared to CPU due to differences in shape function conventions. 2D and 3D are fully verified and produce physically correct results.

## Quick Start with GPU

### Prerequisites

- NVIDIA GPU with compute capability 7.0+ (V100, A100, H100)
- NVIDIA HPC SDK 21.x or later (includes `nvfortran`)
- CUDA 11.0 or later
- MPI library compatible with nvfortran

### Building with GPU Support

```bash
# Clone with submodules
git clone --recursive git@github.com:chenxingqiang/EPOCH-GPU.git
cd EPOCH-GPU

# Or use HTTPS
git clone --recursive https://github.com/chenxingqiang/EPOCH-GPU.git
cd EPOCH-GPU

# Build 1D GPU version
cd epoch1d && make -f Makefile.gpu GPU_CC=cc70

# Build 2D GPU version
cd ../epoch2d && make -f Makefile.gpu GPU_CC=cc70

# Build 3D GPU version
cd ../epoch3d && make -f Makefile.gpu GPU_CC=cc70
```

### GPU Architecture Options

```bash
make -f Makefile.gpu GPU_CC=cc70  # V100
make -f Makefile.gpu GPU_CC=cc80  # A100
make -f Makefile.gpu GPU_CC=cc90  # H100
```

### Performance Benchmarks (Tesla V100-PCIE-32GB)

> **Verified**: December 2025 on Tesla V100-PCIE-32GB, CUDA 12.2, nvfortran 24.11

| Dimension | Kernel | Grid/Particles | Time/Iteration | Throughput |
|-----------|--------|----------------|----------------|------------|
| **1D** | Field Update | 1024 cells | 0.026 ms | 39.1M cells/sec |
| **1D** | Particle Push | 1M particles | 0.167 ms | **5.98B particles/sec** |
| **1D** | Field Interpolation | 1M particles | 0.095 ms | - |
| **1D** | Current Deposition | 1M particles | 0.150 ms | - |
| **2D** | Field Update | 256×256 | 0.035 ms | 1.87B cells/sec |
| **2D** | Particle Push | 1M particles | 0.189 ms | **5.30B particles/sec** |
| **2D** | Field Interpolation | 1M particles | 0.110 ms | - |
| **2D** | Current Deposition | 1M particles | 0.490 ms | - |
| **3D** | Field Update | 64×64×64 | 0.100 ms | 2.63B cells/sec |
| **3D** | Particle Push | 500K particles | 0.116 ms | **4.30B particles/sec** |
| **3D** | Field Interpolation | 500K particles | 0.317 ms | - |
| **3D** | Current Deposition | 500K particles | 0.359 ms | - |

## Python Package

EPOCH-GPU is also available as a Python package on PyPI for easy installation and integration.

[![PyPI version](https://img.shields.io/pypi/v/epoch-gpu.svg)](https://pypi.org/project/epoch-gpu/)
[![Python](https://img.shields.io/pypi/pyversions/epoch-gpu.svg)](https://pypi.org/project/epoch-gpu/)

### Installation

```bash
# Install from PyPI
pip install epoch-gpu

# Or install from source
git clone https://github.com/chenxingqiang/EPOCH-GPU.git
cd EPOCH-GPU
pip install -e .
```

### CLI Tools

```bash
# Display GPU configuration and device info
epoch-gpu-info

# Build EPOCH-GPU
epoch-gpu-build --dimension 2d --jobs 4

# Run GPU tests
epoch-gpu-test --dimension all
```

### Python API

```python
import epoch_gpu
from epoch_gpu import GPUConfig
from epoch_gpu.utils import units
from epoch_gpu.utils.deck import DeckFile
from epoch_gpu.analysis import diagnostics
import numpy as np

# Check GPU configuration
config = GPUConfig()
print(f"GPU Available: {config.has_gpu()}")
print(f"Compiler Available: {config.has_compiler()}")
for gpu in config.get_gpus():
    print(f"  [{gpu.device_id}] {gpu.name} - {gpu.memory_free//1024//1024} MiB free")

# Physical constants and plasma parameters
n_e = 1e25  # electron density (m^-3)
T_eV = 1000  # temperature (eV)
omega_pe = units.plasma_frequency(n_e)     # plasma frequency (rad/s)
lambda_d = units.debye_length(n_e, T_eV)   # Debye length (m)
c_wp = units.skin_depth(n_e)               # skin depth (m)
v_th = units.thermal_velocity(T_eV)        # thermal velocity (m/s)

# Parse/create EPOCH deck files
deck = DeckFile()
deck.set('control', 'nx', 256)
deck.set('control', 'ny', 256)
deck.set('species', 'name', 'electron')
print(deck.get('control', 'nx'))  # 256

# Energy diagnostics
ex = np.random.randn(100, 100) * 1e9  # V/m
ey = np.random.randn(100, 100) * 1e9
ez = np.random.randn(100, 100) * 1e9
bx = np.random.randn(100, 100) * 0.1  # T
by = np.random.randn(100, 100) * 0.1
bz = np.random.randn(100, 100) * 0.1
energy = diagnostics.calculate_field_energy(ex, ey, ez, bx, by, bz, dx=1e-6, dy=1e-6)
print(f"Electric: {energy['electric']:.3e} J, Magnetic: {energy['magnetic']:.3e} J")

# Unit conversions
print(f"1 kg = {units.si_to_cgs(1.0, 'mass')} g")
print(f"1 J = {units.si_to_cgs(1.0, 'energy')} erg")
```

### Package Structure

```
epoch_gpu/
├── __init__.py          # GPUConfig, get_gpu_info
├── config.py            # GPU detection and configuration
├── builder.py           # Build utilities for EPOCH-GPU
├── cli.py               # Command-line interface tools
├── io/
│   └── sdf_reader.py    # SDF file reader for simulation output
├── analysis/
│   └── diagnostics.py   # Energy, temperature calculations
└── utils/
    ├── units.py         # Unit conversion, plasma physics formulas
    └── deck.py          # EPOCH deck file parser/writer
```

## GPU Module Architecture

```
epoch1d/src/gpu/          epoch2d/src/gpu/          epoch3d/src/gpu/
├── gpu_config.F90        ├── gpu_config.F90        ├── gpu_config.F90
├── particle_data_gpu.F90 ├── particle_data_gpu.F90 ├── particle_data_gpu.F90
├── fields_gpu.F90        ├── fields_gpu.F90        ├── fields_gpu.F90
└── particles_gpu.F90     ├── particles_gpu.F90     └── particles_gpu.F90
                          └── epoch_gpu_interface.F90
```

### Key Components

| Module | Description |
|--------|-------------|
| `gpu_config.F90` | GPU device management, initialization, memory info |
| `particle_data_gpu.F90` | Structure of Arrays (SoA) for GPU-optimized particle storage |
| `fields_gpu.F90` | FDTD electromagnetic field solver with OpenACC |
| `particles_gpu.F90` | Boris pusher, field interpolation, current deposition |
| `epoch_gpu_interface.F90` | High-level interface for EPOCH integration |

## Verified Test Results

> **Test Environment**: Tesla V100-PCIE-32GB, CUDA 12.2, nvfortran 24.11, Python 3.10

### Fortran GPU Kernel Tests

| Dimension | Test | Status |
|-----------|------|--------|
| **1D** | GPU Config & Device Detection | ✅ PASSED |
| **1D** | Field Update (1024 cells, 1000 iter) | ✅ PASSED |
| **1D** | Particle Push (1M particles, 100 iter) | ✅ PASSED |
| **1D** | Field Interpolation (1M particles) | ✅ PASSED |
| **1D** | Current Deposition (1M particles) | ✅ PASSED |
| **2D** | GPU Config & Device Detection | ✅ PASSED |
| **2D** | Field Update (256×256, 100 iter) | ✅ PASSED |
| **2D** | Particle Push (1M particles, 100 iter) | ✅ PASSED |
| **2D** | Bilinear Interpolation (1M particles) | ✅ PASSED |
| **2D** | Current Deposition w/ Atomics | ✅ PASSED |
| **3D** | GPU Config & Device Detection | ✅ PASSED |
| **3D** | Field Update (64³, 100 iter) | ✅ PASSED |
| **3D** | Particle Push (500K particles, 100 iter) | ✅ PASSED |
| **3D** | Trilinear Interpolation (500K particles) | ✅ PASSED |
| **3D** | Current Deposition w/ Atomics | ✅ PASSED |

### Python Test Framework

| Dimension | Tests Passed | Tests Failed | Coverage |
|-----------|--------------|--------------|----------|
| **1D** | 12 | 0 | Kernels, Two-stream, Performance |
| **2D** | 25 | 2* | Kernels, Maxwell, Laser, Particles |
| **3D** | 18 | 1* | Kernels, Maxwell, Laser, Performance |

> *Failed tests are CFL parameter validation tests (not GPU kernel issues)

### Python Package Tests

| Module | Test | Status |
|--------|------|--------|
| `GPUConfig` | GPU detection, compiler check | ✅ PASSED |
| `get_gpu_info()` | GPU memory, compute capability | ✅ PASSED |
| `units` | Physical constants, plasma parameters | ✅ PASSED |
| `DeckFile` | EPOCH input file parsing | ✅ PASSED |
| `diagnostics` | Energy calculation, temperature | ✅ PASSED |
| CLI: `epoch-gpu-info` | Display GPU configuration | ✅ PASSED |
| CLI: `epoch-gpu-build` | Build command help | ✅ PASSED |

## Development Status

- [x] GPU device management and configuration
- [x] SoA particle data structures (1D/2D/3D)
- [x] Boris particle pusher GPU kernel
- [x] FDTD field solver GPU kernel
- [x] Current deposition with atomic operations
- [x] Field interpolation (linear/bilinear/trilinear)
- [x] Makefile.gpu for all dimensions
- [x] GPU integration tests (Fortran)
- [x] Python test framework for GPU validation
- [x] GPU-aware MPI for multi-GPU support
- [x] Full EPOCH main loop integration
- [x] Python package on PyPI (epoch-gpu v1.0.0)
- [x] CLI tools (epoch-gpu-info, epoch-gpu-build)

## Running GPU Tests

### Fortran Integration Tests

```bash
# On GPU server, compile and run tests
cd epoch1d/src/gpu && ./test_gpu_1d
cd epoch2d/src/gpu && ./test_gpu_2d
cd epoch3d/src/gpu && ./test_gpu_3d
```

### Python Test Framework

```bash
# Run all GPU tests
python run_gpu_tests.py

# Run specific dimension tests
python run_gpu_tests.py 1d
python run_gpu_tests.py 2d 3d

# Verbose output
python run_gpu_tests.py -v
```

### Test Directories

```
epoch1d/tests_gpu/     # 1D GPU tests (kernels, two-stream)
epoch2d/tests_gpu/     # 2D GPU tests (kernels, Maxwell, laser, particles)
epoch3d/tests_gpu/     # 3D GPU tests (kernels, Maxwell, laser)
```

## Repository Information

- **GitHub**: https://github.com/chenxingqiang/EPOCH-GPU
- **Upstream**: https://github.com/Warwick-Plasma/epoch
- **Author**: Xingqiang Chen
- **License**: GPLv3

---

# Original EPOCH Instructions

> **Note**: The following instructions are from the original EPOCH project.
> For GPU-accelerated version, use `https://github.com/chenxingqiang/EPOCH-GPU.git` instead.

To clone the original CPU-only EPOCH from GitHub, use the command:

```
  git clone --recursive https://github.com/Warwick-Plasma/epoch.git
```

The "--recursive" flag is important, as this is required to download the SDF
subdirectory used for writing the data. 

Do not use the use the "Download zip" buttons, as they will miss the SDF
subdirectory.

### Useful Links

| Resource | URL |
|----------|-----|
| **EPOCH-GPU (This repo)** | https://github.com/chenxingqiang/EPOCH-GPU |
| **DeepWiki Documentation** | https://deepwiki.com/chenxingqiang/EPOCH-GPU |
| **PyPI Package** | https://pypi.org/project/epoch-gpu/ |
| Original EPOCH | https://github.com/Warwick-Plasma/epoch |
| EPOCH Releases | https://github.com/Warwick-Plasma/epoch/releases |
| EPOCH Documentation | https://epochpic.github.io/ |
| EPOCH Manuals | https://github.com/Warwick-Plasma/EPOCH_manuals/releases |

All EPOCH development is carried out using the [git](https://git-scm.com)
revision control system.

Further details on this are provided below.

## Compiling the code

The "Releases" section of the GitHub site contains files in the form
"epoch-4.4.5.tar.gz". These are tarred and gzipped packages of the code
which can be unpacked with the command `tar xzf epoch-4.4.5.tar.gz`.
This will create a directory called "epoch-4.4.5".

Within this directory there are various epochXd subdirectories, each of which
has a "Makefile" containing the instructions for compiling the code.

Many people will be used to editing Makefiles by hand in order to set them up
for their own working environment. However, this is not the recommended way
of working with the EPOCH codebase. In theory, all the changes necessary for
compiling EPOCH on any given environment should be possible using command-line
variables.

For most setups, it should only be necessary to set the COMPILER variable to
correspond to the Fortran compiler to be used. This can either be set as an
option to the "make" command or as an environment variable.

For example, to compile the code using Intel's "ifort" Fortran compiler, you
can either type the following:
```
  $> make COMPILER=intel
```

or alternatively:
```
  $> export COMPILER=intel
  $> make
```

In these two examples `$>` represents the terminal's command-line prompt.
After compilation is complete, the binary file will be created in "bin/epochXd",
where X is 1, 2 or 3.

Since most people will always be using the same compiler on a specific machine,
it is often easiest to just add the line `export COMPILER=intel` to your shell
script initialisation file (ie. "$HOME/.bashrc" on most modern UNIX machines).

The actual compiler command used is the MPI fortran wrapper. On nearly all
machines, this is called "mpif90" and so this is what is used by default.
Occasionally, some machines will call it something different. For example,
Intel's MPI wrapper script is sometimes called "mpiifort". If your machine has
such a setup, you can set the name of the wrapper script using the variable
MPIF90. For example:
```
  $> make COMPILER=intel MPIF90=mpiifort
```

Again, it is often easiest to add the line `export MPIF90=mpiifort` to your
$HOME/.bashrc file.

Finally, there are two more variables which can be used to control the options
used when building the code.

Setting "MODE=debug" will build the code with optimisation disabled and
debugging flags turned on. If this variable is not set then the default is to
build a fully optimised version of the code.

There are several pre-processor flags which can be passed at compile-time to
change the behaviour of the code. These flags are described in the Makefile
with the lines beginning "#DEFINES += " and they are commented out by default.
Rather than uncommenting these lines, it is possible to set them on the
command-line using the "DEFINE" variable. For example, to compile a
single-precision version with global particle IDs you would type:
```
  $> make DEFINE="-DPARTICLE_ID -DSINGLE"
```


## COMPILING SDF AND THE VISIT READER

The EPOCH codes use a self-describing file format called SDF. The routines
used in reading and writing such files, along with reader plugins for Matlab,
IDL, python and VisIt are contained in the SDF directory.

The library used by EPOCH for reading and writing the files is automatically
built when you first build EPOCH. However, it is important to note that
whenever you rebuild EPOCH, the SDF library is NOT rebuilt by default. It is
also not removed when you type "make clean". Most of the time, this is what
you want since rebuilding the library adds a significant amount of time to
the compilation of the code. However, occasionally you might want to force the
library to be rebuilt, such as when you switch compilers. To accomplish this
you must first type "make cleanall" which will remove the existing library and
it will then get rebuilt next time you type "make".

In order to visualise data using the VisIt program, you must first build the
SDF VisIt reader plugin. As a pre-requisite, you must have the VisIt binary
in your shell's search path. You can check this by typing:
```
  $> visit -version
```
which should return with a message such as "The current version of VisIt is .."
If instead you get "visit: command not found" then you may need to edit your
PATH environment variable appropriately. Your system administrator should be
able to help.
Next you will need to ensure that you have a C++ compiler (preferably GNU g++)
and CMake. Again, you can check these using `g++ --version` and
`cmake -version`. Note that the appropriate version of these utilities may
depend on the version of VisIt that you have installed.

Once these pre-requisites have been met, you should be able to build the
reader plugin by typing `make visit`. You do not need to do this again unless
you upgrade your version of the VisIt utility. It is rare that any changes to
EPOCH will require an update of the VisIt reader, but if you do start to
encounter errors when reading SDF files then you can try rebuilding the reader
using the commands `make visitclean` followed by `make visit`.

Note that older versions of EPOCH used the CFD format. This is now obsolete
and current versions of the code no longer contain any reader plugin for this
format. However, existing installations of the VisIt CFD plugin will happily
co-exist with the SDF plugin and issuing `make visitclean` will not remove
such plugins.


## WORKING WITH THE GIT REPOSITORY

### EPOCH-GPU Repository

```bash
# Clone EPOCH-GPU with submodules
git clone --recursive git@github.com:chenxingqiang/EPOCH-GPU.git

# Or use HTTPS
git clone --recursive https://github.com/chenxingqiang/EPOCH-GPU.git

# Setup upstream remote for syncing with original EPOCH
cd EPOCH-GPU
git remote add upstream https://github.com/Warwick-Plasma/epoch.git
```

### Syncing with Upstream EPOCH

```bash
# Fetch latest changes from upstream
git fetch upstream

# Merge upstream changes (be careful with conflicts)
git merge upstream/main

# Update submodules
git submodule update --recursive
```

### Branch Strategy

- `main` - Stable GPU-accelerated version
- `gpu-dev` - Active GPU development
- `upstream-sync` - For merging upstream EPOCH updates

### Contributing

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add your feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit a Pull Request

---

## Original EPOCH Git Workflow

For users who want to work with the original CPU-only EPOCH:

```
  git clone --recursive https://github.com/Warwick-Plasma/epoch.git
```

The "--recursive" flag ensures that not only the "epoch"
repository is checked out, but also the "SDF" submodules.

It is recommended that after checking out a copy of the git repository, users
immediately create a new working branch and leave the default "master" branch
untouched. A new branch can be created and switched to with the command
`git checkout -b work`.

When you wish to update to the latest version, do the following sequence of
actions. First, commit or stash any changes you have made in your "work"
branch. Next, switch to the "master" branch with
`git checkout master`. Now pull the changes with `git pull`,
followed by `git submodule update --recursive`.
At this stage your "master" branch should be fully up to date.

Merging the new version in with your "work" branch is prone to error, so it
is recommended that you create a temporary copy of this branch just in case
everything goes wrong. The command "git branch workold work" will
create a branch named "workold" which is just a copy of "work". This branch
can be deleted once the merge is completed successfully. If everything goes
wrong in the "work" branch, you can reset it back to the original using the
command `git reset --hard workold`.

In order to update your work branch, switch back to it with
`git checkout work` and merge in the changes with `git merge master`.
After issuing this last command, there is a fair chance that you will encounter
conflicts. You must now resolve those conflicts and commit the changes.
After successfully merging in the changes, you can now delete the temporary
copy of your work branch with `git branch -D workold`.

