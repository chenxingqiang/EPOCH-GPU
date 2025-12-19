# EPOCH-GPU: GPU-Accelerated EPOCH

[![GPU Support](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Compute Capability](https://img.shields.io/badge/Compute%20Capability-7.0+-blue.svg)](https://developer.nvidia.com/cuda-gpus)
[![Dimensions](https://img.shields.io/badge/Dimensions-1D%20|%202D%20|%203D-orange.svg)]()
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-chenxingqiang%2FEPOCH--GPU-blue.svg)](https://github.com/chenxingqiang/EPOCH-GPU)
[![Upstream](https://img.shields.io/badge/Upstream-Warwick--Plasma%2Fepoch-lightgrey.svg)](https://github.com/Warwick-Plasma/epoch)

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

| Dimension | Status | GPU Kernels |
|-----------|--------|-------------|
| **epoch1d** | ✅ Ready | Field solver, Particle pusher, Current deposition |
| **epoch2d** | ✅ Ready | Field solver, Particle pusher, Current deposition, Bilinear interpolation |
| **epoch3d** | ✅ Ready | Field solver, Particle pusher, Current deposition, Trilinear interpolation |

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

### Performance Benchmarks (Tesla V100-32GB)

| Dimension | Kernel | Grid/Particles | Time/Iteration | Throughput |
|-----------|--------|----------------|----------------|------------|
| **1D** | Field Update | 1024 cells | 0.027 ms | - |
| **1D** | Particle Push | 1M particles | 0.17 ms | **5.9B particles/sec** |
| **1D** | Field Interpolation | 1M particles | 0.097 ms | - |
| **1D** | Current Deposition | 1M particles | 0.15 ms | - |
| **2D** | Field Update | 256×256 | 0.035 ms | 1.9B cells/sec |
| **2D** | Particle Push | 1M particles | 0.19 ms | **5.3B particles/sec** |
| **2D** | Field Interpolation | 1M particles | 0.11 ms | - |
| **2D** | Current Deposition | 1M particles | 0.49 ms | - |
| **3D** | Field Update | 64×64×64 | 0.096 ms | 2.7B cells/sec |
| **3D** | Particle Push | 500K particles | 0.11 ms | **4.4B particles/sec** |
| **3D** | Field Interpolation | 500K particles | 0.35 ms | - |
| **3D** | Current Deposition | 500K particles | 0.38 ms | - |

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

