---
title: Prerequisites & Environment Setup
linktitle: Prerequisites
toc: true
type: docs
date: "2024-12-20T00:00:00Z"
draft: false
menu:
  gpu:
    parent: GPU Guide
    weight: 2
---

This guide provides detailed instructions for setting up the GPU environment
for EPOCH-GPU on various platforms and configurations.

## Hardware Requirements

### Supported NVIDIA GPUs

EPOCH-GPU requires NVIDIA GPUs with **Compute Capability 7.0 or higher**:

| GPU Model | Compute Capability | Memory | Architecture | Recommended Use |
|-----------|-------------------|--------|--------------|-----------------|
| Tesla V100 | 7.0 | 16/32 GB | Volta | Production runs |
| Tesla A100 | 8.0 | 40/80 GB | Ampere | Large-scale simulations |
| Tesla H100 | 9.0 | 80 GB | Hopper | Maximum performance |
| RTX 2080 Ti | 7.5 | 11 GB | Turing | Development |
| RTX 3080/3090 | 8.6 | 10/24 GB | Ampere | Development/testing |
| RTX 4080/4090 | 8.9 | 16/24 GB | Ada Lovelace | Development/testing |
| A6000 | 8.6 | 48 GB | Ampere | Production |
| L40/L40S | 8.9 | 48 GB | Ada Lovelace | Production |

> **Note**: GPUs with Compute Capability < 7.0 (e.g., GTX 1080, P100) are NOT supported.

### How to Check Your GPU

```bash
# Method 1: nvidia-smi
nvidia-smi

# Method 2: Query specific properties
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# Method 3: Using Python
python3 -c "import subprocess; print(subprocess.check_output(['nvidia-smi', '-L']).decode())"
```

Example output:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|=========================================+======================+======================|
|   0  Tesla V100-PCIE-32GB           On  | 00000000:67:04.0 Off |                    0 |
| N/A   39C    P0              26W / 250W |      0MiB / 32768MiB |      0%      Default |
+-----------------------------------------+----------------------+----------------------+
```

### GPU Memory Requirements

| Simulation Type | Grid Size | Particles | Estimated Memory |
|-----------------|-----------|-----------|------------------|
| Small test | 128² | 100K | ~500 MB |
| Medium 2D | 256² | 1M | ~1 GB |
| Large 2D | 512² | 10M | ~6 GB |
| Small 3D | 64³ | 500K | ~1.5 GB |
| Medium 3D | 128³ | 5M | ~8 GB |
| Large 3D | 256³ | 50M | ~50 GB |

Memory estimation formula:
```
GPU Memory (bytes) = 
    Grid: nx × ny × nz × 9 fields × 8 bytes
  + Particles: n_particles × 7 arrays × 8 bytes
  + Overhead: ~20%
```

---

## Software Requirements Overview

| Component | Minimum Version | Recommended Version | Purpose |
|-----------|-----------------|---------------------|---------|
| NVIDIA Driver | 450.x | 535.x+ | GPU access |
| CUDA Toolkit | 11.0 | 12.2+ | CUDA runtime |
| NVIDIA HPC SDK | 21.x | 24.11 | nvfortran compiler |
| MPI | Any | NVIDIA HPC MPI | Parallel execution |
| Python | 3.8 | 3.10+ | Testing & analysis |

---

## Installation Guide by Platform

### Ubuntu/Debian Linux

#### Step 1: Install NVIDIA Driver

```bash
# Check current driver
nvidia-smi

# If no driver, install from Ubuntu repository
sudo apt update
sudo apt install nvidia-driver-535

# Or use NVIDIA's repository for latest drivers
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-545

# Reboot required
sudo reboot
```

#### Step 2: Install CUDA Toolkit (Optional, usually bundled with HPC SDK)

```bash
# Download CUDA from NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run

# Install (driver already installed, so skip it)
sudo sh cuda_12.2.0_535.54.03_linux.run --toolkit --silent

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Step 3: Install NVIDIA HPC SDK

**Method A: Using .deb package (Recommended)**

```bash
# Download HPC SDK
wget https://developer.download.nvidia.com/hpc-sdk/24.11/nvhpc-24-11_24.11_amd64.deb
wget https://developer.download.nvidia.com/hpc-sdk/24.11/nvhpc-2024_24.11_amd64.deb

# Install
sudo apt install ./nvhpc-24-11_24.11_amd64.deb ./nvhpc-2024_24.11_amd64.deb

# Setup environment
export NVHPC=/opt/nvidia/hpc_sdk
export PATH=$NVHPC/Linux_x86_64/24.11/compilers/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC/Linux_x86_64/24.11/compilers/lib:$LD_LIBRARY_PATH
export MANPATH=$NVHPC/Linux_x86_64/24.11/compilers/man:$MANPATH

# Make permanent
echo 'export NVHPC=/opt/nvidia/hpc_sdk' >> ~/.bashrc
echo 'export PATH=$NVHPC/Linux_x86_64/24.11/compilers/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$NVHPC/Linux_x86_64/24.11/compilers/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Method B: Using tarball**

```bash
# Download tarball
wget https://developer.download.nvidia.com/hpc-sdk/24.11/nvhpc_2024_2411_Linux_x86_64_cuda_12.6.tar.gz

# Extract
tar xpzf nvhpc_2024_2411_Linux_x86_64_cuda_12.6.tar.gz

# Install to /opt/nvidia/hpc_sdk
sudo ./nvhpc_2024_2411_Linux_x86_64_cuda_12.6/install

# Follow on-screen instructions
```

**Method C: Using module system (HPC clusters)**

```bash
# List available modules
module avail nvhpc

# Load module
module load nvhpc/24.11

# Make it permanent
echo 'module load nvhpc/24.11' >> ~/.bashrc
```

#### Step 4: Verify Installation

```bash
# Check nvfortran
nvfortran --version
# Expected: nvfortran 24.11-0 64-bit target on x86-64 Linux

# Check CUDA
nvcc --version
# Expected: Cuda compilation tools, release 12.x

# Check GPU access
nvaccelinfo
# Should show GPU details
```

---

### RHEL/CentOS/Rocky Linux

#### Step 1: Install NVIDIA Driver

```bash
# Enable EPEL repository
sudo dnf install epel-release

# Install NVIDIA driver
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf install nvidia-driver

# Reboot
sudo reboot
```

#### Step 2: Install HPC SDK

```bash
# Add NVIDIA HPC SDK repository
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/hpc-sdk/rhel/nvhpc.repo

# Install
sudo dnf install nvhpc-24.11

# Setup environment (same as Ubuntu)
export NVHPC=/opt/nvidia/hpc_sdk
export PATH=$NVHPC/Linux_x86_64/24.11/compilers/bin:$PATH
```

---

### Cloud Platforms

#### AWS (Amazon Web Services)

**Recommended Instances:**
- `p3.2xlarge` - 1x V100 (16 GB)
- `p3.8xlarge` - 4x V100 (64 GB total)
- `p4d.24xlarge` - 8x A100 (320 GB total)

**Setup on AWS EC2:**

```bash
# Use Deep Learning AMI (Ubuntu) - comes with drivers pre-installed
# After launching instance:

# Verify GPU
nvidia-smi

# Install HPC SDK
wget https://developer.download.nvidia.com/hpc-sdk/24.11/nvhpc-24-11_24.11_amd64.deb
sudo apt install ./nvhpc-24-11_24.11_amd64.deb

# Setup environment
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/bin:$PATH
```

#### Google Cloud Platform (GCP)

**Recommended Machine Types:**
- `n1-standard-8` + 1x V100
- `a2-highgpu-1g` - 1x A100
- `a2-highgpu-8g` - 8x A100

**Setup:**

```bash
# Use Deep Learning VM Image
# After instance creation:

# Install HPC SDK
curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list
sudo apt update
sudo apt install nvhpc-24-11
```

#### Chinese Cloud Platforms (国内云平台)

**AutoDL / 矩池云 / 恒源云:**

这些平台通常预装了 NVIDIA 驱动和 CUDA，只需安装 HPC SDK：

```bash
# 检查预装环境
nvidia-smi
nvcc --version

# 安装 HPC SDK (如果没有)
# 国内镜像可能较慢，建议使用清华镜像或自行上传
wget https://developer.download.nvidia.com/hpc-sdk/24.11/nvhpc_2024_2411_Linux_x86_64_cuda_12.6.tar.gz

# 或者使用 conda (如果可用)
conda install -c nvidia nvhpc
```

---

### Docker Container

#### Using NVIDIA NGC Container

```dockerfile
# Dockerfile for EPOCH-GPU
FROM nvcr.io/nvidia/nvhpc:24.11-devel-cuda12.6-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    make \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Clone EPOCH-GPU
RUN git clone --recursive https://github.com/chenxingqiang/EPOCH-GPU.git /opt/epoch-gpu

# Build
WORKDIR /opt/epoch-gpu/epoch2d
RUN make -f Makefile.gpu GPU_CC=cc70

# Set entrypoint
WORKDIR /workspace
```

**Build and run:**

```bash
# Build container
docker build -t epoch-gpu .

# Run with GPU access
docker run --gpus all -it epoch-gpu

# Or with nvidia-docker
nvidia-docker run -it epoch-gpu
```

#### Using Singularity (for HPC)

```bash
# Create Singularity definition file
cat > epoch-gpu.def << 'EOF'
Bootstrap: docker
From: nvcr.io/nvidia/nvhpc:24.11-devel-cuda12.6-ubuntu22.04

%post
    apt-get update && apt-get install -y git make
    git clone --recursive https://github.com/chenxingqiang/EPOCH-GPU.git /opt/epoch-gpu
    cd /opt/epoch-gpu/epoch2d && make -f Makefile.gpu GPU_CC=cc70

%environment
    export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/bin:$PATH

%runscript
    exec /opt/epoch-gpu/epoch2d/bin/epoch2d_gpu "$@"
EOF

# Build
singularity build epoch-gpu.sif epoch-gpu.def

# Run
singularity run --nv epoch-gpu.sif
```

---

## MPI Configuration

### NVIDIA HPC SDK MPI (Recommended)

```bash
# Add to PATH
export PATH=$NVHPC/Linux_x86_64/24.11/comm_libs/mpi/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC/Linux_x86_64/24.11/comm_libs/mpi/lib:$LD_LIBRARY_PATH

# Verify
mpirun --version
# Expected: Open MPI or similar

# Test
mpirun -n 2 hostname
```

### OpenMPI with CUDA Support

```bash
# Install OpenMPI with CUDA support
sudo apt install openmpi-bin libopenmpi-dev

# Check CUDA awareness
ompi_info | grep cuda
# Should show: MPI extensions: cuda

# If not, compile OpenMPI with CUDA:
./configure --with-cuda=/usr/local/cuda --enable-mpi-fortran
make -j$(nproc)
sudo make install
```

### MPICH

```bash
# Install MPICH
sudo apt install mpich

# Use with nvfortran
export MPIF90=mpif90
make COMPILER=nvidia
```

---

## Environment Variables Reference

### Essential Variables

```bash
# NVIDIA HPC SDK base
export NVHPC=/opt/nvidia/hpc_sdk

# Compiler paths
export PATH=$NVHPC/Linux_x86_64/24.11/compilers/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC/Linux_x86_64/24.11/compilers/lib:$LD_LIBRARY_PATH

# MPI paths
export PATH=$NVHPC/Linux_x86_64/24.11/comm_libs/mpi/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC/Linux_x86_64/24.11/comm_libs/mpi/lib:$LD_LIBRARY_PATH

# CUDA paths (if separate installation)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### GPU Runtime Variables

```bash
# Select specific GPU
export CUDA_VISIBLE_DEVICES=0        # Use only GPU 0
export CUDA_VISIBLE_DEVICES=0,1      # Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=""       # Disable all GPUs

# GPU memory allocation
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1  # Force device allocation

# Debugging
export NVCOMPILER_ACC_NOTIFY=1       # Show kernel launches
export NVCOMPILER_ACC_NOTIFY=3       # Verbose output
export NVCOMPILER_ACC_DEBUG=1        # Debug mode
export NVCOMPILER_ACC_TIME=1         # Timing information

# GPU-aware MPI
export EPOCH_GPU_AWARE_MPI=1         # Enable GPU-aware MPI
export EPOCH_GPUS_PER_NODE=4         # GPUs per node (for multi-GPU)
```

### Complete .bashrc Example

```bash
# Add to ~/.bashrc for permanent setup

# NVIDIA HPC SDK
export NVHPC=/opt/nvidia/hpc_sdk
export NVHPC_VERSION=24.11
export PATH=$NVHPC/Linux_x86_64/$NVHPC_VERSION/compilers/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC/Linux_x86_64/$NVHPC_VERSION/compilers/lib:$LD_LIBRARY_PATH
export MANPATH=$NVHPC/Linux_x86_64/$NVHPC_VERSION/compilers/man:$MANPATH

# MPI from HPC SDK
export PATH=$NVHPC/Linux_x86_64/$NVHPC_VERSION/comm_libs/mpi/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC/Linux_x86_64/$NVHPC_VERSION/comm_libs/mpi/lib:$LD_LIBRARY_PATH

# CUDA (if needed)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# EPOCH-GPU specific
export EPOCH_GPU_HOME=/path/to/EPOCH-GPU
alias epoch1d='$EPOCH_GPU_HOME/epoch1d/bin/epoch1d_gpu'
alias epoch2d='$EPOCH_GPU_HOME/epoch2d/bin/epoch2d_gpu'
alias epoch3d='$EPOCH_GPU_HOME/epoch3d/bin/epoch3d_gpu'
```

---

## Verification Script

Save this as `check_epoch_gpu_env.sh`:

```bash
#!/bin/bash
set -e

echo "=============================================="
echo "  EPOCH-GPU Environment Verification"
echo "=============================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}!${NC} $1"
}

echo "=== 1. GPU Hardware ==="
if nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    check_pass "GPU detected: $GPU_NAME ($GPU_MEM)"
    
    # Check compute capability
    CC_MAJOR=$(echo $GPU_CC | cut -d'.' -f1)
    if [ "$CC_MAJOR" -ge 7 ]; then
        check_pass "Compute Capability: $GPU_CC (≥7.0 required)"
    else
        check_fail "Compute Capability: $GPU_CC (≥7.0 required)"
    fi
else
    check_fail "No NVIDIA GPU detected"
fi
echo ""

echo "=== 2. NVIDIA Driver ==="
if nvidia-smi &> /dev/null; then
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    check_pass "Driver version: $DRIVER"
else
    check_fail "NVIDIA driver not installed"
fi
echo ""

echo "=== 3. CUDA Toolkit ==="
if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version | grep release | awk '{print $5}' | tr -d ',')
    check_pass "CUDA version: $CUDA_VER"
else
    check_warn "nvcc not in PATH (may be bundled with HPC SDK)"
fi
echo ""

echo "=== 4. NVIDIA HPC SDK ==="
if command -v nvfortran &> /dev/null; then
    NVF_VER=$(nvfortran --version 2>&1 | head -1)
    check_pass "nvfortran: $NVF_VER"
else
    check_fail "nvfortran not found - install NVIDIA HPC SDK"
fi
echo ""

echo "=== 5. MPI ==="
if command -v mpirun &> /dev/null; then
    MPI_VER=$(mpirun --version 2>&1 | head -1)
    check_pass "MPI: $MPI_VER"
    
    # Check CUDA awareness
    if ompi_info 2>/dev/null | grep -q "MPI extensions: cuda"; then
        check_pass "MPI has CUDA support"
    else
        check_warn "MPI may not have CUDA support (OK for single GPU)"
    fi
else
    check_warn "MPI not found (needed for multi-process runs)"
fi
echo ""

echo "=== 6. Python (for testing/analysis) ==="
if command -v python3 &> /dev/null; then
    PY_VER=$(python3 --version)
    check_pass "$PY_VER"
    
    # Check epoch-gpu package
    if python3 -c "import epoch_gpu" 2>/dev/null; then
        EPOCH_VER=$(python3 -c "import epoch_gpu; print(epoch_gpu.__version__)")
        check_pass "epoch-gpu package: v$EPOCH_VER"
    else
        check_warn "epoch-gpu package not installed (pip install epoch-gpu)"
    fi
else
    check_warn "Python3 not found"
fi
echo ""

echo "=== 7. Environment Variables ==="
if [ -n "$NVHPC" ]; then
    check_pass "NVHPC=$NVHPC"
else
    check_warn "NVHPC not set"
fi

if [ -n "$CUDA_HOME" ] || [ -n "$CUDA_PATH" ]; then
    check_pass "CUDA_HOME=${CUDA_HOME:-$CUDA_PATH}"
else
    check_warn "CUDA_HOME not set (may be OK if CUDA is in standard path)"
fi
echo ""

echo "=== 8. Compile Test ==="
TEMP_DIR=$(mktemp -d)
cat > $TEMP_DIR/test_gpu.f90 << 'EOF'
program test_gpu
  implicit none
  integer :: ndevices
  !$acc set device_num(0)
  ndevices = 1
  print *, "GPU OpenACC test passed!"
end program
EOF

cd $TEMP_DIR
if nvfortran -acc -gpu=managed test_gpu.f90 -o test_gpu 2>/dev/null; then
    if ./test_gpu 2>/dev/null | grep -q "passed"; then
        check_pass "OpenACC compilation and execution"
    else
        check_fail "OpenACC execution failed"
    fi
else
    check_fail "OpenACC compilation failed"
fi
rm -rf $TEMP_DIR
echo ""

echo "=============================================="
echo "  Verification Complete"
echo "=============================================="
```

Run with:

```bash
chmod +x check_epoch_gpu_env.sh
./check_epoch_gpu_env.sh
```

---

## Troubleshooting Common Issues

### Issue: "nvfortran: command not found"

**Cause**: HPC SDK not in PATH

**Solution**:
```bash
# Find installation
find /opt -name "nvfortran" 2>/dev/null

# Add to PATH (adjust version as needed)
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/bin:$PATH
```

### Issue: "CUDA driver version is insufficient"

**Cause**: Driver too old for CUDA version

**Solution**:
```bash
# Check versions
nvidia-smi  # Shows driver version
nvcc --version  # Shows CUDA version

# Update driver
sudo apt install nvidia-driver-535
sudo reboot
```

### Issue: "No CUDA-capable device detected"

**Cause**: GPU not accessible

**Solutions**:
```bash
# Check if GPU is visible
lspci | grep -i nvidia

# Check permissions
ls -la /dev/nvidia*

# Add user to video group
sudo usermod -a -G video $USER

# Logout and login again
```

### Issue: "out of memory" during compilation

**Cause**: Large code generation

**Solution**:
```bash
# Use less optimization
make -f Makefile.gpu GPU_CC=cc70 OPTIMIZATION=-O1

# Or compile fewer files at once
make -f Makefile.gpu -j1
```

### Issue: Module file not found

**Cause**: Compilation order issue

**Solution**:
```bash
# Clean and rebuild
make -f Makefile.gpu clean
make -f Makefile.gpu GPU_CC=cc70
```

---

## Quick Start Checklist

- [ ] NVIDIA GPU with Compute Capability ≥ 7.0
- [ ] NVIDIA driver installed (`nvidia-smi` works)
- [ ] NVIDIA HPC SDK installed (`nvfortran --version` works)
- [ ] MPI available (`mpirun --version` works)
- [ ] Environment variables set in `.bashrc`
- [ ] Verification script passes

Once all checks pass, proceed to [Building EPOCH-GPU]({{< relref "building.md" >}}).
