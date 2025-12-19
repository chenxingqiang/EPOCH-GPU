---
title: Troubleshooting
linktitle: Troubleshooting
toc: true
type: docs
date: "2024-12-19T00:00:00Z"
draft: false
menu:
  gpu:
    parent: GPU Guide
    weight: 7
---

# Troubleshooting GPU Issues

## Compilation Errors

### "nvfortran: command not found"

**Cause**: NVIDIA HPC SDK not in PATH.

**Solution**:
```bash
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/lib:$LD_LIBRARY_PATH
```

### "NVFORTRAN-S-0034-Syntax error at or near device"

**Cause**: Using CUDA Fortran syntax with OpenACC.

**Solution**: Replace `DEVICE` attribute with `ALLOCATABLE` and use OpenACC data directives.

### "Unable to read MODULE file xxx.mod"

**Cause**: Module compilation order incorrect.

**Solution**: Compile modules in dependency order:
```bash
nvfortran -c gpu_config.F90
nvfortran -c particle_data_gpu.F90
nvfortran -c fields_gpu.F90
nvfortran -c particles_gpu.F90
```

### "Unrecognized target architecture"

**Cause**: Invalid GPU compute capability.

**Solution**: Check your GPU and use correct flag:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
# Use the result, e.g., for V100 (7.0): GPU_CC=cc70
```

## Runtime Errors

### "data in PRESENT clause was not found on device"

**Cause**: Data not allocated on GPU before kernel launch.

**Solution**: Ensure data is on device:
```fortran
!$ACC ENTER DATA CREATE(array)
! ... use array in kernels ...
!$ACC EXIT DATA DELETE(array)
```

### "out of memory"

**Cause**: GPU memory exhausted.

**Solutions**:
1. Reduce grid size or particle count
2. Use single precision: `make -f Makefile.gpu DEFINE="-DSINGLE"`
3. Check for memory leaks with `cuda-memcheck`

### "illegal memory access"

**Cause**: Array bounds violation on GPU.

**Debug**:
```bash
cuda-memcheck ./epoch2d_gpu
```

### Simulation hangs

**Cause**: Often MPI-related or infinite loop.

**Debug**:
```bash
# Check GPU status
nvidia-smi

# Enable verbose output
export NVCOMPILER_ACC_NOTIFY=3
```

## Performance Issues

### Low GPU utilization

**Symptoms**: `nvidia-smi` shows <50% GPU utilization.

**Solutions**:
1. Increase particle count (>100K)
2. Increase grid size
3. Reduce I/O frequency

### Slow current deposition

**Symptoms**: Current deposition takes >50% of runtime.

**Cause**: Atomic operation contention.

**Mitigation**:
- This is expected behavior for PIC codes
- Consider histogram-based deposition (future feature)

### Excessive data transfers

**Symptoms**: Performance worse than CPU.

**Debug**:
```bash
export NVCOMPILER_ACC_TIME=1
./epoch2d_gpu
```

**Solution**: Keep data on device, minimize `UPDATE HOST/DEVICE`.

## Validation Errors

### Results differ from CPU version

**Expected**: Small differences due to floating-point order of operations.

**Acceptable**: Relative difference < 1e-10 (double precision).

**If larger differences**:
1. Check array indexing
2. Verify boundary conditions
3. Compare intermediate values

### NaN or Inf in output

**Cause**: Numerical instability or uninitialized memory.

**Debug**:
```fortran
! Add checks
IF (ISNAN(ex(ix,iy))) THEN
  WRITE(*,*) 'NaN detected at', ix, iy
  STOP
END IF
```

## Getting Help

1. **GitHub Issues**: [EPOCH-GPU Issues](https://github.com/chenxingqiang/EPOCH-GPU/issues)
2. **Original EPOCH**: [Warwick-Plasma/epoch](https://github.com/Warwick-Plasma/epoch)
3. **NVIDIA Forums**: [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

When reporting issues, include:
- GPU model and driver version (`nvidia-smi`)
- NVIDIA HPC SDK version (`nvfortran --version`)
- Complete error message
- Minimal reproducing input deck

