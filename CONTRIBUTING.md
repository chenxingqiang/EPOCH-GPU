# Contributing to EPOCH-GPU

EPOCH-GPU is a GPU-accelerated fork of the EPOCH Particle-In-Cell code.
We welcome contributions that improve GPU performance, add new GPU features,
or enhance compatibility with the original EPOCH codebase.

## Project Links

- **EPOCH-GPU Repository**: https://github.com/chenxingqiang/EPOCH-GPU
- **Original EPOCH**: https://github.com/Warwick-Plasma/epoch
- **Issue Tracker**: https://github.com/chenxingqiang/EPOCH-GPU/issues

## Types of Contributions

### GPU-Specific Contributions
- Performance optimizations for existing GPU kernels
- New GPU kernel implementations
- Multi-GPU (MPI + GPU) improvements
- Support for new GPU architectures
- GPU memory optimization

### General Contributions
- Bug fixes
- Documentation improvements
- Test coverage improvements
- Compatibility fixes with upstream EPOCH

## How to Contribute

1. **Fork the repository** on GitHub
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make your changes** following the coding style guidelines
4. **Test your changes** using the GPU test framework
5. **Commit with clear messages**: `git commit -am 'Add GPU optimization for XYZ'`
6. **Push to your fork**: `git push origin feature/your-feature`
7. **Submit a Pull Request** against the `main` branch

### GPU Code Guidelines

For GPU-specific code in `epoch*/src/gpu/`:
- Use OpenACC directives for portability where possible
- Use CUDA Fortran only when OpenACC is insufficient
- Include performance benchmarks for new kernels
- Maintain compatibility with compute capability 7.0+

```fortran
! Good: OpenACC with explicit data management
!$ACC PARALLEL LOOP PRESENT(array) PRIVATE(temp)
DO i = 1, n
  ! kernel code
END DO
!$ACC END PARALLEL LOOP
```

## Licensing

By submitting your changes, you agree that they will be licensed under
GPLv3, the same license as EPOCH and EPOCH-GPU.

---

# Original EPOCH Contributions

For contributions to the core EPOCH algorithms (non-GPU), please also consider
contributing to the upstream repository: https://github.com/Warwick-Plasma/epoch

## Coding Style

This codebase conforms to a single, consistent coding style which is
outlined in the CODING_STYLE document in the root directory.
New submissions will be rejected if they do not conform to this style.

In particular, please note that all code must conform to the Fortran F95
standard (ISO/IEC 1539-1: 1997), and keep all lines, including comments, to
less than 80 characters long.

Please also follow the standard commit-message format:
- First line: subject, <50 characters
- Second line: blank
- Subsequent lines: message body, <72 characters

## Physics Packages

EPOCH is designed to be modular and extensible. Adding physics capability
should be done via a physics package in the physics_packages subdirectory.
Conditional compilation (#ifdef) should be used only if absolutely necessary.
Where possible, all control should be via the input deck. Backwards
compatibility must be preserved.
