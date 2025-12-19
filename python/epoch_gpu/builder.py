"""
EPOCH-GPU build utilities.

Provides functions for building EPOCH-GPU from source.
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List, Dict


# Supported dimensions
DIMENSIONS = ["1d", "2d", "3d"]


def get_epoch_root() -> Path:
    """Get the EPOCH-GPU root directory."""
    # Try to find relative to this file
    current = Path(__file__).resolve()
    for _ in range(5):
        current = current.parent
        if (current / "epoch2d").exists() and (current / "README.md").exists():
            return current
    
    # Fall back to environment variable
    if "EPOCH_ROOT" in os.environ:
        return Path(os.environ["EPOCH_ROOT"])
    
    raise RuntimeError(
        "Cannot find EPOCH-GPU root directory. "
        "Set EPOCH_ROOT environment variable."
    )


def build_epoch(
    dimension: str = "2d",
    gpu: bool = True,
    compiler: Optional[str] = None,
    mpi: bool = True,
    parallel_jobs: int = 4,
    clean_first: bool = False,
    extra_flags: Optional[List[str]] = None,
) -> bool:
    """
    Build EPOCH-GPU for specified dimension.
    
    Args:
        dimension: Simulation dimension ("1d", "2d", or "3d")
        gpu: Build with GPU support (requires nvfortran)
        compiler: Override compiler (default: nvfortran for GPU, gfortran for CPU)
        mpi: Build with MPI support
        parallel_jobs: Number of parallel make jobs
        clean_first: Run 'make clean' before building
        extra_flags: Additional compiler flags
        
    Returns:
        True if build succeeded, False otherwise
    """
    dimension = dimension.lower()
    if dimension not in DIMENSIONS:
        raise ValueError(f"Invalid dimension: {dimension}. Must be one of {DIMENSIONS}")
    
    epoch_root = get_epoch_root()
    build_dir = epoch_root / f"epoch{dimension}"
    
    if not build_dir.exists():
        raise RuntimeError(f"Build directory not found: {build_dir}")
    
    # Determine makefile
    makefile = "Makefile.gpu" if gpu else "Makefile"
    makefile_path = build_dir / makefile
    
    if not makefile_path.exists():
        raise RuntimeError(f"Makefile not found: {makefile_path}")
    
    # Build environment
    env = os.environ.copy()
    
    # Set compiler if specified
    if compiler:
        env["FC"] = compiler
    
    # Clean if requested
    if clean_first:
        clean_cmd = ["make", "-f", makefile, "clean"]
        result = subprocess.run(
            clean_cmd,
            cwd=build_dir,
            env=env,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Warning: clean failed: {result.stderr}")
    
    # Build command
    build_cmd = ["make", "-f", makefile, f"-j{parallel_jobs}"]
    
    if extra_flags:
        build_cmd.extend(extra_flags)
    
    print(f"Building EPOCH-GPU {dimension.upper()}...")
    print(f"  Directory: {build_dir}")
    print(f"  Makefile: {makefile}")
    print(f"  GPU: {gpu}")
    print(f"  Command: {' '.join(build_cmd)}")
    print()
    
    result = subprocess.run(
        build_cmd,
        cwd=build_dir,
        env=env,
        capture_output=False,  # Show output in real-time
    )
    
    if result.returncode == 0:
        print(f"\n✓ Build successful!")
        executable = build_dir / "bin" / f"epoch{dimension}{'_gpu' if gpu else ''}"
        print(f"  Executable: {executable}")
        return True
    else:
        print(f"\n✗ Build failed!")
        return False


def clean_build(dimension: str = "all", gpu: bool = True) -> bool:
    """
    Clean build artifacts.
    
    Args:
        dimension: Dimension to clean ("1d", "2d", "3d", or "all")
        gpu: Clean GPU build artifacts
        
    Returns:
        True if clean succeeded
    """
    epoch_root = get_epoch_root()
    
    dims = DIMENSIONS if dimension == "all" else [dimension.lower()]
    
    success = True
    for dim in dims:
        if dim not in DIMENSIONS:
            print(f"Warning: Invalid dimension {dim}, skipping")
            continue
            
        build_dir = epoch_root / f"epoch{dim}"
        if not build_dir.exists():
            continue
        
        makefile = "Makefile.gpu" if gpu else "Makefile"
        makefile_path = build_dir / makefile
        
        if not makefile_path.exists():
            continue
        
        print(f"Cleaning epoch{dim}...")
        result = subprocess.run(
            ["make", "-f", makefile, "clean"],
            cwd=build_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"  Warning: {result.stderr}")
            success = False
        else:
            print(f"  ✓ Cleaned")
    
    return success


def get_build_info() -> Dict[str, any]:
    """
    Get information about available builds.
    
    Returns:
        Dictionary with build information for each dimension
    """
    epoch_root = get_epoch_root()
    
    info = {
        "epoch_root": str(epoch_root),
        "dimensions": {},
    }
    
    for dim in DIMENSIONS:
        dim_info = {
            "cpu_makefile": False,
            "gpu_makefile": False,
            "cpu_executable": False,
            "gpu_executable": False,
        }
        
        build_dir = epoch_root / f"epoch{dim}"
        if build_dir.exists():
            dim_info["cpu_makefile"] = (build_dir / "Makefile").exists()
            dim_info["gpu_makefile"] = (build_dir / "Makefile.gpu").exists()
            dim_info["cpu_executable"] = (build_dir / "bin" / f"epoch{dim}").exists()
            dim_info["gpu_executable"] = (build_dir / "bin" / f"epoch{dim}_gpu").exists()
        
        info["dimensions"][dim] = dim_info
    
    return info

