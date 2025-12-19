"""
EPOCH-GPU: GPU-accelerated Particle-In-Cell code for plasma physics simulations.

This package provides Python tools for:
- Reading and analyzing EPOCH simulation output (SDF files)
- Building and configuring EPOCH-GPU
- Running GPU-accelerated simulations
- Visualizing simulation results
"""

__version__ = "1.0.0"
__author__ = "EPOCH-GPU Development Team"
__license__ = "GPL-3.0"

from .config import GPUConfig, get_gpu_info
from .builder import build_epoch, clean_build

__all__ = [
    "__version__",
    "GPUConfig",
    "get_gpu_info",
    "build_epoch",
    "clean_build",
]

