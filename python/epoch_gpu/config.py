"""
GPU configuration and device management for EPOCH-GPU.
"""

import os
import subprocess
import shutil
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class GPUInfo:
    """Information about an NVIDIA GPU device."""
    device_id: int
    name: str
    memory_total: int  # bytes
    memory_free: int   # bytes
    compute_capability: str
    cuda_version: str


class GPUConfig:
    """
    GPU configuration manager for EPOCH-GPU.
    
    Provides utilities for detecting and configuring NVIDIA GPUs
    for EPOCH-GPU simulations.
    """
    
    def __init__(self):
        self._gpus: List[GPUInfo] = []
        self._nvidia_smi_path: Optional[str] = None
        self._nvfortran_path: Optional[str] = None
        self._detect_environment()
    
    def _detect_environment(self) -> None:
        """Detect GPU environment and tools."""
        # Find nvidia-smi
        self._nvidia_smi_path = shutil.which("nvidia-smi")
        
        # Find nvfortran
        self._nvfortran_path = shutil.which("nvfortran")
        if not self._nvfortran_path:
            # Check common HPC SDK locations
            hpc_sdk_paths = [
                "/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/bin/nvfortran",
                "/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/nvfortran",
                "/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/bin/nvfortran",
            ]
            for path in hpc_sdk_paths:
                if os.path.exists(path):
                    self._nvfortran_path = path
                    break
    
    def has_gpu(self) -> bool:
        """Check if NVIDIA GPU is available."""
        return self._nvidia_smi_path is not None and self._query_gpu_count() > 0
    
    def has_compiler(self) -> bool:
        """Check if nvfortran compiler is available."""
        return self._nvfortran_path is not None
    
    def _query_gpu_count(self) -> int:
        """Query number of available GPUs."""
        if not self._nvidia_smi_path:
            return 0
        try:
            result = subprocess.run(
                [self._nvidia_smi_path, "--query-gpu=count", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return int(result.stdout.strip().split('\n')[0])
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            pass
        return 0
    
    def get_gpus(self, refresh: bool = False) -> List[GPUInfo]:
        """
        Get list of available GPUs.
        
        Args:
            refresh: Force refresh of GPU information
            
        Returns:
            List of GPUInfo objects
        """
        if self._gpus and not refresh:
            return self._gpus
        
        self._gpus = []
        if not self._nvidia_smi_path:
            return self._gpus
        
        try:
            # Query GPU information
            result = subprocess.run(
                [
                    self._nvidia_smi_path,
                    "--query-gpu=index,name,memory.total,memory.free,compute_cap",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Get CUDA version
                cuda_version = self._get_cuda_version()
                
                for line in result.stdout.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        gpu = GPUInfo(
                            device_id=int(parts[0]),
                            name=parts[1],
                            memory_total=int(parts[2]) * 1024 * 1024,  # MiB to bytes
                            memory_free=int(parts[3]) * 1024 * 1024,
                            compute_capability=parts[4],
                            cuda_version=cuda_version
                        )
                        self._gpus.append(gpu)
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            pass
        
        return self._gpus
    
    def _get_cuda_version(self) -> str:
        """Get CUDA driver version."""
        if not self._nvidia_smi_path:
            return "unknown"
        try:
            result = subprocess.run(
                [self._nvidia_smi_path, "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            pass
        return "unknown"
    
    def get_recommended_gpu(self) -> Optional[GPUInfo]:
        """
        Get recommended GPU for simulation.
        
        Returns the GPU with most free memory.
        """
        gpus = self.get_gpus()
        if not gpus:
            return None
        return max(gpus, key=lambda g: g.memory_free)
    
    def print_info(self) -> None:
        """Print GPU configuration information."""
        print("=" * 60)
        print("EPOCH-GPU Configuration")
        print("=" * 60)
        
        print(f"\nNVIDIA Tools:")
        print(f"  nvidia-smi: {self._nvidia_smi_path or 'Not found'}")
        print(f"  nvfortran:  {self._nvfortran_path or 'Not found'}")
        
        gpus = self.get_gpus()
        if gpus:
            print(f"\nAvailable GPUs ({len(gpus)}):")
            for gpu in gpus:
                print(f"  [{gpu.device_id}] {gpu.name}")
                print(f"      Memory: {gpu.memory_free // (1024**2)} MiB free / "
                      f"{gpu.memory_total // (1024**2)} MiB total")
                print(f"      Compute Capability: {gpu.compute_capability}")
                print(f"      CUDA Driver: {gpu.cuda_version}")
        else:
            print("\nNo NVIDIA GPUs detected")
        
        print("=" * 60)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        gpus = self.get_gpus()
        return {
            "nvidia_smi": self._nvidia_smi_path,
            "nvfortran": self._nvfortran_path,
            "gpu_count": len(gpus),
            "gpus": [
                {
                    "id": g.device_id,
                    "name": g.name,
                    "memory_total_mb": g.memory_total // (1024**2),
                    "memory_free_mb": g.memory_free // (1024**2),
                    "compute_capability": g.compute_capability,
                    "cuda_version": g.cuda_version,
                }
                for g in gpus
            ]
        }


def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information as a dictionary.
    
    Convenience function for quick GPU status check.
    
    Returns:
        Dictionary with GPU configuration details
    """
    config = GPUConfig()
    return config.to_dict()

