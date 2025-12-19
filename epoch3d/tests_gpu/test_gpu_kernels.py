#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2024 EPOCH-GPU Development Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
GPU Kernel Unit Tests for EPOCH3D

Tests individual GPU kernels for 3D simulations.
"""

import unittest
import subprocess
import os
import re


class TestGPU3DKernels(unittest.TestCase):
    """Test 3D GPU kernel implementations."""
    
    @classmethod
    def setUpClass(cls):
        cls.gpu_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'gpu')
        cls.test_program = os.path.join(cls.gpu_dir, 'test_gpu_3d')
    
    def test_gpu_available(self):
        """Test that GPU is detected."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertIn('GPU Acceleration: ENABLED', result.stdout)
    
    def test_3d_field_solver(self):
        """Test 3D FDTD field solver."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertIn('3D Field Update GPU', result.stdout)
        self.assertEqual(result.returncode, 0)
        
        # Check throughput
        match = re.search(r'Cells/sec:\s+([\d.E+-]+)', result.stdout)
        if match:
            cells_per_sec = float(match.group(1))
            print(f"\n3D Field solver: {cells_per_sec:.2e} cells/sec")
            self.assertGreater(cells_per_sec, 1e9)
    
    def test_3d_particle_pusher(self):
        """Test 3D Boris particle pusher."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertIn('3D Particle Push GPU', result.stdout)
        
        match = re.search(r'Particles/sec:\s+([\d.E+-]+)', result.stdout)
        if match:
            particles_per_sec = float(match.group(1))
            print(f"\n3D Particle pusher: {particles_per_sec:.2e} particles/sec")
            self.assertGreater(particles_per_sec, 1e9)
    
    def test_3d_trilinear_interpolation(self):
        """Test 3D trilinear field interpolation."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertIn('3D Field Interpolation GPU', result.stdout)
    
    def test_3d_current_deposition(self):
        """Test 3D current deposition with atomics."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertIn('3D Current Deposition GPU', result.stdout)
    
    def test_3d_overall_pass(self):
        """Test all 3D GPU tests pass."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('EPOCH3D GPU Test PASSED', result.stdout)


class TestGPU3DPerformance(unittest.TestCase):
    """Performance benchmarks for 3D GPU kernels."""
    
    def test_3d_performance_metrics(self):
        """Extract and display 3D performance metrics."""
        gpu_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'gpu')
        test_program = os.path.join(gpu_dir, 'test_gpu_3d')
        
        if not os.path.exists(test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [test_program],
            capture_output=True,
            text=True,
            cwd=gpu_dir
        )
        
        print("\n" + "="*50)
        print("EPOCH3D GPU Performance Metrics (V100)")
        print("="*50)
        
        # Field update
        match = re.search(r'Test 1:.*?Per iteration:\s+([\d.E+-]+)', 
                         result.stdout, re.DOTALL)
        if match:
            print(f"  Field update: {float(match.group(1)):.4f} ms/iter")
        
        # Cells/sec
        match = re.search(r'Cells/sec:\s+([\d.E+-]+)', result.stdout)
        if match:
            print(f"  Cells/sec: {float(match.group(1)):.2e}")
        
        # Particles/sec
        match = re.search(r'Particles/sec:\s+([\d.E+-]+)', result.stdout)
        if match:
            print(f"  Particles/sec: {float(match.group(1)):.2e}")
        
        print("="*50)
    
    def test_3d_memory_requirements(self):
        """Estimate memory requirements for 3D simulation."""
        import numpy as np
        
        # Test grid size
        nx, ny, nz = 64, 64, 64
        ng = 2  # ghost cells
        
        # 6 field components (Ex, Ey, Ez, Bx, By, Bz)
        # 3 current components (Jx, Jy, Jz)
        n_field_arrays = 9
        field_mem = n_field_arrays * (nx + 2*ng) * (ny + 2*ng) * (nz + 2*ng) * 8  # bytes
        
        # Particles: 6 arrays (x, y, z, px, py, pz) + weight
        n_particles = 500000
        particle_mem = 7 * n_particles * 8  # bytes
        
        total_mem = field_mem + particle_mem
        
        print(f"\n3D Memory Requirements (64³ grid, 500K particles):")
        print(f"  Field arrays: {field_mem / 1e6:.1f} MB")
        print(f"  Particle arrays: {particle_mem / 1e6:.1f} MB")
        print(f"  Total: {total_mem / 1e6:.1f} MB")
        
        # V100 has 32 GB
        self.assertLess(total_mem, 32e9, "Should fit in V100 memory")


class TestGPU3DLaser(unittest.TestCase):
    """Test 3D laser propagation on GPU."""
    
    def test_laser_physics(self):
        """Verify laser parameters for 3D simulation."""
        import numpy as np
        
        c = 2.99792458e8
        
        # Typical laser parameters
        lambda_l = 0.8e-6  # 800 nm (Ti:Sapphire)
        omega_l = 2 * np.pi * c / lambda_l
        
        # Gaussian beam parameters
        w0 = 5e-6  # waist size
        z_R = np.pi * w0**2 / lambda_l  # Rayleigh length
        
        print(f"\n3D Laser parameters:")
        print(f"  Wavelength: {lambda_l*1e9:.0f} nm")
        print(f"  Frequency: {omega_l:.3e} rad/s")
        print(f"  Waist: {w0*1e6:.1f} μm")
        print(f"  Rayleigh length: {z_R*1e6:.1f} μm")
    
    def test_diffraction_limit(self):
        """Verify diffraction-limited focusing."""
        import numpy as np
        
        lambda_l = 0.8e-6
        f_number = 3.0  # f/#
        
        # Diffraction-limited spot size
        w0 = lambda_l * f_number / np.pi
        
        # Required grid resolution (at least 10 cells per wavelength)
        dx_max = lambda_l / 10
        
        print(f"\nDiffraction-limited focusing:")
        print(f"  f/# = {f_number}")
        print(f"  Spot size: {w0*1e6:.2f} μm")
        print(f"  Required dx: < {dx_max*1e9:.0f} nm")


if __name__ == '__main__':
    unittest.main(verbosity=2)

