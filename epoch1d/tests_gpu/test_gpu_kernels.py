#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2025 EPOCH-GPU Development Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
GPU Kernel Unit Tests for EPOCH1D

Tests individual GPU kernels for correctness and performance.
"""

import unittest
import subprocess
import os
import re


class TestGPU1DKernels(unittest.TestCase):
    """Test 1D GPU kernel implementations."""
    
    @classmethod
    def setUpClass(cls):
        cls.gpu_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'gpu')
        cls.test_program = os.path.join(cls.gpu_dir, 'test_gpu_1d')
    
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
    
    def test_1d_field_solver(self):
        """Test 1D FDTD field solver."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertIn('1D Field Update GPU', result.stdout)
        self.assertEqual(result.returncode, 0)
    
    def test_1d_particle_pusher(self):
        """Test 1D Boris particle pusher."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertIn('1D Particle Push GPU', result.stdout)
        
        match = re.search(r'Particles/sec:\s+([\d.E+-]+)', result.stdout)
        if match:
            particles_per_sec = float(match.group(1))
            print(f"\n1D Particle pusher: {particles_per_sec:.2e} particles/sec")
            self.assertGreater(particles_per_sec, 1e9)
    
    def test_1d_field_interpolation(self):
        """Test 1D field interpolation (linear)."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertIn('1D Field Interpolation GPU', result.stdout)
    
    def test_1d_current_deposition(self):
        """Test 1D current deposition with atomics."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertIn('1D Current Deposition GPU', result.stdout)
    
    def test_1d_overall_pass(self):
        """Test all 1D GPU tests pass."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('EPOCH1D GPU Test PASSED', result.stdout)


class TestGPU1DPerformance(unittest.TestCase):
    """Performance benchmarks for 1D GPU kernels."""
    
    def test_1d_performance_metrics(self):
        """Extract and display 1D performance metrics."""
        gpu_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'gpu')
        test_program = os.path.join(gpu_dir, 'test_gpu_1d')
        
        if not os.path.exists(test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [test_program],
            capture_output=True,
            text=True,
            cwd=gpu_dir
        )
        
        print("\n" + "="*50)
        print("EPOCH1D GPU Performance Metrics (V100)")
        print("="*50)
        
        # Field update
        match = re.search(r'Test 1:.*?Per iteration:\s+([\d.E+-]+)', 
                         result.stdout, re.DOTALL)
        if match:
            print(f"  Field update: {float(match.group(1)):.4f} ms/iter")
        
        # Particle push
        match = re.search(r'Particles/sec:\s+([\d.E+-]+)', result.stdout)
        if match:
            print(f"  Particles/sec: {float(match.group(1)):.2e}")
        
        print("="*50)


class TestGPU1DTwoStream(unittest.TestCase):
    """Test two-stream instability simulation on GPU."""
    
    def test_twostream_physics(self):
        """Verify two-stream instability parameters."""
        import numpy as np
        
        # Two-stream parameters
        v_beam = 0.1 * 2.99792458e8  # 10% of c
        n_e = 1e20  # m^-3
        
        # Plasma frequency
        epsilon_0 = 8.854e-12
        m_e = 9.109e-31
        q_e = 1.602e-19
        
        omega_p = np.sqrt(n_e * q_e**2 / (epsilon_0 * m_e))
        
        # Growth rate (approximate)
        gamma_max = omega_p * (v_beam / (2.99792458e8)) ** (1/3)
        
        print(f"\nTwo-stream instability parameters:")
        print(f"  Beam velocity: {v_beam/1e6:.1f} km/s")
        print(f"  Plasma frequency: {omega_p:.3e} rad/s")
        print(f"  Est. growth rate: {gamma_max:.3e} /s")


if __name__ == '__main__':
    unittest.main(verbosity=2)

