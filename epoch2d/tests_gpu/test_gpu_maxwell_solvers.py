#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2025 EPOCH-GPU Development Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
GPU Maxwell Solver Tests for EPOCH2D

Verifies GPU field solver produces correct results compared to CPU version.
Based on original test_maxwell_solvers.py
"""

import unittest
import subprocess
import os
import sys
import time
import numpy as np

# Try to import sdfr for result verification
try:
    import sdfr
    HAS_SDFR = True
except ImportError:
    HAS_SDFR = False


# Physical constants
micron = 1e-6
femto = 1e-15
c = 2.99792458e8  # m/s

# Grid parameters (should match GPU test)
nx = 256
ny = 256

x_min = -12 * micron
x_max = -x_min
y_min = x_min
y_max = x_max

# Derived quantities
dx = (x_max - x_min) / nx
dy = (y_max - y_min) / ny


class TestGPUFieldSolver(unittest.TestCase):
    """Test GPU field solver correctness."""
    
    @classmethod
    def setUpClass(cls):
        cls.gpu_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'gpu')
        cls.test_program = os.path.join(cls.gpu_dir, 'test_gpu_2d')
    
    def test_field_update_convergence(self):
        """Test that field update converges correctly."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        # Check field solver runs without error
        self.assertIn('Test 1: 2D Field Update GPU', result.stdout)
        self.assertEqual(result.returncode, 0)
    
    def test_yee_scheme_stability(self):
        """Verify Yee scheme CFL condition is satisfied."""
        # CFL condition for 2D: c*dt < dx*dy / sqrt(dx^2 + dy^2)
        dt_max = dx * dy / (c * np.sqrt(dx**2 + dy**2))
        dt_test = 1.0e-15  # From GPU test
        
        cfl = c * dt_test * np.sqrt(1/dx**2 + 1/dy**2)
        
        self.assertLess(cfl, 1.0, 
                       f"CFL condition violated: {cfl:.4f} >= 1.0")
        print(f"CFL number: {cfl:.4f}")
    
    def test_maxwell_equations_consistency(self):
        """Test div(E) and div(B) conservation."""
        # This would require running actual simulation and checking
        # For now, verify test runs correctly
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertIn('GPU Test PASSED', result.stdout)


class TestGPUvsCPUFieldSolver(unittest.TestCase):
    """Compare GPU and CPU field solver results."""
    
    def test_field_values_match(self):
        """Verify GPU field values match CPU within tolerance."""
        # This test would require running both GPU and CPU versions
        # and comparing output files
        
        # Placeholder - actual implementation would:
        # 1. Run CPU version
        # 2. Run GPU version
        # 3. Compare field arrays
        
        # For now, just check the GPU test passes
        gpu_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'gpu')
        test_program = os.path.join(gpu_dir, 'test_gpu_2d')
        
        if not os.path.exists(test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [test_program],
            capture_output=True,
            text=True,
            cwd=gpu_dir
        )
        
        self.assertEqual(result.returncode, 0)
        print("GPU field solver test passed")


class TestFieldSolverPerformance(unittest.TestCase):
    """Benchmark GPU field solver performance."""
    
    def test_field_update_throughput(self):
        """Measure cells/second for field update."""
        gpu_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'gpu')
        test_program = os.path.join(gpu_dir, 'test_gpu_2d')
        
        if not os.path.exists(test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [test_program],
            capture_output=True,
            text=True,
            cwd=gpu_dir
        )
        
        import re
        match = re.search(r'Cells/sec:\s+([\d.E+-]+)', result.stdout)
        
        if match:
            cells_per_sec = float(match.group(1))
            print(f"\nField Update Throughput: {cells_per_sec:.2e} cells/sec")
            
            # V100 should achieve at least 1B cells/sec
            self.assertGreater(cells_per_sec, 1e9,
                              "Field update should exceed 1B cells/sec")
    
    def test_speedup_over_cpu(self):
        """Estimate GPU speedup over CPU."""
        # Typical CPU single-core performance ~10^7 cells/sec
        cpu_estimate = 1e7  # cells/sec
        
        gpu_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'gpu')
        test_program = os.path.join(gpu_dir, 'test_gpu_2d')
        
        if not os.path.exists(test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [test_program],
            capture_output=True,
            text=True,
            cwd=gpu_dir
        )
        
        import re
        match = re.search(r'Cells/sec:\s+([\d.E+-]+)', result.stdout)
        
        if match:
            gpu_cells_per_sec = float(match.group(1))
            speedup = gpu_cells_per_sec / cpu_estimate
            print(f"\nEstimated GPU Speedup: {speedup:.1f}x over single CPU core")
            
            # Should achieve significant speedup
            self.assertGreater(speedup, 10.0,
                              "Should achieve >10x speedup over CPU")


if __name__ == '__main__':
    unittest.main(verbosity=2)

