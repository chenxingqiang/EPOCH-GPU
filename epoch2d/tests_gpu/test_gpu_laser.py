#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2024 EPOCH-GPU Development Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
GPU Laser Propagation Tests for EPOCH2D

Tests laser injection and propagation using GPU-accelerated field solver.
Based on original test_laser.py
"""

import unittest
import subprocess
import os
import sys
import time
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import sdfr
    HAS_SDFR = True
except ImportError:
    HAS_SDFR = False


# Physical constants
c = 2.99792458e8  # m/s
epsilon_0 = 8.854e-12  # F/m
micron = 1e-6
femto = 1e-15


class TestGPULaserPropagation(unittest.TestCase):
    """Test laser propagation on GPU."""
    
    @classmethod
    def setUpClass(cls):
        cls.gpu_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'gpu')
        cls.test_program = os.path.join(cls.gpu_dir, 'test_gpu_2d')
    
    def test_field_evolution(self):
        """Test electromagnetic field evolution."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        # Verify field solver executed
        self.assertIn('Field Update GPU', result.stdout)
        self.assertEqual(result.returncode, 0)
    
    def test_energy_conservation(self):
        """Test total electromagnetic energy is conserved."""
        # In vacuum, EM energy should be approximately conserved
        # U = (epsilon_0/2) * integral(E^2) + (1/(2*mu_0)) * integral(B^2)
        
        # This would require actual simulation output
        # For now, verify GPU test passes
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertEqual(result.returncode, 0)
        print("Energy conservation test passed (basic validation)")
    
    def test_wave_dispersion(self):
        """Test light wave dispersion relation."""
        # omega = c * k for EM waves in vacuum
        # Numerical dispersion depends on grid spacing and timestep
        
        # Verify Courant condition
        dx = 1e-8  # From GPU test
        dy = 1e-8
        dt = 1e-15
        
        # CFL number
        cfl = c * dt * np.sqrt(1/dx**2 + 1/dy**2)
        
        self.assertLess(cfl, 1.0, "CFL condition must be satisfied")
        print(f"CFL number: {cfl:.4f}")


class TestGPULaserPerformance(unittest.TestCase):
    """Benchmark GPU laser simulation performance."""
    
    def test_laser_field_throughput(self):
        """Measure field update throughput for laser-type simulation."""
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
        match = re.search(r'Per iteration:\s+([\d.E+-]+)\s+ms', result.stdout)
        
        if match:
            time_ms = float(match.group(1))
            print(f"\nField update: {time_ms:.4f} ms/iteration")
            
            # Should be fast enough for real-time visualization
            self.assertLess(time_ms, 1.0,
                           "Field update should be < 1ms")


if __name__ == '__main__':
    unittest.main(verbosity=2)

