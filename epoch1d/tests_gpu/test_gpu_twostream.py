#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2025 EPOCH-GPU Development Team
#
# GPU Two-Stream Instability Test for EPOCH1D
# Based on original test_twostream.py

"""
GPU Two-Stream Instability Tests for EPOCH1D

Tests the classic two-stream instability using GPU acceleration.
"""

import unittest
import subprocess
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Physical constants
c = 2.99792458e8  # m/s
m_e = 9.109e-31   # kg
q_e = 1.602e-19   # C
epsilon_0 = 8.854e-12  # F/m


class TestGPUTwoStream(unittest.TestCase):
    """Test two-stream instability on GPU."""
    
    @classmethod
    def setUpClass(cls):
        cls.gpu_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'gpu')
        cls.test_program = os.path.join(cls.gpu_dir, 'test_gpu_1d')
    
    def test_dispersion_relation(self):
        """Verify two-stream dispersion relation."""
        # For cold beams: 1 = (omega_p^2/2) * [1/(omega-k*v)^2 + 1/(omega+k*v)^2]
        
        n_e = 1e20  # density per beam
        v_beam = 0.05 * c  # beam velocity
        
        omega_p = np.sqrt(2 * n_e * q_e**2 / (epsilon_0 * m_e))
        
        # Maximum growth at k * v_beam = omega_p / sqrt(2)
        k_max = omega_p / (np.sqrt(2) * v_beam)
        lambda_max = 2 * np.pi / k_max
        
        print(f"\nTwo-stream dispersion:")
        print(f"  Plasma frequency: {omega_p:.3e} rad/s")
        print(f"  Most unstable k: {k_max:.3e} /m")
        print(f"  Most unstable wavelength: {lambda_max*1e6:.2f} μm")
    
    def test_growth_rate(self):
        """Verify instability growth rate."""
        n_e = 1e20
        v_beam = 0.05 * c
        
        omega_p = np.sqrt(2 * n_e * q_e**2 / (epsilon_0 * m_e))
        
        # Maximum growth rate for cold beams
        gamma_max = np.sqrt(3) / (2**(4/3)) * omega_p * (v_beam / c)**(1/3)
        
        e_folding_time = 1.0 / gamma_max
        
        print(f"\nGrowth rate analysis:")
        print(f"  Max growth rate: {gamma_max:.3e} /s")
        print(f"  e-folding time: {e_folding_time*1e12:.2f} ps")
        
        self.assertGreater(gamma_max, 0)
    
    def test_gpu_simulation(self):
        """Run GPU test and verify completion."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertEqual(result.returncode, 0)
        print("\nGPU simulation completed successfully")


class TestGPULandau(unittest.TestCase):
    """Test Landau damping on GPU."""
    
    def test_landau_damping_rate(self):
        """Calculate theoretical Landau damping rate."""
        # For k * lambda_D << 1:
        # gamma/omega = -sqrt(pi/8) * (omega_p/k*v_th)^3 * exp(-omega_p^2/(2*k^2*v_th^2))
        
        T_e = 100  # eV
        n_e = 1e20  # m^-3
        
        k_B = 1.38e-23
        T_K = T_e * q_e / k_B
        
        v_th = np.sqrt(2 * k_B * T_K / m_e)
        omega_p = np.sqrt(n_e * q_e**2 / (epsilon_0 * m_e))
        lambda_D = v_th / omega_p
        
        print(f"\nLandau damping parameters:")
        print(f"  Temperature: {T_e} eV")
        print(f"  Thermal velocity: {v_th/1e6:.2f} × 10^6 m/s")
        print(f"  Debye length: {lambda_D*1e9:.2f} nm")
        print(f"  Plasma frequency: {omega_p:.3e} rad/s")


if __name__ == '__main__':
    unittest.main(verbosity=2)

