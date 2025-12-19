#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2024 EPOCH-GPU Development Team
#
# GPU Laser Tests for EPOCH3D
# Based on original test_laser.py

"""
GPU Laser Propagation Tests for EPOCH3D

Tests 3D laser injection and propagation on GPU.
"""

import unittest
import subprocess
import os
import numpy as np


# Physical constants
c = 2.99792458e8
epsilon_0 = 8.854e-12
mu_0 = 4 * np.pi * 1e-7
micron = 1e-6
femto = 1e-15


class TestGPU3DLaser(unittest.TestCase):
    """Test 3D laser propagation on GPU."""
    
    @classmethod
    def setUpClass(cls):
        cls.gpu_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'gpu')
        cls.test_program = os.path.join(cls.gpu_dir, 'test_gpu_3d')
    
    def test_gaussian_beam_parameters(self):
        """Calculate Gaussian beam parameters."""
        lambda_l = 0.8 * micron  # 800 nm
        w0 = 5 * micron  # beam waist
        
        # Rayleigh length
        z_R = np.pi * w0**2 / lambda_l
        
        # Divergence angle
        theta = lambda_l / (np.pi * w0)
        
        # Intensity ratio at z = z_R
        I_ratio = 0.5  # I(z_R) / I(0)
        
        print(f"\nGaussian beam parameters:")
        print(f"  Wavelength: {lambda_l*1e9:.0f} nm")
        print(f"  Waist: {w0*1e6:.1f} μm")
        print(f"  Rayleigh length: {z_R*1e6:.1f} μm")
        print(f"  Divergence: {theta*1e3:.2f} mrad")
    
    def test_laser_intensity(self):
        """Calculate laser intensity from field amplitude."""
        # E0 in V/m
        E0 = 1e11  # typical laser field
        
        # Intensity I = (1/2) * epsilon_0 * c * E0^2
        I = 0.5 * epsilon_0 * c * E0**2
        
        # Normalize to I_18 (10^18 W/cm^2)
        I_18 = I / 1e18 / 1e4  # convert from W/m^2 to W/cm^2 / 10^18
        
        # a0 parameter
        lambda_l = 0.8 * micron
        omega_l = 2 * np.pi * c / lambda_l
        m_e = 9.109e-31
        q_e = 1.602e-19
        
        a0 = q_e * E0 / (m_e * omega_l * c)
        
        print(f"\nLaser intensity calculation:")
        print(f"  E0 = {E0:.2e} V/m")
        print(f"  I = {I:.2e} W/m²")
        print(f"  I_18 = {I_18:.2f}")
        print(f"  a0 = {a0:.2f}")
    
    def test_propagation(self):
        """Test laser propagation on GPU."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertEqual(result.returncode, 0)


class TestGPU3DLaserPerformance(unittest.TestCase):
    """Performance tests for 3D laser simulation."""
    
    def test_photon_budget(self):
        """Estimate photon budget for simulation."""
        # Typical PW laser pulse
        E_laser = 30.0  # Joules
        tau_laser = 30 * femto  # pulse duration
        P_laser = E_laser / tau_laser  # Peak power
        
        lambda_l = 0.8 * micron
        E_photon = 6.626e-34 * c / lambda_l  # photon energy
        
        N_photons = E_laser / E_photon
        
        print(f"\nLaser photon budget:")
        print(f"  Pulse energy: {E_laser:.0f} J")
        print(f"  Pulse duration: {tau_laser*1e15:.0f} fs")
        print(f"  Peak power: {P_laser/1e15:.1f} PW")
        print(f"  Photon count: {N_photons:.2e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)

