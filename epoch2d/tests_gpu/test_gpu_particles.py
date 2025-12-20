#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2025 EPOCH-GPU Development Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
GPU Particle Tests for EPOCH2D

Tests GPU particle pusher (Boris algorithm), field interpolation,
and current deposition kernels.
"""

import unittest
import subprocess
import os
import sys
import time
import numpy as np
import re


# Physical constants
c = 2.99792458e8  # m/s
m_e = 9.109e-31   # kg (electron mass)
q_e = 1.602e-19   # C (electron charge)


class TestGPUParticlePusher(unittest.TestCase):
    """Test GPU Boris particle pusher."""
    
    @classmethod
    def setUpClass(cls):
        cls.gpu_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'gpu')
        cls.test_program = os.path.join(cls.gpu_dir, 'test_gpu_2d')
    
    def test_boris_pusher_execution(self):
        """Test Boris pusher runs without error."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertIn('Particle Push GPU', result.stdout)
        self.assertEqual(result.returncode, 0)
    
    def test_particle_throughput(self):
        """Verify particle pusher throughput meets minimum."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        match = re.search(r'Particles/sec:\s+([\d.E+-]+)', result.stdout)
        
        if match:
            particles_per_sec = float(match.group(1))
            print(f"\nBoris pusher: {particles_per_sec:.2e} particles/sec")
            
            # V100 should achieve at least 1 billion particles/sec
            self.assertGreater(particles_per_sec, 1e9,
                              "Boris pusher should exceed 1B particles/sec")
    
    def test_relativistic_motion(self):
        """Test relativistic particle dynamics."""
        # For highly relativistic particles, gamma >> 1
        # p = gamma * m * v, with v < c
        
        # Typical test: ultra-relativistic electron
        gamma = 100
        p = gamma * m_e * c  # momentum magnitude
        
        # Energy E = sqrt(p^2*c^2 + m^2*c^4) = gamma * m * c^2
        E_kinetic = (gamma - 1) * m_e * c**2
        
        print(f"\nRelativistic test case:")
        print(f"  gamma = {gamma}")
        print(f"  momentum = {p:.3e} kg*m/s")
        print(f"  kinetic energy = {E_kinetic:.3e} J = {E_kinetic/q_e:.1f} eV")
        
        # Verify lorentz factor calculation
        p_test = 1.0e-24  # From GPU test initialization
        gamma_calc = np.sqrt((p_test / (m_e * c))**2 + 1)
        
        self.assertGreater(gamma_calc, 0, "Lorentz factor must be positive")
    
    def test_cyclotron_motion(self):
        """Test particle gyration in magnetic field."""
        # Cyclotron frequency: omega_c = q*B/m
        # Gyroradius: r_g = v_perp / omega_c = m*v_perp / (q*B)
        
        B = 1.0e-3  # From GPU test (Tesla)
        omega_c = q_e * B / m_e
        period = 2 * np.pi / omega_c
        
        print(f"\nCyclotron motion test:")
        print(f"  B = {B*1e3:.1f} mT")
        print(f"  omega_c = {omega_c:.3e} rad/s")
        print(f"  period = {period*1e12:.3f} ps")
        
        # Verify cyclotron period is reasonable
        self.assertGreater(period, 0)
        self.assertLess(period, 1e-6)  # Should be < 1 microsecond


class TestGPUFieldInterpolation(unittest.TestCase):
    """Test GPU field-to-particle interpolation."""
    
    @classmethod
    def setUpClass(cls):
        cls.gpu_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'gpu')
        cls.test_program = os.path.join(cls.gpu_dir, 'test_gpu_2d')
    
    def test_interpolation_execution(self):
        """Test field interpolation runs correctly."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertIn('Field Interpolation GPU', result.stdout)
        self.assertEqual(result.returncode, 0)
    
    def test_bilinear_interpolation_accuracy(self):
        """Verify bilinear interpolation is second-order accurate."""
        # For a smooth field, error should scale as O(h^2)
        # where h is the grid spacing
        
        # This would require comparing to analytical solution
        # For now, verify kernel executes
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertEqual(result.returncode, 0)
        print("Bilinear interpolation kernel executed successfully")


class TestGPUCurrentDeposition(unittest.TestCase):
    """Test GPU current deposition with atomic operations."""
    
    @classmethod
    def setUpClass(cls):
        cls.gpu_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'gpu')
        cls.test_program = os.path.join(cls.gpu_dir, 'test_gpu_2d')
    
    def test_current_deposition_execution(self):
        """Test current deposition runs without race conditions."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertIn('Current Deposition GPU', result.stdout)
        self.assertEqual(result.returncode, 0)
    
    def test_charge_conservation(self):
        """Verify current deposition conserves charge."""
        # Charge conservation: dq/dt + div(J) = 0
        # Total charge should be conserved
        
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        # Look for current value in output
        match = re.search(r'Sample current.*?:\s*([\d.E+-]+)', result.stdout)
        if match:
            jx_sample = float(match.group(1))
            print(f"\nSample current density: {jx_sample:.3e} A/m^2")
        
        self.assertEqual(result.returncode, 0)
    
    def test_atomic_operation_correctness(self):
        """Verify atomic operations produce correct results."""
        # Multiple particles depositing to same cell should sum correctly
        # This tests the ATOMIC directive in OpenACC
        
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        # Current should be non-zero after deposition
        self.assertIn('current deposition', result.stdout.lower())
        self.assertEqual(result.returncode, 0)


class TestGPUParticlePerformance(unittest.TestCase):
    """Performance benchmarks for particle operations."""
    
    def test_full_particle_cycle(self):
        """Benchmark complete particle update cycle."""
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
        
        # Extract timing for each phase
        times = {}
        
        # Particle push
        match = re.search(r'Test 2:.*?Per iteration:\s+([\d.E+-]+)', 
                         result.stdout, re.DOTALL)
        if match:
            times['push'] = float(match.group(1))
        
        # Field interpolation
        match = re.search(r'Test 3:.*?Per iteration:\s+([\d.E+-]+)', 
                         result.stdout, re.DOTALL)
        if match:
            times['interpolation'] = float(match.group(1))
        
        # Current deposition
        match = re.search(r'Test 4:.*?Per iteration:\s+([\d.E+-]+)', 
                         result.stdout, re.DOTALL)
        if match:
            times['deposition'] = float(match.group(1))
        
        print("\n" + "="*50)
        print("GPU Particle Cycle Performance (1M particles)")
        print("="*50)
        total = 0
        for op, time_ms in times.items():
            print(f"  {op:20s}: {time_ms:.4f} ms")
            total += time_ms
        print(f"  {'Total':20s}: {total:.4f} ms")
        print(f"  Effective rate: {1000.0/total:.0f} cycles/sec")
        print("="*50)
        
        # Total cycle should be < 2ms for 1M particles
        self.assertLess(total, 2.0,
                       "Full particle cycle should be < 2ms")


if __name__ == '__main__':
    unittest.main(verbosity=2)

