#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2024 EPOCH-GPU Development Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
GPU Kernel Unit Tests for EPOCH2D

Tests individual GPU kernels for correctness and performance.
Run compiled Fortran test programs and verify results.
"""

import unittest
import subprocess
import os
import sys
import time
import re


class TestGPUKernels(unittest.TestCase):
    """Test GPU kernel implementations."""
    
    @classmethod
    def setUpClass(cls):
        """Compile GPU test program if needed."""
        cls.gpu_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'gpu')
        cls.test_program = os.path.join(cls.gpu_dir, 'test_gpu_2d')
        
        # Check if test program exists
        if not os.path.exists(cls.test_program):
            print("Compiling GPU test program...")
            cls._compile_test_program()
    
    @classmethod
    def _compile_test_program(cls):
        """Compile the GPU test program."""
        compile_cmd = '''
        cd {gpu_dir} && \
        nvfortran -O3 -r8 -acc -gpu=cc70 -cuda -DGPU_ENABLED \
            -o test_gpu_2d \
            gpu_config.o particle_data_gpu.o fields_gpu.o particles_gpu.o \
            test_gpu_2d.F90
        '''.format(gpu_dir=cls.gpu_dir)
        
        result = subprocess.run(compile_cmd, shell=True, capture_output=True)
        if result.returncode != 0:
            print("Compilation failed:", result.stderr.decode())
    
    def test_gpu_available(self):
        """Test that GPU is available and detected."""
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertIn('GPU Acceleration: ENABLED', result.stdout,
                      "GPU should be detected and enabled")
        self.assertIn('Tesla', result.stdout,
                      "Should detect NVIDIA GPU")
    
    def test_field_solver(self):
        """Test 2D field solver GPU kernel."""
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertIn('Test 1: 2D Field Update GPU', result.stdout)
        self.assertIn('field updates:', result.stdout)
        
        # Extract timing
        match = re.search(r'Per iteration:\s+([\d.E+-]+)\s+ms', result.stdout)
        if match:
            time_per_iter = float(match.group(1))
            print(f"Field solver: {time_per_iter:.4f} ms/iteration")
            # Should complete in reasonable time
            self.assertLess(time_per_iter, 1.0, 
                           "Field update should be < 1ms per iteration")
    
    def test_particle_pusher(self):
        """Test Boris particle pusher GPU kernel."""
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertIn('Test 2: 2D Particle Push GPU', result.stdout)
        
        # Extract particles/sec
        match = re.search(r'Particles/sec:\s+([\d.E+-]+)', result.stdout)
        if match:
            particles_per_sec = float(match.group(1))
            print(f"Particle pusher: {particles_per_sec:.2e} particles/sec")
            # Should achieve at least 1 billion particles/sec on GPU
            self.assertGreater(particles_per_sec, 1e9,
                              "Should push > 1B particles/sec on GPU")
    
    def test_field_interpolation(self):
        """Test field interpolation GPU kernel."""
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertIn('Test 3: 2D Field Interpolation GPU', result.stdout)
        self.assertIn('interpolation iterations:', result.stdout)
    
    def test_current_deposition(self):
        """Test current deposition GPU kernel with atomics."""
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertIn('Test 4: 2D Current Deposition GPU', result.stdout)
        self.assertIn('current deposition iterations:', result.stdout)
    
    def test_overall_pass(self):
        """Test that all GPU tests pass."""
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        self.assertEqual(result.returncode, 0,
                        "GPU test program should exit successfully")
        self.assertIn('GPU Test PASSED', result.stdout,
                     "All GPU tests should pass")


class TestGPUPerformance(unittest.TestCase):
    """Performance benchmark tests for GPU kernels."""
    
    @classmethod
    def setUpClass(cls):
        cls.gpu_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'gpu')
        cls.test_program = os.path.join(cls.gpu_dir, 'test_gpu_2d')
    
    def test_performance_metrics(self):
        """Extract and validate performance metrics."""
        if not os.path.exists(self.test_program):
            self.skipTest("GPU test program not compiled")
        
        result = subprocess.run(
            [self.test_program],
            capture_output=True,
            text=True,
            cwd=self.gpu_dir
        )
        
        metrics = {}
        
        # Extract field update time
        match = re.search(r'Test 1:.*?Per iteration:\s+([\d.E+-]+)', 
                         result.stdout, re.DOTALL)
        if match:
            metrics['field_update_ms'] = float(match.group(1))
        
        # Extract particle push throughput
        match = re.search(r'Particles/sec:\s+([\d.E+-]+)', result.stdout)
        if match:
            metrics['particles_per_sec'] = float(match.group(1))
        
        # Extract cells/sec
        match = re.search(r'Cells/sec:\s+([\d.E+-]+)', result.stdout)
        if match:
            metrics['cells_per_sec'] = float(match.group(1))
        
        print("\n" + "="*50)
        print("GPU Performance Metrics (V100)")
        print("="*50)
        for key, value in metrics.items():
            if 'per_sec' in key:
                print(f"  {key}: {value:.2e}")
            else:
                print(f"  {key}: {value:.4f}")
        print("="*50)
        
        # Validate minimum performance
        if 'particles_per_sec' in metrics:
            self.assertGreater(metrics['particles_per_sec'], 1e9,
                              "Particle throughput should exceed 1B/sec")


if __name__ == '__main__':
    unittest.main(verbosity=2)

