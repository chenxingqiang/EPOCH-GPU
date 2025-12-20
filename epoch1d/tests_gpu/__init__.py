# Copyright (C) 2025 EPOCH-GPU Development Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
EPOCH-GPU Test Framework for 1D
GPU-accelerated version of EPOCH tests
"""

import unittest
import os
import subprocess
import time


class GPUSimTest(unittest.TestCase):
    """
    Base class for GPU-accelerated EPOCH1D tests.
    """

    @classmethod
    def setUpClass(cls):
        simdir = cls.__name__[5:]
        os.chdir(os.path.join(os.path.dirname(__file__), simdir))
        
        cls.start_time = time.time()
        exitcode = subprocess.call('make gpu', shell=True)
        cls.epochexitcode = exitcode
        cls.elapsed_time = time.time() - cls.start_time
        
        if exitcode != 0:
            os.chdir('..')

    def setUp(self):
        if self.epochexitcode:
            self.fail('running EPOCH1D-GPU errored (exitcode {:})'
                      .format(self.epochexitcode))

    @classmethod
    def tearDownClass(cls):
        if "NOCLEAN" not in os.environ:
            subprocess.call('make clean', shell=True)
        os.chdir('..')
        print(f"\nGPU Test completed in {cls.elapsed_time:.2f} seconds")


class GPUKernelTest(unittest.TestCase):
    """Base class for GPU kernel unit tests."""
    
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.dirname(__file__)
        
    def assertGPUResult(self, gpu_result, cpu_result, rtol=1e-5, atol=1e-8):
        """Assert GPU and CPU results match within tolerance."""
        import numpy as np
        np.testing.assert_allclose(gpu_result, cpu_result, rtol=rtol, atol=atol,
                                   err_msg="GPU result differs from CPU reference")

