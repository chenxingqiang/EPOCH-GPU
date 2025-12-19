"""
Unit tests for epoch_gpu Python package.
"""

import unittest
import tempfile
import os
from pathlib import Path


class TestGPUConfig(unittest.TestCase):
    """Tests for GPU configuration module."""
    
    def test_import(self):
        """Test that the package imports correctly."""
        from epoch_gpu import __version__, GPUConfig
        self.assertEqual(__version__, "1.0.0")
        self.assertIsNotNone(GPUConfig)
    
    def test_gpu_config_creation(self):
        """Test GPUConfig object creation."""
        from epoch_gpu import GPUConfig
        config = GPUConfig()
        self.assertIsNotNone(config)
    
    def test_gpu_config_to_dict(self):
        """Test GPUConfig.to_dict() method."""
        from epoch_gpu import GPUConfig
        config = GPUConfig()
        info = config.to_dict()
        self.assertIsInstance(info, dict)
        self.assertIn('gpu_count', info)
        self.assertIn('gpus', info)
    
    def test_get_gpu_info(self):
        """Test get_gpu_info() function."""
        from epoch_gpu import get_gpu_info
        info = get_gpu_info()
        self.assertIsInstance(info, dict)


class TestUnits(unittest.TestCase):
    """Tests for unit conversion utilities."""
    
    def test_plasma_frequency(self):
        """Test plasma frequency calculation."""
        from epoch_gpu.utils.units import plasma_frequency, E_MASS, E_CHARGE, EPSILON_0
        import numpy as np
        
        # Test with known values
        n_e = 1e20  # 1e20 m^-3
        omega_p = plasma_frequency(n_e)
        
        # Expected: sqrt(n_e * e^2 / (epsilon_0 * m_e))
        expected = np.sqrt(n_e * E_CHARGE**2 / (EPSILON_0 * E_MASS))
        self.assertAlmostEqual(omega_p, expected, places=5)
    
    def test_debye_length(self):
        """Test Debye length calculation."""
        from epoch_gpu.utils.units import debye_length
        
        n_e = 1e20  # m^-3
        T_e = 1e6   # K (about 86 eV)
        
        lambda_d = debye_length(n_e, T_e)
        self.assertGreater(lambda_d, 0)
        self.assertLess(lambda_d, 1e-3)  # Should be sub-mm
    
    def test_unit_conversion(self):
        """Test SI to CGS conversion."""
        from epoch_gpu.utils.units import si_to_cgs, cgs_to_si
        
        # Test length conversion: 1 m = 100 cm
        self.assertAlmostEqual(si_to_cgs(1.0, 'length'), 100.0)
        self.assertAlmostEqual(cgs_to_si(100.0, 'length'), 1.0)
        
        # Test energy conversion: 1 J = 1e7 erg
        self.assertAlmostEqual(si_to_cgs(1.0, 'energy'), 1e7)


class TestDeckFile(unittest.TestCase):
    """Tests for deck file parser."""
    
    def test_deck_parsing(self):
        """Test parsing a deck file."""
        from epoch_gpu.utils.deck import DeckFile, write_deck
        
        # Create a temporary deck file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.deck', delete=False) as f:
            f.write("""
begin:control
    nx = 1024
    ny = 512
    t_end = 1.0e-12
    use_gpu = T
end:control

begin:species
    name = electron
    charge = -1.0
    mass = 1.0
end:species
""")
            temp_path = f.name
        
        try:
            # Parse the deck
            deck = DeckFile(temp_path)
            
            # Check blocks exist
            self.assertIn('control', deck.blocks)
            self.assertIn('species', deck.blocks)
            
            # Check values
            self.assertEqual(deck['control']['nx'], 1024)
            self.assertEqual(deck['control']['ny'], 512)
            self.assertEqual(deck['control']['t_end'], 1.0e-12)
            self.assertTrue(deck['control']['use_gpu'])
            
            self.assertEqual(deck['species']['name'], 'electron')
            self.assertEqual(deck['species']['charge'], -1.0)
        finally:
            os.unlink(temp_path)
    
    def test_deck_writing(self):
        """Test writing a deck file."""
        from epoch_gpu.utils.deck import DeckFile
        
        deck = DeckFile()
        deck['control'] = {'nx': 256, 'ny': 256, 't_end': 1e-10}
        deck['boundaries'] = {'bc_x_min': 'periodic', 'bc_x_max': 'periodic'}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.deck', delete=False) as f:
            temp_path = f.name
        
        try:
            deck.write(temp_path)
            
            # Read back
            deck2 = DeckFile(temp_path)
            self.assertEqual(deck2['control']['nx'], 256)
            self.assertEqual(deck2['boundaries']['bc_x_min'], 'periodic')
        finally:
            os.unlink(temp_path)


class TestDiagnostics(unittest.TestCase):
    """Tests for diagnostic calculations."""
    
    def test_field_energy(self):
        """Test field energy calculation."""
        from epoch_gpu.analysis.diagnostics import calculate_field_energy
        import numpy as np
        
        # Create uniform fields
        shape = (10, 10)
        ex = np.ones(shape) * 1e6  # 1 MV/m
        ey = np.zeros(shape)
        ez = np.zeros(shape)
        bx = np.zeros(shape)
        by = np.zeros(shape)
        bz = np.ones(shape) * 1e-3  # 1 mT
        
        energy = calculate_field_energy(ex, ey, ez, bx, by, bz, dx=1e-6, dy=1e-6)
        
        self.assertIn('electric', energy)
        self.assertIn('magnetic', energy)
        self.assertIn('total', energy)
        self.assertGreater(energy['electric'], 0)
        self.assertGreater(energy['magnetic'], 0)
    
    def test_particle_energy(self):
        """Test particle energy calculation."""
        from epoch_gpu.analysis.diagnostics import calculate_particle_energy, E_MASS
        import numpy as np
        
        # Create particles with known momentum
        n_part = 100
        px = np.ones(n_part) * 1e-24  # kg*m/s
        py = np.zeros(n_part)
        pz = np.zeros(n_part)
        weight = np.ones(n_part) * 1e10
        
        ke = calculate_particle_energy(px, py, pz, weight, E_MASS)
        self.assertGreater(ke, 0)
    
    def test_temperature(self):
        """Test temperature calculation."""
        from epoch_gpu.analysis.diagnostics import calculate_temperature, E_MASS
        from epoch_gpu.utils.units import K_B
        import numpy as np
        
        # Create thermal distribution
        np.random.seed(42)
        n_part = 10000
        T_expected = 1e6  # 1 MK
        v_th = np.sqrt(K_B * T_expected / E_MASS)
        
        # Generate Maxwell-Boltzmann momenta
        px = np.random.normal(0, E_MASS * v_th, n_part)
        py = np.random.normal(0, E_MASS * v_th, n_part)
        pz = np.random.normal(0, E_MASS * v_th, n_part)
        weight = np.ones(n_part)
        
        temp = calculate_temperature(px, py, pz, weight, E_MASS)
        
        # Should be close to expected temperature (within 10%)
        self.assertAlmostEqual(temp['T'] / T_expected, 1.0, delta=0.1)


class TestBuilder(unittest.TestCase):
    """Tests for build utilities."""
    
    def test_get_build_info(self):
        """Test getting build information."""
        from epoch_gpu.builder import get_build_info
        
        info = get_build_info()
        self.assertIn('epoch_root', info)
        self.assertIn('dimensions', info)
        self.assertIn('1d', info['dimensions'])
        self.assertIn('2d', info['dimensions'])
        self.assertIn('3d', info['dimensions'])


if __name__ == '__main__':
    unittest.main()

