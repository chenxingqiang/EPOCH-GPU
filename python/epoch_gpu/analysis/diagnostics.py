"""
Diagnostic calculations for EPOCH-GPU simulations.

Provides functions for calculating common physical quantities
from simulation data.
"""

import numpy as np
from typing import Optional, Dict, Any, Union

# Physical constants
EPSILON_0 = 8.854187817e-12  # F/m, permittivity of free space
MU_0 = 1.2566370614e-6       # H/m, permeability of free space
C = 299792458.0              # m/s, speed of light
E_CHARGE = 1.602176634e-19   # C, elementary charge
E_MASS = 9.1093837015e-31    # kg, electron mass
PROTON_MASS = 1.67262192e-27 # kg, proton mass


def calculate_field_energy(
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    dx: float,
    dy: float = 1.0,
    dz: float = 1.0,
) -> Dict[str, float]:
    """
    Calculate electromagnetic field energy.
    
    Args:
        ex, ey, ez: Electric field components (V/m)
        bx, by, bz: Magnetic field components (T)
        dx, dy, dz: Grid spacing (m)
        
    Returns:
        Dictionary with 'electric', 'magnetic', and 'total' energy in Joules
    """
    # Calculate cell volume
    dv = dx * dy * dz
    
    # Electric field energy density: (1/2) * epsilon_0 * E^2
    e_squared = ex**2 + ey**2 + ez**2
    e_energy = 0.5 * EPSILON_0 * np.sum(e_squared) * dv
    
    # Magnetic field energy density: (1/2) * B^2 / mu_0
    b_squared = bx**2 + by**2 + bz**2
    b_energy = 0.5 * np.sum(b_squared) / MU_0 * dv
    
    return {
        'electric': e_energy,
        'magnetic': b_energy,
        'total': e_energy + b_energy,
    }


def calculate_particle_energy(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    weight: np.ndarray,
    mass: float,
) -> float:
    """
    Calculate total kinetic energy of particles.
    
    Uses relativistic formula: E = sqrt(p^2*c^2 + m^2*c^4) - m*c^2
    
    Args:
        px, py, pz: Particle momenta (kg*m/s)
        weight: Particle weights (number of real particles per macro-particle)
        mass: Particle mass (kg)
        
    Returns:
        Total kinetic energy in Joules
    """
    # Calculate p^2 for each particle
    p_squared = px**2 + py**2 + pz**2
    
    # Relativistic energy: E = sqrt(p^2*c^2 + m^2*c^4)
    rest_energy = mass * C**2
    total_energy = np.sqrt(p_squared * C**2 + rest_energy**2)
    
    # Kinetic energy = total - rest mass energy
    kinetic = total_energy - rest_energy
    
    # Sum weighted kinetic energy
    return np.sum(weight * kinetic)


def calculate_energy(
    fields: Optional[Dict[str, np.ndarray]] = None,
    particles: Optional[Dict[str, np.ndarray]] = None,
    grid: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Calculate total energy in the system.
    
    Args:
        fields: Dictionary with 'ex', 'ey', 'ez', 'bx', 'by', 'bz'
        particles: Dictionary with 'px', 'py', 'pz', 'weight', 'mass' for each species
        grid: Dictionary with 'dx', 'dy', 'dz' grid spacing
        
    Returns:
        Dictionary with energy components and total
    """
    result = {
        'field_electric': 0.0,
        'field_magnetic': 0.0,
        'particle_kinetic': 0.0,
        'total': 0.0,
    }
    
    # Field energy
    if fields and grid:
        dx = grid.get('dx', 1.0)
        dy = grid.get('dy', 1.0)
        dz = grid.get('dz', 1.0)
        
        field_energy = calculate_field_energy(
            fields.get('ex', np.zeros(1)),
            fields.get('ey', np.zeros(1)),
            fields.get('ez', np.zeros(1)),
            fields.get('bx', np.zeros(1)),
            fields.get('by', np.zeros(1)),
            fields.get('bz', np.zeros(1)),
            dx, dy, dz
        )
        result['field_electric'] = field_energy['electric']
        result['field_magnetic'] = field_energy['magnetic']
    
    # Particle energy
    if particles:
        for species_name, species_data in particles.items():
            if isinstance(species_data, dict):
                ke = calculate_particle_energy(
                    species_data.get('px', np.zeros(1)),
                    species_data.get('py', np.zeros(1)),
                    species_data.get('pz', np.zeros(1)),
                    species_data.get('weight', np.ones(1)),
                    species_data.get('mass', E_MASS),
                )
                result['particle_kinetic'] += ke
                result[f'particle_{species_name}'] = ke
    
    result['total'] = (
        result['field_electric'] + 
        result['field_magnetic'] + 
        result['particle_kinetic']
    )
    
    return result


def calculate_temperature(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    weight: np.ndarray,
    mass: float,
) -> Dict[str, float]:
    """
    Calculate temperature from particle momentum distribution.
    
    Uses T = <p^2> / (3 * m * k_B) for non-relativistic particles.
    
    Args:
        px, py, pz: Particle momenta (kg*m/s)
        weight: Particle weights
        mass: Particle mass (kg)
        
    Returns:
        Dictionary with 'Tx', 'Ty', 'Tz', 'T' temperatures in Kelvin
    """
    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    
    # Weight sum
    total_weight = np.sum(weight)
    if total_weight == 0:
        return {'Tx': 0.0, 'Ty': 0.0, 'Tz': 0.0, 'T': 0.0}
    
    # Mean momentum (should be ~0 for thermal distribution)
    px_mean = np.sum(weight * px) / total_weight
    py_mean = np.sum(weight * py) / total_weight
    pz_mean = np.sum(weight * pz) / total_weight
    
    # Variance of momentum
    px_var = np.sum(weight * (px - px_mean)**2) / total_weight
    py_var = np.sum(weight * (py - py_mean)**2) / total_weight
    pz_var = np.sum(weight * (pz - pz_mean)**2) / total_weight
    
    # Temperature: T = <p^2> / (m * k_B)
    # For each direction: T_i = <p_i^2> / (m * k_B)
    Tx = px_var / (mass * k_B)
    Ty = py_var / (mass * k_B)
    Tz = pz_var / (mass * k_B)
    
    # Average temperature
    T = (Tx + Ty + Tz) / 3.0
    
    return {'Tx': Tx, 'Ty': Ty, 'Tz': Tz, 'T': T}

