"""
Unit conversion and plasma physics calculations.
"""

import numpy as np
from typing import Union

# Physical constants in SI
C = 299792458.0              # m/s, speed of light
EPSILON_0 = 8.854187817e-12  # F/m, permittivity of free space
MU_0 = 1.2566370614e-6       # H/m, permeability of free space
E_CHARGE = 1.602176634e-19   # C, elementary charge
E_MASS = 9.1093837015e-31    # kg, electron mass
PROTON_MASS = 1.67262192e-27 # kg, proton mass
K_B = 1.380649e-23           # J/K, Boltzmann constant


def si_to_cgs(value: float, quantity: str) -> float:
    """
    Convert SI units to CGS (Gaussian) units.
    
    Args:
        value: Value in SI units
        quantity: Type of quantity ('length', 'mass', 'time', 'charge', 
                  'E_field', 'B_field', 'energy')
    
    Returns:
        Value in CGS units
    """
    conversions = {
        'length': 100.0,           # m -> cm
        'mass': 1000.0,            # kg -> g
        'time': 1.0,               # s -> s
        'charge': 2997924580.0,    # C -> statcoulombs
        'E_field': 1.0 / 299.792,  # V/m -> statV/cm
        'B_field': 10000.0,        # T -> Gauss
        'energy': 1e7,             # J -> erg
    }
    
    if quantity not in conversions:
        raise ValueError(f"Unknown quantity: {quantity}")
    
    return value * conversions[quantity]


def cgs_to_si(value: float, quantity: str) -> float:
    """
    Convert CGS (Gaussian) units to SI units.
    
    Args:
        value: Value in CGS units
        quantity: Type of quantity
    
    Returns:
        Value in SI units
    """
    return value / si_to_cgs(1.0, quantity)


def plasma_frequency(
    n_e: Union[float, np.ndarray],
    mass: float = E_MASS,
    charge: float = E_CHARGE,
) -> Union[float, np.ndarray]:
    """
    Calculate plasma frequency.
    
    omega_p = sqrt(n_e * e^2 / (epsilon_0 * m))
    
    Args:
        n_e: Electron density (m^-3)
        mass: Particle mass (kg), default electron
        charge: Particle charge (C), default electron
    
    Returns:
        Plasma frequency (rad/s)
    """
    return np.sqrt(n_e * charge**2 / (EPSILON_0 * mass))


def debye_length(
    n_e: Union[float, np.ndarray],
    T_e: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate Debye length.
    
    lambda_D = sqrt(epsilon_0 * k_B * T_e / (n_e * e^2))
    
    Args:
        n_e: Electron density (m^-3)
        T_e: Electron temperature (K)
    
    Returns:
        Debye length (m)
    """
    return np.sqrt(EPSILON_0 * K_B * T_e / (n_e * E_CHARGE**2))


def skin_depth(
    n_e: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate collisionless skin depth (electron inertial length).
    
    c / omega_p
    
    Args:
        n_e: Electron density (m^-3)
    
    Returns:
        Skin depth (m)
    """
    omega_p = plasma_frequency(n_e)
    return C / omega_p


def cyclotron_frequency(
    B: Union[float, np.ndarray],
    mass: float = E_MASS,
    charge: float = E_CHARGE,
) -> Union[float, np.ndarray]:
    """
    Calculate cyclotron (gyro) frequency.
    
    omega_c = |q| * B / m
    
    Args:
        B: Magnetic field strength (T)
        mass: Particle mass (kg), default electron
        charge: Particle charge magnitude (C), default electron
    
    Returns:
        Cyclotron frequency (rad/s)
    """
    return np.abs(charge) * B / mass


def larmor_radius(
    v_perp: Union[float, np.ndarray],
    B: Union[float, np.ndarray],
    mass: float = E_MASS,
    charge: float = E_CHARGE,
) -> Union[float, np.ndarray]:
    """
    Calculate Larmor (gyro) radius.
    
    r_L = m * v_perp / (|q| * B)
    
    Args:
        v_perp: Perpendicular velocity (m/s)
        B: Magnetic field strength (T)
        mass: Particle mass (kg)
        charge: Particle charge magnitude (C)
    
    Returns:
        Larmor radius (m)
    """
    return mass * v_perp / (np.abs(charge) * B)


def thermal_velocity(
    T: Union[float, np.ndarray],
    mass: float = E_MASS,
) -> Union[float, np.ndarray]:
    """
    Calculate thermal velocity.
    
    v_th = sqrt(k_B * T / m)
    
    Args:
        T: Temperature (K)
        mass: Particle mass (kg)
    
    Returns:
        Thermal velocity (m/s)
    """
    return np.sqrt(K_B * T / mass)

