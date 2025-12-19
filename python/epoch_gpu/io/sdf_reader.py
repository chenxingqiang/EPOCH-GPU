"""
SDF file reader for EPOCH-GPU output.

This module provides a high-level interface for reading EPOCH SDF output files.
It wraps the low-level sdf module from SDF/utilities.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

# Try to import sdf module
_sdf_available = False
try:
    import sdf
    _sdf_available = True
except ImportError:
    pass

# Also try sdf_helper
_sdf_helper_available = False
try:
    import sdf_helper
    _sdf_helper_available = True
except ImportError:
    pass


def _check_sdf_available():
    """Check if SDF module is available, raise helpful error if not."""
    if not _sdf_available:
        raise ImportError(
            "The 'sdf' module is not installed. To read SDF files, you need to:\n"
            "1. Build the SDF C library:\n"
            "   cd SDF/C && make\n"
            "2. Install the Python module:\n"
            "   cd SDF/utilities && pip install -e .\n"
        )


@dataclass
class SDFVariable:
    """Container for a single SDF variable."""
    name: str
    data: Any
    units: str
    dims: tuple
    grid: Optional[str] = None
    
    def __repr__(self):
        return f"SDFVariable(name='{self.name}', shape={self.data.shape}, units='{self.units}')"


class SDFFile:
    """
    High-level interface for reading EPOCH SDF output files.
    
    Example usage:
        >>> sdf = SDFFile("0001.sdf")
        >>> print(sdf.variables)  # List all variables
        >>> ex = sdf["Electric Field/Ex"]  # Get variable data
        >>> sdf.close()
        
        # Or using context manager:
        >>> with SDFFile("0001.sdf") as sdf:
        ...     ex = sdf["Electric Field/Ex"]
    """
    
    def __init__(self, filename: Union[str, Path]):
        """
        Open an SDF file for reading.
        
        Args:
            filename: Path to SDF file
        """
        _check_sdf_available()
        
        self.filename = Path(filename)
        if not self.filename.exists():
            raise FileNotFoundError(f"SDF file not found: {self.filename}")
        
        self._data = sdf.read(str(self.filename))
        self._variables: Dict[str, Any] = {}
        self._parse_contents()
    
    def _parse_contents(self):
        """Parse SDF file contents into variable dictionary."""
        for name in dir(self._data):
            if name.startswith('_'):
                continue
            attr = getattr(self._data, name)
            if hasattr(attr, 'data'):
                self._variables[name] = attr
    
    @property
    def variables(self) -> List[str]:
        """List of variable names in the file."""
        return list(self._variables.keys())
    
    @property
    def header(self) -> Dict[str, Any]:
        """SDF file header information."""
        return {
            'filename': str(self.filename),
            'step': getattr(self._data, 'Header', {}).get('step', 0),
            'time': getattr(self._data, 'Header', {}).get('time', 0.0),
            'code_name': getattr(self._data, 'Header', {}).get('code_name', 'EPOCH'),
        }
    
    def __getitem__(self, name: str) -> SDFVariable:
        """
        Get a variable by name.
        
        Args:
            name: Variable name (supports partial matching)
            
        Returns:
            SDFVariable object containing the data
        """
        # Try exact match first
        if name in self._variables:
            var = self._variables[name]
            return SDFVariable(
                name=name,
                data=var.data,
                units=getattr(var, 'units', ''),
                dims=var.data.shape,
                grid=getattr(var, 'grid', None),
            )
        
        # Try partial match
        matches = [k for k in self._variables if name.lower() in k.lower()]
        if len(matches) == 1:
            return self[matches[0]]
        elif len(matches) > 1:
            raise KeyError(f"Ambiguous variable name '{name}'. Matches: {matches}")
        
        raise KeyError(f"Variable '{name}' not found. Available: {self.variables}")
    
    def get(self, name: str, default: Any = None) -> Optional[SDFVariable]:
        """
        Get a variable by name, returning default if not found.
        
        Args:
            name: Variable name
            default: Default value if not found
            
        Returns:
            SDFVariable or default
        """
        try:
            return self[name]
        except KeyError:
            return default
    
    def close(self):
        """Close the SDF file and release resources."""
        self._data = None
        self._variables = {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def __repr__(self):
        return f"SDFFile('{self.filename}', variables={len(self._variables)})"


def read_sdf(filename: Union[str, Path]) -> SDFFile:
    """
    Read an SDF file.
    
    Convenience function that creates an SDFFile object.
    
    Args:
        filename: Path to SDF file
        
    Returns:
        SDFFile object
    """
    return SDFFile(filename)


def list_sdf_files(directory: Union[str, Path] = ".") -> List[Path]:
    """
    List all SDF files in a directory.
    
    Args:
        directory: Directory to search (default: current directory)
        
    Returns:
        List of SDF file paths, sorted by name
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    sdf_files = list(directory.glob("*.sdf"))
    return sorted(sdf_files)

