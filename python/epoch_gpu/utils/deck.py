"""
EPOCH input deck file parser and writer.

Provides utilities for reading and writing EPOCH .deck files.
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
from collections import OrderedDict


class DeckFile:
    """
    Parser and writer for EPOCH input deck files.
    
    EPOCH deck files use a simple block-based format:
    
        begin:block_name
            key = value
            key = value
        end:block_name
    
    Example usage:
        >>> deck = DeckFile("input.deck")
        >>> deck['control']['nx'] = 1024
        >>> deck.write("output.deck")
    """
    
    def __init__(self, filename: Optional[Union[str, Path]] = None):
        """
        Create a DeckFile, optionally loading from file.
        
        Args:
            filename: Path to deck file to load (optional)
        """
        self._blocks: Dict[str, OrderedDict] = OrderedDict()
        self._comments: List[str] = []
        self.filename: Optional[Path] = None
        
        if filename:
            self.load(filename)
    
    def load(self, filename: Union[str, Path]) -> None:
        """
        Load deck file from disk.
        
        Args:
            filename: Path to deck file
        """
        self.filename = Path(filename)
        if not self.filename.exists():
            raise FileNotFoundError(f"Deck file not found: {self.filename}")
        
        self._blocks = OrderedDict()
        self._comments = []
        
        current_block = None
        
        with open(self.filename, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Store comments
                if line.startswith('#'):
                    self._comments.append(line)
                    continue
                
                # Check for block start
                match = re.match(r'^begin:(\w+)$', line, re.IGNORECASE)
                if match:
                    current_block = match.group(1).lower()
                    if current_block not in self._blocks:
                        self._blocks[current_block] = OrderedDict()
                    continue
                
                # Check for block end
                match = re.match(r'^end:(\w+)$', line, re.IGNORECASE)
                if match:
                    current_block = None
                    continue
                
                # Parse key = value
                if current_block and '=' in line:
                    # Handle inline comments
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to convert value to appropriate type
                    value = self._parse_value(value)
                    
                    self._blocks[current_block][key] = value
    
    def _parse_value(self, value: str) -> Any:
        """Parse a string value to appropriate Python type."""
        # Remove quotes
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        
        # Boolean
        if value.lower() in ('t', 'true', '.true.'):
            return True
        if value.lower() in ('f', 'false', '.false.'):
            return False
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float (handle Fortran notation)
        try:
            value_float = value.replace('d', 'e').replace('D', 'E')
            return float(value_float)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _format_value(self, value: Any) -> str:
        """Format a Python value for deck file."""
        if isinstance(value, bool):
            return 'T' if value else 'F'
        elif isinstance(value, float):
            # Use scientific notation for very small/large numbers
            if abs(value) < 1e-3 or abs(value) > 1e6:
                return f'{value:.6e}'
            return str(value)
        elif isinstance(value, str):
            # Quote strings with spaces
            if ' ' in value or ',' in value:
                return f'"{value}"'
            return value
        return str(value)
    
    @property
    def blocks(self) -> List[str]:
        """List of block names in the deck."""
        return list(self._blocks.keys())
    
    def __getitem__(self, block_name: str) -> OrderedDict:
        """Get a block by name."""
        block_name = block_name.lower()
        if block_name not in self._blocks:
            self._blocks[block_name] = OrderedDict()
        return self._blocks[block_name]
    
    def __setitem__(self, block_name: str, values: Dict) -> None:
        """Set a block's values."""
        self._blocks[block_name.lower()] = OrderedDict(values)
    
    def __contains__(self, block_name: str) -> bool:
        """Check if block exists."""
        return block_name.lower() in self._blocks
    
    def get(self, block_name: str, key: str, default: Any = None) -> Any:
        """
        Get a value from a block.
        
        Args:
            block_name: Name of the block
            key: Key within the block
            default: Default value if not found
            
        Returns:
            Value or default
        """
        block = self._blocks.get(block_name.lower(), {})
        return block.get(key, default)
    
    def set(self, block_name: str, key: str, value: Any) -> None:
        """
        Set a value in a block.
        
        Args:
            block_name: Name of the block
            key: Key within the block
            value: Value to set
        """
        block_name = block_name.lower()
        if block_name not in self._blocks:
            self._blocks[block_name] = OrderedDict()
        self._blocks[block_name][key] = value
    
    def write(self, filename: Optional[Union[str, Path]] = None) -> None:
        """
        Write deck file to disk.
        
        Args:
            filename: Output path (default: overwrite original)
        """
        if filename:
            self.filename = Path(filename)
        
        if not self.filename:
            raise ValueError("No filename specified")
        
        with open(self.filename, 'w') as f:
            # Write header comments
            for comment in self._comments:
                f.write(comment + '\n')
            if self._comments:
                f.write('\n')
            
            # Write blocks
            for block_name, block_data in self._blocks.items():
                f.write(f'begin:{block_name}\n')
                for key, value in block_data.items():
                    formatted = self._format_value(value)
                    f.write(f'    {key} = {formatted}\n')
                f.write(f'end:{block_name}\n\n')
    
    def to_dict(self) -> Dict[str, Dict]:
        """Export deck as nested dictionary."""
        return {k: dict(v) for k, v in self._blocks.items()}
    
    def __repr__(self):
        return f"DeckFile('{self.filename}', blocks={self.blocks})"


def read_deck(filename: Union[str, Path]) -> DeckFile:
    """
    Read an EPOCH deck file.
    
    Args:
        filename: Path to deck file
        
    Returns:
        DeckFile object
    """
    return DeckFile(filename)


def write_deck(
    filename: Union[str, Path],
    blocks: Dict[str, Dict],
    comments: Optional[List[str]] = None,
) -> None:
    """
    Write an EPOCH deck file from scratch.
    
    Args:
        filename: Output path
        blocks: Dictionary of block name -> key-value pairs
        comments: Optional header comments
    """
    deck = DeckFile()
    deck.filename = Path(filename)
    deck._comments = comments or []
    
    for block_name, block_data in blocks.items():
        deck[block_name] = block_data
    
    deck.write()

