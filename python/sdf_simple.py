#!/usr/bin/env python3
"""
Simple SDF file reader for EPOCH output.
Reads basic SDF file structure and extracts grid and field data.
"""

import struct
import numpy as np
from pathlib import Path


class SDFBlock:
    """Represents a single block in an SDF file."""
    def __init__(self):
        self.id = ""
        self.name = ""
        self.block_type = 0
        self.data_type = 0
        self.ndims = 0
        self.dims = []
        self.data = None


class SDFFile:
    """Simple SDF file reader."""
    
    # Block types
    BLOCKTYPE_SCRUBBED = -1
    BLOCKTYPE_NULL = 0
    BLOCKTYPE_PLAIN_MESH = 1
    BLOCKTYPE_POINT_MESH = 2
    BLOCKTYPE_PLAIN_VARIABLE = 3
    BLOCKTYPE_POINT_VARIABLE = 4
    BLOCKTYPE_CONSTANT = 5
    BLOCKTYPE_ARRAY = 6
    BLOCKTYPE_RUN_INFO = 7
    BLOCKTYPE_SOURCE = 8
    BLOCKTYPE_STITCHED_TENSOR = 9
    BLOCKTYPE_STITCHED_MATERIAL = 10
    BLOCKTYPE_STITCHED_MATVAR = 11
    BLOCKTYPE_STITCHED_SPECIES = 12
    BLOCKTYPE_SPECIES = 13
    BLOCKTYPE_PLAIN_DERIVED = 14
    BLOCKTYPE_POINT_DERIVED = 15
    
    # Data types
    DATATYPE_NULL = 0
    DATATYPE_INTEGER4 = 1
    DATATYPE_INTEGER8 = 2
    DATATYPE_REAL4 = 3
    DATATYPE_REAL8 = 4
    DATATYPE_REAL16 = 5
    DATATYPE_CHARACTER = 6
    DATATYPE_LOGICAL = 7
    
    def __init__(self, filename):
        self.filename = filename
        self.blocks = {}
        self.time = 0.0
        self.step = 0
        self._read()
    
    def _read(self):
        """Read the SDF file."""
        with open(self.filename, 'rb') as f:
            # Read header
            magic = f.read(4).decode('ascii')
            if magic != 'SDF1':
                raise ValueError(f"Not an SDF file: {self.filename}")
            
            # Read version info (4 bytes)
            endianness, version, revision, _ = struct.unpack('<BBBB', f.read(4))
            
            # Read more header info
            header_data = struct.unpack('<ii', f.read(8))
            
            # Read code name (32 bytes)
            self.code_name = f.read(32).decode('ascii').strip('\x00').strip()
            
            # Read first block location and other metadata
            first_block_location, summary_location, summary_size = struct.unpack('<qqq', f.read(24))
            nblocks, block_header_length = struct.unpack('<ii', f.read(8))
            step_data = struct.unpack('<i', f.read(4))
            self.step = step_data[0]
            time_data = struct.unpack('<d', f.read(8))
            self.time = time_data[0]
            
            # Skip to first block
            f.seek(first_block_location)
            
            # Read blocks
            for _ in range(nblocks):
                block = self._read_block_header(f, block_header_length)
                if block:
                    self.blocks[block.name] = block
    
    def _read_block_header(self, f, header_length):
        """Read a block header."""
        start_pos = f.tell()
        
        block = SDFBlock()
        
        # Read next block location and data location
        next_block_location, data_location, block.id = struct.unpack('<qq32s', f.read(48))
        block.id = block.id.decode('ascii').strip('\x00').strip()
        
        # Read data length and block type
        data_length, block.block_type, block.data_type, block.ndims = struct.unpack('<qiii', f.read(20))
        
        # Read block name
        block.name = f.read(32).decode('ascii').strip('\x00').strip()
        
        # Try to read data for simple variable blocks
        if block.block_type == self.BLOCKTYPE_PLAIN_VARIABLE:
            self._read_plain_variable(f, block, data_location, data_length, start_pos, header_length)
        elif block.block_type == self.BLOCKTYPE_CONSTANT:
            self._read_constant(f, block, data_location, data_length)
        
        # Move to next block
        if next_block_location > 0:
            f.seek(next_block_location)
        
        return block
    
    def _read_plain_variable(self, f, block, data_location, data_length, header_start, header_length):
        """Read a plain variable block."""
        # Move past header info to get dimensions
        f.seek(header_start + 100)  # Approximate offset to dims
        
        try:
            # Read stagger and dims
            if block.ndims >= 1:
                dims_data = f.read(block.ndims * 4)
                block.dims = list(struct.unpack(f'<{block.ndims}i', dims_data))
            
            # Read data
            f.seek(data_location)
            
            if block.data_type == self.DATATYPE_REAL8:
                dtype = np.float64
            elif block.data_type == self.DATATYPE_REAL4:
                dtype = np.float32
            elif block.data_type == self.DATATYPE_INTEGER4:
                dtype = np.int32
            elif block.data_type == self.DATATYPE_INTEGER8:
                dtype = np.int64
            else:
                return
            
            total_elements = data_length // np.dtype(dtype).itemsize
            block.data = np.frombuffer(f.read(data_length), dtype=dtype)
            
            if len(block.dims) > 0 and np.prod(block.dims) == len(block.data):
                block.data = block.data.reshape(block.dims[::-1])  # Fortran order
        except Exception as e:
            pass
    
    def _read_constant(self, f, block, data_location, data_length):
        """Read a constant block."""
        f.seek(data_location)
        
        if block.data_type == self.DATATYPE_REAL8:
            block.data = struct.unpack('<d', f.read(8))[0]
        elif block.data_type == self.DATATYPE_REAL4:
            block.data = struct.unpack('<f', f.read(4))[0]
        elif block.data_type == self.DATATYPE_INTEGER4:
            block.data = struct.unpack('<i', f.read(4))[0]
        elif block.data_type == self.DATATYPE_INTEGER8:
            block.data = struct.unpack('<q', f.read(8))[0]
    
    def list_variables(self):
        """List all variables in the file."""
        return list(self.blocks.keys())
    
    def get(self, name):
        """Get a variable by name."""
        if name in self.blocks:
            return self.blocks[name].data
        return None
    
    def __getitem__(self, name):
        return self.get(name)


def read_sdf(filename):
    """Read an SDF file and return an SDFFile object."""
    return SDFFile(filename)


def analyze_simulation(output_dir):
    """Analyze a simulation output directory."""
    output_dir = Path(output_dir)
    sdf_files = sorted(output_dir.glob('*.sdf'))
    
    if not sdf_files:
        print(f"No SDF files found in {output_dir}")
        return
    
    print(f"\nFound {len(sdf_files)} SDF files")
    print("-" * 40)
    
    # Read first and last file
    first = read_sdf(sdf_files[0])
    last = read_sdf(sdf_files[-1])
    
    print(f"Time range: {first.time:.4e} -> {last.time:.4e} s")
    print(f"Step range: {first.step} -> {last.step}")
    print(f"\nVariables in file:")
    for name in first.list_variables()[:10]:
        block = first.blocks[name]
        if block.data is not None:
            if isinstance(block.data, np.ndarray):
                print(f"  {name}: array{block.data.shape}")
            else:
                print(f"  {name}: {block.data}")
    
    # Check for electric field
    ex_names = [n for n in first.list_variables() if 'Ex' in n or 'ex' in n.lower()]
    if ex_names:
        print(f"\nElectric field analysis:")
        for name in ex_names[:1]:
            ex_first = first.get(name)
            ex_last = last.get(name)
            if ex_first is not None and ex_last is not None:
                if isinstance(ex_first, np.ndarray):
                    print(f"  {name} initial max: {np.max(np.abs(ex_first)):.4e}")
                    print(f"  {name} final max:   {np.max(np.abs(ex_last)):.4e}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        analyze_simulation(sys.argv[1])
    else:
        print("Usage: python sdf_simple.py <output_directory>")

