# Copyright (C) 2024 EPOCH-GPU Development Team
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

# GPU test makefile template
# Include this in each GPU test subdirectory

# GPU binary location
EPOCH_GPU?=../../bin/epoch2d_gpu
EPOCH_CPU?=../../bin/epoch2d

# MPI configuration
MPIPROCS?=1

# GPU compute capability
GPU_CC?=cc70

default: gpu

# Run with GPU acceleration
gpu: epoch2d_gpu.dat

epoch2d_gpu.dat: input.deck $(EPOCH_GPU)
	echo . | mpirun --allow-run-as-root -n $(MPIPROCS) $(EPOCH_GPU)

# Run CPU version for comparison
cpu: epoch2d.dat

epoch2d.dat: input.deck $(EPOCH_CPU)
	echo . | mpirun --allow-run-as-root -n $(MPIPROCS) $(EPOCH_CPU)

# Run both and compare
compare: gpu cpu
	@echo "Comparing GPU vs CPU results..."
	python3 compare_results.py

# Performance benchmark
benchmark: gpu cpu
	@echo "Performance comparison:"
	@python3 benchmark.py

clean:
	rm -rf *.png *.sdf deck.status epoch{1,2,3}d*.dat *.visit
	rm -rf __pycache__ *.log

.PHONY: default gpu cpu compare benchmark clean

