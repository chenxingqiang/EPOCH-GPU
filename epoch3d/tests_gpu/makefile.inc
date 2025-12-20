# Copyright (C) 2025 EPOCH-GPU Development Team
#
# GPU test makefile template for EPOCH3D

EPOCH_GPU?=../../bin/epoch3d_gpu
EPOCH_CPU?=../../bin/epoch3d
MPIPROCS?=1
GPU_CC?=cc70

default: gpu

gpu: epoch3d_gpu.dat

epoch3d_gpu.dat: input.deck $(EPOCH_GPU)
	echo . | mpirun --allow-run-as-root -n $(MPIPROCS) $(EPOCH_GPU)

cpu: epoch3d.dat

epoch3d.dat: input.deck $(EPOCH_CPU)
	echo . | mpirun --allow-run-as-root -n $(MPIPROCS) $(EPOCH_CPU)

compare: gpu cpu
	@echo "Comparing GPU vs CPU results..."
	python3 compare_results.py

benchmark: gpu cpu
	@echo "Performance comparison:"
	@python3 benchmark.py

clean:
	rm -rf *.png *.sdf deck.status epoch{1,2,3}d*.dat *.visit
	rm -rf __pycache__ *.log

.PHONY: default gpu cpu compare benchmark clean

