# Copyright (C) 2024 EPOCH-GPU Development Team
#
# GPU test makefile template for EPOCH1D

EPOCH_GPU?=../../bin/epoch1d_gpu
EPOCH_CPU?=../../bin/epoch1d
MPIPROCS?=1
GPU_CC?=cc70

default: gpu

gpu: epoch1d_gpu.dat

epoch1d_gpu.dat: input.deck $(EPOCH_GPU)
	echo . | mpirun --allow-run-as-root -n $(MPIPROCS) $(EPOCH_GPU)

cpu: epoch1d.dat

epoch1d.dat: input.deck $(EPOCH_CPU)
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

