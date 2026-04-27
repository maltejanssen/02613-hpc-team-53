#!/bin/bash

#BSUB -J dwellings_gpu
#BSUB -q gpua100
#BSUB -W 00:30
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o dwellings_outputs_%J/job_%J.out
#BSUB -e dwellings_outputs_%J/job_%J.err
#BSUB -B
#BSUB -N
#BSUB -n 8
#BSUB -R "span[hosts=1]"

# Run the comparison on 100 buildings
NUM_BUILDINGS=100
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="benchmarks/comp_${TIMESTAMP}"
mkdir -p "$BASE_DIR"

# Initialize python environment
source /dtu/projects/02613_2026/conda/conda_init.sh
conda activate 02613

pip install --user --no-cache-dir cupy-cuda12x

module load cuda/12.1
#  Run NumPy Version
echo "Running NumPy (Original)..."
mkdir -p "$BASE_DIR/numpy"
/usr/bin/time -v -o "$BASE_DIR/numpy/resources.txt" \
python -u "simulate_before_cupy.py" "$NUM_BUILDINGS" > "$BASE_DIR/numpy/runtime_log.txt" 2>&1

#  Run CuPy Version
echo "Running CuPy (GPU Accelerated)..."
mkdir -p "$BASE_DIR/cupy"
/usr/bin/time -v -o "$BASE_DIR/cupy/resources.txt" \
python -u "simulate_after_cupy.py" "$NUM_BUILDINGS" > "$BASE_DIR/cupy/runtime_log.txt" 2>&1

