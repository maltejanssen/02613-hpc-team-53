#!/bin/bash
#BSUB -J cupy_profile
#BSUB -q gpuv100
#BSUB -W 00:30
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o dwellings_outputs_%J/job_%J.out
#BSUB -e dwellings_outputs_%J/job_%J.err
#BSUB -B
#BSUB -N
#BSUB -n 8
#BSUB -R "span[hosts=1]"

# Create the output directory for BSUB logs immediately
mkdir -p "dwellings_outputs_${LSB_JOBID}"

# Setup variables for consistency
NUM_BUILDINGS=5
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="benchmarks/comp_${TIMESTAMP}"
mkdir -p "$BASE_DIR/nsys_profile"

# Initialize python environment
source /dtu/projects/02613_2026/conda/conda_init.sh
conda activate 02613

# Load CUDA module for nsys
module load cuda/12.1

echo "Starting Nsight Systems Profiling on $NUM_BUILDINGS buildings..."

# --- Run Profiling ---
nsys profile \
    --trace=cuda,osrt,nvtx \
    --output="$BASE_DIR/nsys_profile/cupy_report_${TIMESTAMP}" \
    --force-overwrite true \
    python -u "simulate_after_cupy_optimized.py" "$NUM_BUILDINGS" > "$BASE_DIR/nsys_profile/nsys_log_${TIMESTAMP}.txt" 2>&1

echo "Profiling finished. Report stored in $BASE_DIR/nsys_profile"
