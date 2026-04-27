#!/bin/bash

#BSUB -J dwellings_gpu_profile
#BSUB -q gpua40
#BSUB -W 00:30
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o dwellings_outputs_%J/job_%J.out
#BSUB -e dwellings_outputs_%J/job_%J.err
#BSUB -B
#BSUB -N
#BSUB -n 2
#BSUB -R "span[hosts=1]"

PROF_BUILDINGS=5
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CUR_DIR=$(pwd)
BASE_DIR="benchmarks/comp_${TIMESTAMP}"
SCRIPT_PATH="$CUR_DIR/simulate_after_cupy.py"

mkdir -p "$CUR_DIR/$BASE_DIR/nsys_profile"

source /dtu/projects/02613_2026/conda/conda_init.sh
conda activate 02613

pip install --user --no-cache-dir cupy-cuda12x
module load cuda/12.1

# Run Profiling
nsys profile \
    --trace=cuda,osrt,nvtx,python-api \
    --output="$CUR_DIR/$BASE_DIR/nsys_profile/cupy_report_%J" \
    --force-overwrite true \
    python -u "$SCRIPT_PATH" "$PROF_BUILDINGS" > "$CUR_DIR/$BASE_DIR/nsys_profile/nsys_log_%J.txt" 2>&1
