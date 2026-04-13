#!/bin/bash
#BSUB -J dwellings
#BSUB -q hpc
#BSUB -W 00:10
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "select[model == XeonE5_2660v3]"
#BSUB -o dwellings_outputs_%J/job_%J.out
#BSUB -e dwellings_outputs_%J/job_%J.err
#BSUB -B
#BSUB -N
#BSUB -n 8
#BSUB -R "span[hosts=1]"

OUTPUT_DIR="dwellings_outputs_${LSB_JOBID}"
mkdir -p "$OUTPUT_DIR"

# Initialize python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Run python script with the unique directory for all outputs
/usr/bin/time -v -o "${OUTPUT_DIR}/system_resource.time" \
python -u simulate.py 10 --bench > "${OUTPUT_DIR}/simulation_results.log" 2>&1

# Create a symbolic link to the latest output directory to find it easier
ln -sfn "$OUTPUT_DIR" dwellings_outputs_latest

echo "Simulation finished. Results saved in ${OUTPUT_DIR}/"