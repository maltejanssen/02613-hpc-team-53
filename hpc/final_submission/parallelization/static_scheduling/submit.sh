#!/bin/bash
#BSUB -J simulation
#BSUB -q hpc
#BSUB -W 00:35
#BSUB -R "rusage[mem=256MB]"
#BSUB -o simulation_parallel2_%J.out
#BSUB -e simulation_parallel2_%J.err
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model == XeonE5_2650v4]"


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026
echo "cores: $LSB_DJOB_NUMPROC"
python -m simulate 100 $LSB_DJOB_NUMPROC



