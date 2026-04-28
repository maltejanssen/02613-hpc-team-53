#!/bin/bash
#BSUB -J simulation
#BSUB -q hpc
#BSUB -W 00:35
#BSUB -R "rusage[mem=512MB]"
#BSUB -o simulation_%J.out
#BSUB -e simulation_%J.err
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model == XeonE5_2650v4]"


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026
python -m simulate 100   



