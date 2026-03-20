#!/bin/bash
#SBATCH --job-name=HuberReg_combo2
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt

#SBATCH --array=0-3

#SBATCH --time=01:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=1

module purge
module load Python

cd /data/gent/489/vsc48953/ML_enhance

echo "Fold: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"

python HuberReg_combo2.py $SLURM_ARRAY_TASK_ID