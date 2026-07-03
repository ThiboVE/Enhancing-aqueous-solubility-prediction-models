#!/bin/bash

#PBS -N methane_dft           ## job name
#PBS -l nodes=1:ppn=1         ## single-node job, single core
#PBS -l walltime=2:00:00      ## max. 2h of wall time


module purge
module load Python/3.12.3-GCCcore-13.3.0

mkdir -p ./logs

source ./venv/bin/activate

# pip install --prefer-binary pyscf
pip install rdkit

# pip install git+https://github.com/pyscf/properties

cd ./ML_enhance/DFT

export PYTHONPATH=../:$PYTHONPATH

# mkdir -p ./results

echo "Job name: $JOB_NAME"
echo "Fold: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Using ${SLURM_CPUS_PER_TASK} cpu core(s)"

python run_dft.py 