#!/bin/bash

#SBATCH --clusters=brain
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0

#SBATCH --partition=learning.q
#SBATCH --nodelist=dive602
#SBATCH --job-name=inf
#SBATCH --output=inference.out
#SBATCH --error=inference.err

echo "--------"
echo "HOSTNAME = ${HOSTNAME}"
echo "SLURM_JOB_NAME = ${SLURM_JOB_NAME}"
echo "SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "--------"

nvcc --version
python inference.py