#!/bin/bash

#SBATCH --clusters=brain
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=0

#SBATCH --partition=gpu_24g.q
#SBATCH --job-name=conic_final_random
#SBATCH --output=train_hover_3branch_5fold_Likebaseline_copypasterandom_candidate_FinalSplit.out
#SBATCH --error=train_hover_3branch_5fold_Likebaseline_copypasterandom_candidate_FinalSplit.err

echo "--------"
echo "HOSTNAME = ${HOSTNAME}"
echo "SLURM_JOB_NAME = ${SLURM_JOB_NAME}"
echo "SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "--------"

nvcc --version
python train_hover_3branch_5fold_Likebaseline_copypasterandom_candidate_FinalSplit.py