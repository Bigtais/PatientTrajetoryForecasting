#!/bin/bash
#SBATCH --job-name=Pat_Traj_Pred_ClinicalMosaic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40-48:1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --output=stdout.out
#SBATCH --error=stderr.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-2.2

srun -N1 -n1 python cross_val_train_ddp.py
