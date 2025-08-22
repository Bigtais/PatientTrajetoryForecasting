#!/bin/bash
#SBATCH --job-name=Patient_Traj_Pred
#SBATCH --partition=gpu
#SBATCH --nodes=1                 # Single node
#SBATCH --ntasks=1                # Only one task (torchrun handles multi-proc)
#SBATCH --gres=gpu:a40-48:5
#SBATCH --cpus-per-task=5
#SBATCH --time=7-00:00:00
#SBATCH --output=output_logs/stdout/stdout_%j.out
#SBATCH --error=output_logs/stderr/stderr_%j.err

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-2.2

# Master node info
MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29501

# Torchrun launch
torchrun --nnodes=1 --nproc_per_node=5 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
         train_gnn.py
