#!/bin/bash
#SBATCH --job-name=Patient_Traj_Pred
#SBATCH --partition=all
#SBATCH --nodes=1 # Use 1 node to avoid comm overhead stinky af
#SBATCH --cpus-per-task=5
#SBATCH --time=02:00:00
#SBATCH --output=output_logs/stdout/stdout_%j.out
#SBATCH --error=output_logs/stderr/stderr_%j.err

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-2.2

# Set distributed training environment variables
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345
export WORLD_SIZE=$SLURM_NTASKS

# Optional: Set CUDA_VISIBLE_DEVICES for each process (if not handled in Python)
# SLURM_LOCALID can help assign GPUs per task on each node
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Run the Python script with srun
srun python download_data.py