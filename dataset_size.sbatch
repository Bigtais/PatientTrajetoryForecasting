#!/bin/bash
#SBATCH --job-name=Patient_Traj_Pred
#SBATCH --partition=all
#SBATCH --nodes=1 # Use 1 node to avoid comm overhead stinky af
#SBATCH --time=02:00:00
#SBATCH --output=output_logs/stdout/stdout_%j.out
#SBATCH --error=output_logs/stderr/stderr_%j.err

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-2.2

# Run the Python script with srun
srun python dataset_size.py