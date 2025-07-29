#!/bin/bash
#SBATCH --job-name=Patient_Traj_Pred
#SBATCH --nodes=1 # Use 1 node to avoid comm overhead stinky af
#SBATCH --time=00:30:00
#SBATCH --output=stdout_%j.out
#SBATCH --error=stderr_%j.err

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-2.2

# Run the Python script with srun
srun python prepare_data.py