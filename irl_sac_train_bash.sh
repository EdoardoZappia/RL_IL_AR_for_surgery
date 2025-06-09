#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

echo "Inizio training IRL"
python sac_rot.py