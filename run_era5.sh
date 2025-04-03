#!/bin/bash
#SBATCH -p serc
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --time 96:00:00 
#SBATCH --job-name era5-zarr
#SBATCH --output slurm_worker_logs/main.out
#SBATCH --error slurm_worker_logs/error.out
eval "$(conda shell.bash hook)"
conda activate $GROUP_HOME/robcking/condaenvs/ad99py
python run_era5.py 
echo "done"