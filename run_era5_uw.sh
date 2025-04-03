#!/bin/bash
#SBATCH -p serc
#SBATCH --mem 32G
#SBATCH -c 8
#SBATCH --time 60:00:00 
#SBATCH --job-name era5-uw-zarr
#SBATCH --output slurm_worker_logs/main_uw.out
#SBATCH --error slurm_worker_logs/error_uw.out
eval "$(conda shell.bash hook)"
conda activate $GROUP_HOME/robcking/condaenvs/ad99py
python run_era5_uw.py 
echo "done"
