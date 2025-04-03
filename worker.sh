#!/usr/bin/env bash
#SBATCH -J dask-worker-rob
#SBATCH -e slurm_worker_logs/dask-worker-%J-rob.err
#SBATCH -o slurm_worker_logs/dask-worker-%J-rob.out
#SBATCH -p serc
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=17G
#SBATCH -t 96:00:00
/home/groups/aditis2/robcking/condaenvs/ad99py/bin/python -m distributed.cli.dask_worker tcp://10.51.9.69:35370 --name rob-worker-${SLURM_JOB_ID} --nthreads 1 --memory-limit 4.19GiB --nworkers 4 --nanny --death-timeout 60 --interface ib0
