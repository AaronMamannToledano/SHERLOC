#!/bin/bash
# SLURM array job: 100 splits of the lean joint model v2 (train_jm_pada1_v2.R) on PADA-1.
# 3 MCMC chains in parallel (3 cores), 10000 iterations / 3000 burn-in: ~5 min per split.
# Defaults inside the script: residual-SD prior 0.3, time centred at 0.92 months, driver count log1p Gaussian.
# Afterwards:
#   python jm/jm_metrics_pada1.py results/benchmark_response_longitudinal/jm_v2 \
#          results/benchmark_response_longitudinal/JointModel_v2_metrics_per_split.csv
#   python competitors_pada1_summary.py
#SBATCH --job-name=jm2_pada1
#SBATCH --array=0-99
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --partition=cpu_med
#SBATCH --output=results/benchmark_response_longitudinal/jm_v2/slurm_%a.log
mkdir -p results/benchmark_response_longitudinal/jm_v2
export OMP_NUM_THREADS=1
Rscript "$(dirname "$0")/train_jm_pada1_v2.R" ${SLURM_ARRAY_TASK_ID} ${SLURM_ARRAY_TASK_ID} 10000 3000 3 3
