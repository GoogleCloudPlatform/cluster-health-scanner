#!/bin/bash
#
#SBATCH --partition=a3mega
#SBATCH --exclusive
#SBATCH --gpus-per-node 8
#SBATCH --nodes 1

HEALTH_VALIDITY_HOURS=24
DRY_RUN=true
R_LEVEL=2

# Launch the script
sbatch gpu_healthcheck.py