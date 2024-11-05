#!/bin/bash
#
#SBATCH --partition=a3mega
#SBATCH --exclusive
#SBATCH --gpus-per-node 8
#SBATCH --nodes 1

NODE_NAME=$HOSTNAME
HEALTH_VALIDITY_HOURS=24
DRY_RUN=true
R_LEVEL=2

# Launch the script
srun --container-image=./gpu+slurm.sqsh bash -c "python3 /app/gpu_healthcheck.py"
