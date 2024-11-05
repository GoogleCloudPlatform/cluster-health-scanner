#!/bin/bash
#
#SBATCH --partition=a3mega
#SBATCH --exclusive
#SBATCH --gpus-per-node 8
#SBATCH --nodes 1

export NODE_NAME=$HOSTNAME
echo "Running on node: $NODE_NAME"
export HEALTH_VALIDITY_HOURS=24
export DRY_RUN=true
export R_LEVEL=2

# Launch the script
srun --container-image=./gpu+slurm.sqsh bash -c "python3 /app/gpu_healthcheck.py"
