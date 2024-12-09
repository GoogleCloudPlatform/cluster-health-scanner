#!/bin/bash
#
#SBATCH --partition=a3mega
#SBATCH --exclusive
#SBATCH --gpus-per-node 8


export SLEEP_TIME_MINUTES=5
export BASH_FILE="gpu_healthcheck.sh"
export DRY_RUN="true"
export IMAGE_TAG="subset"
export BLAST_MODE_ENABLED="true"
export BLAST_MODE_NUM_TESTS_LIMIT=100
export NODES_CHECKED_PER_TEST=2

CONTAINER_MOUNTS="/usr/local/bin,/usr/local/lib,/var/spool/slurmd,/var/spool/slurm,/var/run/slurm,/etc/passwd:/etc/passwd,/usr/lib64,/var/run/munge"
# Launch the litgpt script

sudo srun --container-image=./hc+slurm.sqsh --container-mounts="${CONTAINER_MOUNTS}" --export=BASH_FILE="gpu_healthcheck.sh",DRY_RUN=true --gpus=8 \
	bash -c "python3 health_runner.py"
