#!/bin/bash
#
#SBATCH --partition=a3mega
#SBATCH --exclusive
#SBATCH --gpus-per-node 8
#SBATCH --nodes 1


export SLEEP_TIME_MINUTES=5
export YAML_FILE="a3plus/nccl_healthcheck.yaml"
export DRY_RUN="true"
export IMAGE_TAG="subset"
export BLAST_MODE_ENABLED="true"
export BLAST_MODE_NUM_TESTS_LIMIT=100
export NODES_CHECKED_PER_TEST=2


# Launch the litgpt script
sbatch health_runner.py