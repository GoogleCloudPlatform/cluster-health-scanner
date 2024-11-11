#!/bin/bash
#
#SBATCH --partition=a3mega
#SBATCH --exclusive
#SBATCH --gpus-per-node 8
#SBATCH --nodes 2

export HEALTH_VALIDITY_HOURS=24
export DRY_RUN=true
export NHOSTS=2
export nr=8
BANDWIDTH_THRESHOLD=90
START_MESSAGE_SIZE="2G"
END_MESSAGE_SIZE="8G"
USE_FASTRAK="true"
NCCL_FASTRAK_USE_SNAP="1"
NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL="0"
NCCL_FASTRAK_NUM_FLOWS="2"
GOOD_THROUGHPUT=130000000000

export NODE_NAME=$HOSTNAME
export NODE_IP=""
export POD_NAME=""
export JOB_COMPLETION_INDEX=-1
export JOB_NAME="neper-healthcheck-${CHECK_TIME_EPOCH_SEC}"
export SERVICE_NAME="neper-headless-svc-${CHECK_TIME_EPOCH_SEC}"
export INSTANCE_TYPE="a3-megagpu-8g"
export GOOD_THROUGHPUT="50000000000"
CONTAINER_MOUNTS="/usr/local/bin,/usr/local/lib,/var/spool/slurmd,/var/spool/slurm,/var/run/slurm,/etc/passwd:/etc/passwd,/usr/lib64,/var/run/munge,/tmp,/root/.ssh:/root/.ssh,/tmp:/tmp"

# Launch the litgpt script
#srun --container-image=./neper+slurm.sqsh --container-mounts="${CONTAINER_MOUNTS}" bash -c "python3 /scripts/neper_runner.py"
sudo srun --container-image=./neper+slurm.sqsh --container-mounts="${CONTAINER_MOUNTS}" --export=HEALTH_VALIDITY_HOURS=24,DRY_RUN=true,NHOSTS=2,nr=8,NODE_NAME=$HOSTNAME,JOB_NAME="neper-healthcheck-${SLURM_JOB_ID}",SERVICE_NAME="neper-headless-svc-${SLURM_JOB_ID}",GOOD_THROUGHPUT="50000000000",NODE_IP=$NODE_IP,POD_NAME=$POD_NAME,NODES=$SLURM_JOB_NODELIST  --gpus=8 --nodes 2 --exclusive bash -c "python3 /scripts/neper_runner.py"
