#!/bin/bash
#
#SBATCH --partition=a3mega
#SBATCH --exclusive
#SBATCH --gpus-per-node 8
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=8

export HEALTH_VALIDITY_HOURS=24
export DRY_RUN=true
export NHOSTS=2
export nr=8
export BANDWIDTH_THRESHOLD=90
export START_MESSAGE_SIZE="2G"
export END_MESSAGE_SIZE="8G"
export USE_FASTRAK="true"
export NCCL_FASTRAK_USE_SNAP="1"
export NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL="0"
export NCCL_FASTRAK_NUM_FLOWS="2"

export NODE_NAME=$HOSTNAME
export NHOSTS=2
CONTAINER_IMAGE=./nccl+slurm.sqsh
#NCCL_LIB_DIR="/usr/local/nvidia/lib64" source /var/lib/tcpxo/lib64/nccl-env-profile.sh
export NCCL_FASTRAK_CTRL_DEV=enp0s12
export NCCL_FASTRAK_IFNAME=enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0
export NCCL_SOCKET_IFNAME=enp0s12
export NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY=/dev/aperture_devices

#HOST_VARS=$(sed 's/ \{1,\}/,/g' <<<"${!NCCL*}")

echo $HOST_VARS

CONTAINER_MOUNTS="/var/tmp:/var/tmp"

CONTAINER_MOUNTS=${CONTAINER_MOUNTS},"/var/lib/tcpxo/lib64/,/usr/local/bin,/usr/local/lib,/usr/sbin,/var/spool/slurmd,/var/spool/slurm,/var/run/slurm,/etc/passwd:/etc/passwd,/usr/lib64,/var/run/munge,/tmp,/root/.ssh:/root/.ssh,/tmp:/tmp,/etc/ssh:/etc/ssh,/opt/apps:/opt/apps"

sudo srun  \
        --container-image=./nccl+slurm.sqsh \
	--mpi=pmi2 \
        --export=HEALTH_VALIDITY_HOURS=24,DRY_RUN=true,NHOSTS=2,nr=8,JOB_NAME="neper-healthcheck-${SLURM_JOB_ID}",SERVICE_NAME="neper-headless-svc-${SLURM_JOB_ID}",GOOD_THROUGHPUT="50000000000",NODE_IP=$NODE_IP,POD_NAME=$POD_NAME,NODES=$SLURM_JOB_NODELIST,NODE_NAME=$HOSTNAME,JOB_COMPLETION_INDEX=0,BANDWIDTH_THRESHOLD="90",START_MESSAGE_SIZE="2G",END_MESSAGE_SIZE="8G",INSTANCE_TYPE="a3-megagpu-8g",ITERATIONS="1" --gpus=8 --nodes 2 --exclusive \
        --container-mounts="${CONTAINER_MOUNTS}" \
        sh -c "pip3 install protobuf && python3 /scripts/nccl_startup.py"
