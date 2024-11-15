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

CONTAINER_IMAGE=./nccl+slurm.sqsh
NCCL_LIB_DIR="/usr/local/nvidia/lib64" source /var/lib/tcpxo/lib64/nccl-env-profile.sh
export NCCL_FASTRAK_CTRL_DEV=enp0s12
export NCCL_FASTRAK_IFNAME=enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0
export NCCL_SOCKET_IFNAME=enp0s12
export NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY=/dev/aperture_devices

HOST_VARS=$(sed 's/ \{1,\}/,/g' <<<"${!NCCL*}")

echo $HOST_VARS

CONTAINER_MOUNTS="/var/tmp:/var/tmp"

#CONTAINER_MOUNTS=${CONTAINER_MOUNTS},"$PWD:/opt/"

CONTAINER_MOUNTS=${CONTAINER_MOUNTS},"/var/lib/tcpxo/lib64/"
#CONTAINER_MOUNTS=${CONTAINER_MOUNTS},"/usr/local/tcpxo/lib64/"
# Launch the litgpt script
#srun -l \
#	-N "${SLURM_NNODES}" \
#	--mpi=pmi2 \
#	--ntasks-per-node=8 \
#	--container-image="${CONTAINER_IMAGE}" \
#	--container-env="${HOST_VARS}" \
#	--container-mounts="${CONTAINER_MOUNTS}" \
#	sh -c "
#  export LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH;
#  /third_party/nccl-tests-mpi/build/all_gather_perf -b 2G -e 8G -f 2 -g 1 -w 5 --iters 200 -c 0;
#  "
#srun -l \
#        -N "${SLURM_NNODES}" \
#        --mpi=pmi2 \
#        --ntasks-per-node=8 \
#        --container-image="${CONTAINER_IMAGE}" \
#        --container-env="${HOST_VARS}" \
#        --container-mounts="${CONTAINER_MOUNTS}" \
#        sh -c "
#  export LD_LIBRARY_PATH=/usr/local/tcpxo/lib64:/usr/local/nvidia/lib64/;
#  /third_party/nccl-tests-mpi/build/all_gather_perf -b 2G -e 8G -f 2 -g 1 -w 5 --iters 200 -c 0;
#  "
srun -l \
        -N "${SLURM_NNODES}" \
        --mpi=pmi2 \
        --ntasks-per-node=8 \
        --container-image="${CONTAINER_IMAGE}" \
        --container-env="${HOST_VARS}" \
        --container-mounts="${CONTAINER_MOUNTS}" \
        sh -c "python3 /scripts/nccl_startup.py"
