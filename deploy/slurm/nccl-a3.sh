#!/bin/bash
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



#SBATCH --partition=a3
#SBATCH --mem=0
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8

set -x
# This should be set to the squashfs file that you created for your application
CONTAINER_IMAGE=./nvidia+pytorch+23.10-py3.sqsh

UDS_PATH="/run/tcpx-${SLURM_JOB_ID}"
USE_TCPX=yes

# Set up NCCL Environment variables
# The following two can be useful for debugging
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET

export NCCL_NET=GPUDirectTCPX_v7
# These network interfaces use Ubuntu's consistent naming scheme. See
# https://manpages.ubuntu.com/manpages/focal/man7/systemd.net-naming-scheme.7.html
export NCCL_SOCKET_IFNAME=enp0s12
export NCCL_GPUDIRECTTCPX_CTRL_DEV=enp0s12
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=enp6s0,enp12s0,enp134s0,enp140s0
export NCCL_CROSS_NIC=0
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=1
export NCCL_DYNAMIC_CHUNK_SIZE=524288
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_P2P_PCI_CHUNKSIZE=524288
export NCCL_P2P_NVL_CHUNKSIZE=1048576
export NCCL_BUFFSIZE=4194304
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_P2P_PXN_LEVEL=0
export NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX=${UDS_PATH}
export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=500000
export NCCL_GPUDIRECTTCPX_TX_BINDINGS="enp6s0:8-21,112-125;enp12s0:8-21,112-125;enp134s0:60-73,164-177;enp140s0:60-73,164-177"
export NCCL_GPUDIRECTTCPX_RX_BINDINGS="enp6s0:22-35,126-139;enp12s0:22-35,126-139;enp134s0:74-87,178-191;enp140s0:74-87,178-191"
export LD_LIBRARY_PATH=/var/lib/tcpx/lib64:${LD_LIBRARY_PATH}


# Here we grab all the environment variables that need to be
# passed down into the container. Slurm would otherwise only pass these env vars
# to the job environment on the host.
# shellcheck disable=SC2001
HOST_VARS=$(sed 's/ \{1,\}/,/g' <<<"${!USE_TCPX*} ${!NCCL*} LD_LIBRARY_PATH")

# Mount /var/tmp to allow the rest of the enroot container to be read-only, and
# mount current $PWD to /nccl to for accessing nccl-tests binary
CONTAINER_MOUNTS="/var/tmp:/var/tmp"

# Mount PWD to /nccl in the enroot container
CONTAINER_MOUNTS=${CONTAINER_MOUNTS},"$PWD:/nccl"

# Mount required directories for GPUDirect-TCPX functionality
CONTAINER_MOUNTS=${CONTAINER_MOUNTS},"/var/lib/tcpx/lib64:/var/lib/tcpx/lib64"
CONTAINER_MOUNTS=${CONTAINER_MOUNTS},${UDS_PATH}:${UDS_PATH}

# Run the workload
srun --mpi=pmi2 \
  --ntasks-per-node=8 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-env="${HOST_VARS}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  sh -c "
  /nccl/nccl-tests/build/all_gather_perf -b 8M -e 8G -f 2 -g 1 -w 5 --iters 300 -c 0;
  "