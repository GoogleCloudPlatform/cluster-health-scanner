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

#!/bin/bash
set -x

# Sample command to run the workload:
#  /scripts/run-nccl-fastrak.sh ${collective_name}_perf "${LD_LIBRARY_PATH}" ${gpu_per_node} ${if_name_list} ${msg_min} ${msg_max} ${channel_per_gpu} ${num_node} ${num_iteration}
#  e.g. /scripts/run-nccl-fastrak.sh all_gather_perf "${LD_LIBRARY_PATH}" 8 eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8 1M 512M 3 2 10

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

run_nccl_rdma() {
  local -r benchmark=$1
  local -r ld_library_path_override=$2
  local -r gpu_per_node=$3
  local -r socket_ifnames=$4
  local -r data_b=$5
  local -r data_e=$6
  local nhosts=2
  if ! [[ -z "$7" ]]; then
    nhosts=$7
  fi
  local channels_per_gpu=1
  if ! [[ -z "$8" ]]; then
    channels_per_gpu=$8
  fi
  local -r iter=20

  LD_LIBRARY_PATH=${ld_library_path_override} \
  mpirun --mca btl tcp,self --mca btl_tcp_if_include eth0 --allow-run-as-root \
    -np $(( gpu_per_node * "${nhosts}" )) \
    --hostfile "${SCRIPT_DIR}/hostfiles${nhosts}/hostfile${gpu_per_node}" \
    -x NCCL_SOCKET_IFNAME=eth0,eth1 \
    -x LD_LIBRARY_PATH -x PATH \
    -x NCCL_CROSS_NIC=0 \
    -x NCCL_ALGO=Ring \
    -x NCCL_PROTO=Simple \
    -x NCCL_NSOCKS_PERTHREAD=4 \
    -x NCCL_SOCKET_NTHREADS=1 \
    -x NCCL_DYNAMIC_CHUNK_SIZE=524288 \
    -x NCCL_BUFFSIZE=4194304 \
    -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -x NCCL_GPUDIRECTTCPX_SOCKET_IFNAME="${socket_ifnames}" \
    -x NCCL_GPUDIRECTTCPX_CTRL_DEV=eth0 \
    -x NCCL_NET_GDR_LEVEL=PIX \
    -x NCCL_P2P_PXN_LEVEL=2 \
    -x NCCL_IB_QPS_PER_CONNECTION=8 \
    -x NCCL_P2P_NET_CHUNKSIZE=524288 \
    -x NCCL_P2P_PCI_CHUNKSIZE=524288 \
    -x NCCL_P2P_NVL_CHUNKSIZE=1048576 \
    -x NCCL_NVLS_CHUNKSIZE=524288 \
    -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=ENV \
    -x NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX="${UNIX_CLIENT_PREFIX}" \
    -x NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=1000000 \
    -x NCCL_GPUDIRECTTCPX_FORCE_ACK \
    /opt/nccl-tests/build/"${benchmark}" \
      -b "${data_b}" -e "${data_e}" -f 2 -g 1 -w 5--iters "${iter}" 2>&1 | \
    tee "${benchmark}_${nhosts}_${gpu_per_node}_${socket_ifnames}_i${iter}.txt"
}

run_nccl_fastrak() {
  local -r benchmark=$1
  local -r ld_library_path_override=$2
  local -r gpu_per_node=$3
  local -r socket_ifnames=$4
  local -r data_b=$5
  local -r data_e=$6
  local nhosts=2
  if [[ -n "$7" ]]; then
    nhosts=$7
  fi
  local channels_per_gpu=3
  if [[ -n "$8" ]]; then
    channels_per_gpu=$8
  fi
  local iter=20
  if [[ -n "$9" ]]; then
    iter=$9
  fi
  local -r num_channel=$((gpu_per_node*channels_per_gpu))
  # Sourcing the nccl-env-profile.sh file to set the most up to date environment variables from nccl team.
  # NCCL_LIB_DIR="/usr/local/nvidia/lib64"
  # source "${NCCL_LIB_DIR}"/nccl-env-profile.sh


  LD_LIBRARY_PATH=${ld_library_path_override} \
  mpirun --mca btl tcp,self --mca btl_tcp_if_include eth0 --allow-run-as-root \
    -np $(( gpu_per_node * "${nhosts}" )) \
    --hostfile "${SCRIPT_DIR}/hostfiles${nhosts}/hostfile${gpu_per_node}" \
    -x LD_LIBRARY_PATH -x PATH \
    -x NCCL_FASTRAK_CTRL_DEV=eth0 \
    -x NCCL_FASTRAK_IFNAME="${socket_ifnames}" \
    -x NCCL_DEBUG_FILE=/tmp/log/"${benchmark}"-%h-%p.log \
    -x NCCL_TOPO_DUMP_FILE=/tmp/log/"${benchmark}"_topo.txt \
    -x NCCL_GRAPH_DUMP_FILE=/tmp/log/"${benchmark}"_graph.txt \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -x NCCL_CROSS_NIC=0 \
    -x NCCL_ALGO="Ring,Tree" \
    -x NCCL_PROTO=Simple \
    -x NCCL_MIN_NCHANNELS=4 \
    -x NCCL_DYNAMIC_CHUNK_SIZE=524288 \
    -x NCCL_P2P_NET_CHUNKSIZE=524288 \
    -x NCCL_P2P_PCI_CHUNKSIZE=524288 \
    -x NCCL_P2P_NVL_CHUNKSIZE=1048576 \
    -x NCCL_FASTRAK_NUM_FLOWS=2 \
    -x NCCL_BUFFSIZE=8388608 \
    -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -x NCCL_NET_GDR_LEVEL=PIX \
    -x NCCL_DEBUG_SUBSYS=INIT,NET \
    -x NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING=0 \
    -x NCCL_FASTRAK_USE_SNAP="1" \
    -x NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL="0" \
    -x NCCL_FASTRAK_USE_LLCM=1 \
    -x NCCL_TUNER_PLUGIN="libnccl-tuner.so" \
    -x NCCL_TUNER_CONFIG_PATH="/usr/local/nvidia/lib64/a3plus_tuner_config.textproto" \
    -x NCCL_NVLS_ENABLE="0" \
    -x NCCL_LIB_DIR="/usr/local/nvidia/lib64" \
    -x NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE="/usr/local/nvidia/lib64/a3plus_guest_config.textproto" \
    -x NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000 \
    taskset -c 32-63 /third_party/nccl-tests-mpi/build/"${benchmark}" \
      -b "${data_b}" -e "${data_e}" -f 2 -g 1 -w 50 --iters "${iter}" 2>&1 | \
    tee "${benchmark}_${nhosts}_${gpu_per_node}_${socket_ifnames}_i${iter}.txt"
}

run_nccl_gpudirect() {
  local -r benchmark=$1
  local -r ld_library_path_override=$2
  local -r gpu_per_node=$3
  local -r socket_ifnames=$4
  local -r data_b=$5
  local -r data_e=$6
  local nhosts=2
  if [[ -n "$7" ]]; then
    nhosts=$7
  fi
  local channels_per_gpu=1
  if [[ -n "$8" ]]; then
    channels_per_gpu=$8
  fi
  local -r iter=20

  LD_LIBRARY_PATH=${ld_library_path_override} \
  mpirun --mca btl tcp,self --mca btl_tcp_if_include eth0 --allow-run-as-root \
    -np $(( gpu_per_node * "${nhosts}" )) \
    --hostfile "${SCRIPT_DIR}/hostfiles${nhosts}/hostfile${gpu_per_node}" \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -x LD_LIBRARY_PATH -x PATH \
    -x NCCL_CROSS_NIC=0 \
    -x NCCL_ALGO=Ring \
    -x NCCL_PROTO=Simple \
    -x NCCL_NSOCKS_PERTHREAD=4 \
    -x NCCL_SOCKET_NTHREADS=1 \
    -x NCCL_MAX_NCHANNELS=${num_channel} \
    -x NCCL_MIN_NCHANNELS=${num_channel} \
    -x NCCL_DYNAMIC_CHUNK_SIZE=524288 \
    -x NCCL_BUFFSIZE=4194304 \
    -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -x NCCL_GPUDIRECTTCPX_SOCKET_IFNAME="${socket_ifnames}" \
    -x NCCL_GPUDIRECTTCPX_CTRL_DEV=eth0 \
    -x NCCL_NET_GDR_LEVEL=PIX \
    -x NCCL_P2P_PXN_LEVEL=0 \
    -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=ENV \
    -x NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX="${UNIX_CLIENT_PREFIX}" \
    -x NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=1000000 \
    -x NCCL_GPUDIRECTTCPX_FORCE_ACK \
    /third_party/nccl-tests-mpi/build/"${benchmark}" \
      -b "${data_b}" -e "${data_e}" -f 2 -g 1 -w 5 --iters 20 2>&1 | \
    tee "${benchmark}_${nhosts}_${gpu_per_node}_${socket_ifnames}_i${iter}.txt"
}


plugin_name=$1
shift
if [[ "$plugin_name" == "rdma" ]]; then
  run_nccl_rdma "$@"
elif [[ "$plugin_name" == "fastrak" ]]; then
  run_nccl_fastrak "$@"
elif [[ "$plugin_name" == "gpudirect" ]]; then
  run_nccl_gpudirect "$@"
else
  echo "Error: Invalid argument. Please specify 'fastrak' or 'gpudirect'."
  exit 1
fi