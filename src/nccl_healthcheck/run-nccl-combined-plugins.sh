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

set -x
echo "Script started with args: $@"  # Debug point 1

# Sample command to run the workload:
#  /scripts/run-nccl-fastrak.sh ${collective_name}_perf "${LD_LIBRARY_PATH}" ${gpu_per_node} ${if_name_list} ${msg_min} ${msg_max} ${channel_per_gpu} ${num_node} ${num_iteration}
#  e.g. /scripts/run-nccl-fastrak.sh all_gather_perf "${LD_LIBRARY_PATH}" 8 eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8 1M 512M 3 2 10

#SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR="/opt/apps"

run_nccl_fastrak() {
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
  local channels_per_gpu=3
  if ! [[ -z "$8" ]]; then
    channels_per_gpu=$8
  fi
  local -r num_channel=$((gpu_per_node*channels_per_gpu))
  local -r iter=20
  #NCCL_LIB_DIR="/var/lib/tcpxo/lib64"
  NCCL_LIB_DIR="/usr/local/nvidia/lib64"
  echo "Sourcing ${NCCL_LIB_DIR}/nccl-env-profile.sh"
  source "${NCCL_LIB_DIR}/nccl-env-profile.sh"
  NCCL_FLAGS=$( env | egrep ^NCCL | awk '{ printf "-x %s ", $0; }' )
  # shellcheck disable=SC2086i
  echo "About to run mpirun"  # Debug point 

  hostfile="/opt/apps/hostfiles${nhosts}/hostfile${gpu_per_node}"
  echo "Hostfiel path: $hostfile"
  if [ -f "$hostfile" ]; then
    echo "Hostfile exists"
    echo "Content:"
    cat "$hostfile"
  else
    echo "Hostfile does not exist"
    ls -l /tmp/hostfiles${nhosts}/ || echo "Cannot list directory"
  fi
  whoami
  id
  which mpirun
  mpirun --version
  dmesg | tail
  journalctl -xe | tail
  echo "=====ompi_info======"
  ompi_info --param plm slurm  # If Open MPI is used
  echo "===openmpi shared libraries are accesible========"
  ldd $(which mpirun)
  echo "=======check if there are any blocked ports or firewall issues======="
  iptables -L
  netstat -tulpn
  echo "=======run scontrol show config, sinfo  =========="
  scontrol show config
  sinfo	
  #mpirun --mca plm slurm -v --allow-run-as-root -np 1 hostname
  echo "======== end of debuging======"
  # Check network interfaces
  cat /opt/apps/hostfiles${nhosts}/hostfile${gpu_per_node}
  #mpirun -v --allow-run-as-root -np 1 hostname
  echo "=====ls -l /dev/shm ======="
  ls -l /dev/shm
  echo "======ompi_info --all | grep btl ===="
  ompi_info --all | grep btl
  echo "========ompi_info | grep pmi====="
  ompi_info | grep pmi
  echo "====ifconfig===="
  ifconfig
  echo "======ip addr======"
  ip addr
  echo "======= mpirun begin try explicitly enabling only TCO by using======="
  #mpirun --allow-run-as-root -np 1 --host localhost hostname
  #mpirun --allow-run-as-root -np 1 --mca btl self,vader hostname
  #mpirun --allow-run-as-root -np 1 --mca btl tcp,self hostname
  #mpirun --allow-run-as-root -np 1 --mca btl self,vader --mca btl_vader_single_copy_mechanism none hostname
  #mpirun --allow-run-as-root -np 1 \
  #  --mca btl self,vader \
  #  --mca btl_vader_single_copy_mechanism none \
  #  --mca opal_common_vader_mmap_enable_nfs_warning 0 \
  #  hostname
  #mpirun --allow-run-as-root -np 1 \
  #  --mca btl ^vader,sm \
  #  --mca btl_tcp_if_include eth0 \
  #  hostname
  #mpirun --allow-run-as-root -np 1 \
  #  -x OMPI_MCA_btl \
  #  -x OMPI_MCA_oob \
  #  -x OMPI_MCA_btl_tcp_if_include \
  #  hostname
  echo "==== mpi run end ====="
  # Try verbose MPI run to get more diagnostic info
  #mpirun -v --allow-run-as-root \
  #-np 1  \
  #--hostfile "/opt/apps/hostfiles${nhosts}/hostfile${gpu_per_node}" \
  #echo "test"

  #mpirun --allow-run-as-root \
  #  -np $(( gpu_per_node * nhosts )) \
  #  --hostfile "/tmp/hostfiles${nhosts}/hostfile${gpu_per_node}" \
  #  echo "test" 
  otherhost=$(head -n 1 "/opt/apps/hostfiles${nhosts}/hostfile${gpu_per_node}")
  LD_LIBRARY_PATH=${ld_library_path_override} \
  #mpirun --mca btl tcp,self --mca btl_tcp_if_include eth0 --allow-run-as-root \
  #  -np $(( gpu_per_node * "${nhosts}" )) \
  #  --host localhost \
  #  -x LD_LIBRARY_PATH -x PATH \
  echo "=============echo LD_LIBRARY_PATH============"
  echo $LD_LIBRARY_PATH
  #export LD_LIBRARY_PATH=/usr/local/gib/lib64:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
  #echo $LD_LIBRARY_PATH
  #echo "======ls /usr/local/gib/lib64/libnccl.so.2==="
  #ls /usr/local/fastrak/lib64/libnccl.so.2
  #echo "======sudo ldconfig /usr/local/gib/lib64====="
  #sudo ldconfig /usr/local/fastrak/lib64

  #echo "======ldconfig -p | grep libnccl========"
  #ldconfig -p | grep libnccl
  #echo "=======ldd /opt/nccl-tests/build/all_gather_perf====="
  #ldd /opt/nccl-tests/build/all_gather_perf
  #echo "======add config file to help ldconfig find the lib"
  #echo "/usr/local/gib/lib64" | sudo tee /etc/ld.so.conf.d/nccl.conf
  #sudo ldconfig
  #echo "====verify the lib details===="
  #file /usr/local/gib/lib64/libnccl.so.2
  #echo "======check lib dependency and compatibility====="
  #ldd /usr/local/gib/lib64/libnccl.so.2
  #echo "=======readlink -f /usr/local/gib/lib64/libnccl.so.2===="
  #readlink -f /usr/local/gib/lib64/libnccl.so.2
  echo "=====end====="
  echo "=======/etc/os-release======"
  cat /etc/os-release
  taskset -c 32-63 /third_party/nccl-tests-mpi/build/"${benchmark}" \
    -b "${data_b}" -e "${data_e}" -f 2 -g 1 -w 5 --iters "${iter}" 2>&1 | \
  tee "/tmp/${benchmark}_${nhosts}_${gpu_per_node}_${socket_ifnames}_i${iter}.txt"
}

run_nccl_gpudirect() {
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
if [[ "$plugin_name" == "fastrak" ]]; then
  echo "Getting into the fastrak func"
  run_nccl_fastrak "$@"
elif [[ "$plugin_name" == "gpudirect" ]]; then
  echo "Getting into the gpudirect fun"
  run_nccl_gpudirect "$@"
else
  echo "Error: Invalid argument. Please specify 'fastrak' or 'gpudirect'."
  exit 1
fi
