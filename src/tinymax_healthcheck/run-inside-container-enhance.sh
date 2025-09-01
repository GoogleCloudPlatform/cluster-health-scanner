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


set -e

echo "Running 1vm.sh"

RUN_NAME="llama-2-1vm-$(date +%Y-%m-%d-%H-%M)"

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY=$VALUE"
done

XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false
 --xla_gpu_enable_command_buffer='' --xla_gpu_enable_highest_priority_async_stream=true
 --xla_gpu_all_reduce_combine_threshold_bytes=134217728 --xla_gpu_all_gather_combine_threshold_bytes=134217728
 --xla_gpu_reduce_scatter_combine_threshold_bytes=67108864 --xla_gpu_enable_pipelined_all_gather=true
 --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true
 --xla_gpu_enable_while_loop_double_buffering=true
 --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false
 --xla_disable_hlo_passes=rematerialization"

function run_maxtext_lm_gpu() {
  python /deps/MaxText/train.py /deps/MaxText/configs/models/gpu/llama2_7b.yml run_name="$RUN_NAME"  \
    dcn_data_parallelism=1 ici_fsdp_parallelism=8 | tee training.log
  if grep -q "completed step: ${ITERATIONS:-29}" training.log; then
    echo "[TINYMAX]!! TinyMax TEST PASS!"
    return 0
  else
    echo "[TINYMAX]!! Potential Abnormal Node Found! Please check the log file."
    return 1
  fi
}

run_maxtext_lm_gpu
