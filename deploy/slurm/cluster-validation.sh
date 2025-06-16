#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


# Modifications from original work:
# - add --drain-bad-nodes flag to drain nodes that fail diagnostics
# - supports a3, a3mega, a3ultra and a4 machine types for NCCL tests
# - Added download and enroot the docker images if they are not already present
# - Change NCCL test to make it work for GCP (a3, a3mega, a3ultra and a4), originally it was for oci
# - add --results-dir flag to specify the directory to store results
# - add --relative-exec-path flag to specify the relative path to the cluster validation scripts

# Original work:
# https://github.com/NVIDIA/NeMo-Framework-Launcher/blob/main/csp_tools/oci/cluster_validation.sh

usage() {
cat <<EOF

Validate cluster compute nodes' GPUs and node-to-node communication using 
DCGM Diagnostics and NCCL all_reduce_perf bus bandwidth test

Usage:
  $0 [--OPTION=[VAL] ...]

  OPTION         DESCRIPTION
  --nodelist     List of nodes to run validation on. Can be a comma-separated
                    list or a range such as "hostname-[1-4,6-8]". Same 
                    format as sinfo.
  --nodes        Number of nodes specified in --nodelist
  --partition    Slurm partition of nodes to run validation on. See sinfo for
                    valid partitions.
  --dcgm         Run only DCGM diagnostic
  --nccl         Run only NCCL test
  --machine-type  Machine type of the nodes. Supported values are a3-highgpu-8g,
                    a3-megagpu-8g, a3-ultragpu-8g and a4-highgpu-8g.
  --drain-bad-nodes  Drain nodes that fail diagnostics
  --relative-exec-path  Relative path to the cluster validation scripts. Use this if
                    the scripts are not in the root directory.
  --results-dir   Directory to store results. Default is results/cluster_validation
EOF
}

required() {
cat <<EOF

Input error. Required Flags:
  --nodelist
  --nodes
  --partition
  --machine-type
EOF
}

error_exit() {
    echo -e "Error: " "$1"
    exit "$2"
}

join () {
  local IFS="$1"
  shift
  echo "$*"
}

RELATIVE_PATH=""

# Define where test logs should be written to and read from
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Define the full path for the output file in that target directory
RESULTS_DIR="$SCRIPT_DIR/../../results/cluster_validation"

# Read arguments
while [[ $# -gt 0 ]]; do
  if [[ "$1" =~ ^-.*= ]]; then
    key="${1%%=*}"
    val="${1#*=}"
    val_separate=0
  else
    key="$1"
    val="$2"
    val_separate=1
  fi
  key="$(echo "$key" | tr '[:upper:]' '[:lower:]')"

  case "$key" in
    --nodelist)
      NODES="$val"
      shift $((val_separate+1))
      ;;
    --nodes)
      NUM_NODES="$val"
      shift $((val_separate+1))
      ;;
    --partition)
      PARTITION="$val"
      shift $((val_separate+1))
      ;;
    --machine-type)
      MACHINE_TYPE="$val"
      shift $((val_separate+1))
      ;;
    --dcgm)
      RUN_DCGMI=1
      shift
      ;;
    --nccl)
      RUN_NCCL=1
      shift
      ;;
    --drain-bad-nodes)
      DRAIN_BAD_NODES=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --relative-exec-path)
      RELATIVE_PATH="$val""/"
      shift
      ;;
    --results-dir)
      RESULTS_DIR="$val"
      shift
      ;;
    *)
      usage
      error_exit "Unrecognized option $key." 1
      ;;
  esac
done

# These arguments are required for sbatch commands
if [[ -z "$NODES" ]] || [[ -z "$NUM_NODES" ]] || [[ -z "$PARTITION" ]] || [[ -z "$MACHINE_TYPE" ]]; then
    required
    exit 1
fi

# Basic check to ensure valid values for required arguments
srun -p "${PARTITION}" -N "${NUM_NODES}" -w "${NODES}" true > /dev/null 2> /dev/null
if [[ $? != 0 ]]; then
    usage
    error_exit "Invalid values for one of --partition --nodes or --nodelist.\nCheck sinfo." 1
fi

# Enable all checks if none specified
if [[ -z $RUN_DCGMI ]] && [[ -z $RUN_NCCL ]]; then
    RUN_DCGMI=1
    RUN_NCCL=1
fi

mkdir -p "$RESULTS_DIR"

if [[ $RUN_DCGMI == 1 ]]; then
    echo "Starting DCGM Diagnostics..."
    JOBID=$(sbatch  -N "$NUM_NODES" \
            -p "$PARTITION" \
            -w "$NODES" \
            -o "$RESULTS_DIR"/dcgmi-%j.out \
            -W "$RELATIVE_PATH"dcgmi-diag.sh)
    JOBID=${JOBID##* } # Remove everything but the slurm job id from output of sbatch command
    grep -i "fail\|error" "$RESULTS_DIR"/dcgmi-"${JOBID}".out > /dev/null # Check log for failures
    FOUND=$?
    if [[ $FOUND == 0 ]]; then
        # One of the diagnostics failed
        FAILED=$(grep -i -P -o "srun: error: [^:]*" "$RESULTS_DIR"/dcgmi-"${JOBID}".out | cut -d':' -f3)
        echo -e "DCGM failed on the following nodes: \n $FAILED"
        echo "See results/cluster_validation/dcgmi-${JOBID}.out for more details"

        # Drain bad nodes if --drain-bad-nodes flag is set
        if [[ $DRAIN_BAD_NODES == 1 ]]; then
            echo "Draining failed nodes..."
            IFS=' ' read -ra FAILED_NODES <<< "$FAILED"
            for node in "${FAILED_NODES[@]}"; do
                echo "Draining node: $node"
                sudo scontrol update nodename="$node" state=drain reason="Failed DCGM"
            done
        fi

        exit 2
    elif [[ $FOUND == 1 ]]; then
        # Something else went wrong
        echo "DCGM diagnostics passing on all nodes!"
    else
        error_exit "DCGM failed to run properly..." 1
    fi
fi

if [[ $RUN_NCCL == 1 ]]; then
    # Build the nccl-tests
    NCCL_TEST_PATH=./nccl-tests/build/all_reduce_perf
    if [[ ! -f $NCCL_TEST_PATH ]]; then
        echo "Building the NCCL tests..."

        # Determine which script to run based on the machine type
        case "$MACHINE_TYPE" in
            a3-highgpu-8g)
                sbatch -N 1 -p "$PARTITION" -W "$RELATIVE_PATH"build-nccl-tests-a3.sh > /dev/null 2> /dev/null
                ;;
            a3-megagpu-8g)
                sbatch -N 1 -p "$PARTITION" -W "$RELATIVE_PATH"build-nccl-tests-a3mega.sh > /dev/null 2> /dev/null
                ;;
            a3-ultragpu-8g)
                sbatch -N 1 -p "$PARTITION" -W "$RELATIVE_PATH"build-nccl-tests-gib.sh > /dev/null 2> /dev/null
                ;;
            a4-highgpu-8g)
                sbatch -N 1 -p "$PARTITION" -W "$RELATIVE_PATH"build-nccl-tests-gib.sh > /dev/null 2> /dev/null
                ;;
            *)
                error_exit "Unsupported machine type: $MACHINE_TYPE" 2
                ;;
        esac

        if [[ -f $NCCL_TEST_PATH ]]; then
            echo "NCCL tests built successfully!"
        else
            error_exit "Failed to build NCCL tests." 2
        fi
    fi

    echo "Starting NCCL all_reduce_perf..."
    # Get list of nodes from sinfo to iterate over
    # and create pairwise all_reduce_perf tests
    NODES=$(sinfo --Node -h --partition="${PARTITION}" --state=idle --nodes="${NODES}")
    NODES_ARR=($NODES)
    ARR_LEN=${#NODES_ARR[@]}

    # Determine which script to run based on the machine type
    if [[ "$MACHINE_TYPE" == "a3-highgpu-8g" ]]; then
        script="./""$RELATIVE_PATH""nccl-a3.sh"
    elif [[ "$MACHINE_TYPE" == "a3-megagpu-8g" ]]; then
        script="./""$RELATIVE_PATH""nccl-a3mega.sh"
    elif [[ "$MACHINE_TYPE" == "a3-ultragpu-8g" ]]; then
        script="./""$RELATIVE_PATH""nccl-gib.sh"
    elif [[ "$MACHINE_TYPE" == "a4-highgpu-8g" ]]; then
        script="./""$RELATIVE_PATH""nccl-gib.sh"
    else
        echo "Unsupported MACHINE_TYPE: $MACHINE_TYPE"
        continue
    fi

    declare -a slurm_ids # ids for all the jobs launched, should be $NODES / 2
    for (( i = 0; i < $ARR_LEN - 1; i+=8 )); do
        j=$((i + 4))
        id=$(sbatch -N 2 \
                    -w "${NODES_ARR[$i]}","${NODES_ARR[$j]}" \
                    -p "$PARTITION" \
                    --parsable \
                    -o "$RESULTS_DIR"/nccl-gcp.sh_%j.log \
                    "$script")
        slurm_ids+=($id)
    done

    all_ids=$(join , "${slurm_ids[@]}")

    # Wait for NCCL jobs to finish
    srun -p "${PARTITION}" -N 1 --dependency="$all_ids" true > /dev/null 2> /dev/null

    LARGE_TEST_SIZE=10737418240
    nccl_pass=0
    for i in "${slurm_ids[@]}"; do
        CURR_NODES=$(grep -oP ' on \K[^\s]+' "$RESULTS_DIR"/nccl-gcp.sh_"${i}".log | sort -u) # Get the nodes in this test
        echo "CURR_NODES:  $CURR_NODES"
        CURR_BUSBW=$(grep "# Avg bus bandwidth" "$RESULTS_DIR"/nccl-gcp.sh_"${i}".log | cut -d':' -f2 | xargs | awk '{printf "%.0f", $1}')
        echo "CURR_BUSBW:  $CURR_BUSBW"

        # Check CURR_BUSBW is a number
        re='^[0-9]+(\.[0-9]+)?$'

        if ! [[ $CURR_BUSBW =~ $re ]]; then
            echo "NCCL failed to run properly on $CURR_NODES ..."
            echo "See results/cluster_validation/nccl-gcp.sh_${i}.log for more details"

            # Drain bad nodes if --drain-bad-nodes flag is set
            if [[ $DRAIN_BAD_NODES == 1 ]]; then
                echo "Draining failed nodes..."
                readarray -t NODES_TO_DRAIN <<< "$CURR_NODES"
                for node in "${NODES_TO_DRAIN[@]}"; do
                  # Trim whitespace
                  node=$(echo "$node" | xargs)
                  if [[ -n "$node" ]]; then
                      echo "Draining node: $node"
                      sudo scontrol update nodename="$node" state=drain reason="Failed NCCL"
                  fi
                done
            fi
            nccl_pass=0
        elif [[ $CURR_BUSBW -lt 100 ]]; then
            echo "Insufficient bus bandwidth on nodes on $CURR_NODES"
            echo "See results/cluster_validation/nccl-gcp.sh_${i}.log for more details"
            nccl_pass=0
        else
            nccl_pass=1
        fi
    done
    if [[ $nccl_pass == 0 ]]; then
        # Fail if any nodes had insufficient busbw or did not complete the test
        exit 2
    else
        echo "NCCL test passing on all nodes!"
    fi

fi