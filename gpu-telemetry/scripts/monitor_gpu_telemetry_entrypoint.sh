#!/bin/bash

export VM_ID=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/id?alt=text" -H "Metadata-Flavor: Google")
export GPU_TELEMETRY_LMS=${GPU_TELEMETRY_LMS:=100}
export GPU_TELEMETRY_TO_CSV=${GPU_TELEMETRY_TO_CSV:="False"}
export CPU_CORES=${CPU_CORES:="all"}

numactl --physcpubind $CPU_CORES python /workspace/gpu_telemetry/scripts/monitor_gpu_telemetry.py -lms=$GPU_TELEMETRY_LMS -to_csv=$GPU_TELEMETRY_TO_CSV -vm_id=$VM_ID