# GPU Telemetry Collection Script

## GPU Telemetry Metrics:

This script wraps an `nvidia-smi` command to log a set of GPU Telemetry metrics
to measure workload performance with sub-second latency. The metrics currently
collected are:

- serial
- pci.bus_id
- temperature.gpu
- utilization.gpu
- temperature.memory
- utilization.memory
- memory.used
- clocks.gr
- clocks.sm
- clocks.mem
- power.draw.instant
- power.draw.average
- clocks_throttle_reasons.hw_thermal_slowdown
- clocks_throttle_reasons.hw_power_brake_slowdown
- clocks_throttle_reasons.sw_thermal_slowdown
- clocks_throttle_reasons.sw_power_cap
- ecc.errors.corrected.volatile.total
- ecc.errors.corrected.aggregate.total
- ecc.errors.uncorrected.volatile.total
- ecc.errors.uncorrected.aggregate.total

## Building Docker Image:

First set the required environment variables and build the GPU Telemetry docker
image:

`docker build -f $DOCKERFILE_PATH/GPUTelemetry.Dockerfile -t $IMAGE_FULL $DOCKERFILE_PATH`

Alternatively, we provide a bash script to build the GPU Telemetry docker image
to be used as reference. After filling in the required fields, the command
can be run as follows:

`bash $DOCKERFILE_PATH/build_and_push_gpu_telemetry.sh`

## Example Docker Run:

Note: the mounted volume `/usr/share/telemetry` is used to determine the
termination criteria of the GPU Telemetry script (See `main()` in
`monitor_gpu_telemetry.py`). As such, the main workload being monitored should
write a file to this directory upon completion.

```
  sudo rm -rf /tmp/telemetry

  docker run --pull=always \\
    --detach \\
    --privileged \\
    --volume /var/lib/nvidia/lib64:/usr/local/nvidia/lib64 \\
    --volume /var/lib/nvidia/bin:/usr/local/nvidia/bin \\
    --device /dev/nvidia0:/dev/nvidia0 \\
    --device /dev/nvidia1:/dev/nvidia1 \\
    --device /dev/nvidia2:/dev/nvidia2 \\
    --device /dev/nvidia3:/dev/nvidia3 \\
    --device /dev/nvidia4:/dev/nvidia4 \\
    --device /dev/nvidia5:/dev/nvidia5 \\
    --device /dev/nvidia6:/dev/nvidia6 \\
    --device /dev/nvidia7:/dev/nvidia7 \\
    --device /dev/nvidia-uvm:/dev/nvidia-uvm \\
    --device /dev/nvidiactl:/dev/nvidiactl \\
    --env GPU_TELEMETRY_LMS=100 \\
    --env GPU_TELEMETRY_TO_CSV=False \\
    --volume /tmp/telemetry:/usr/share/telemetry \\
    {GPU_TELEMETRY_DOCKER_IMAGE} \\
```