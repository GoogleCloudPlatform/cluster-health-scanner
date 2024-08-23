
FROM nvcr.io/nvidia/pytorch:23.04-py3

RUN pip install absl-py

COPY scripts /workspace/gpu_telemetry/scripts

ENTRYPOINT ["/bin/bash", "/workspace/gpu_telemetry/scripts/monitor_gpu_telemetry_entrypoint.sh"]