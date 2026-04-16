"""Supported resources for CHS."""

MACHINE_TYPE_RESOURCE_MAP = {
    "a3-ultra": {"instance_type": "a3-ultragpu-8g", "gpus": 8},
    "a3-mega": {"instance_type": "a3-megagpu-8g", "gpus": 8},
    "a4": {"instance_type": "a4-highgpu-8g", "gpus": 8},
    "a4x": {"instance_type": "a4x-highgpu-4g", "gpus": 4},
    "tpu7x": {"instance_type": "tpu7x-standard-4t", "gpus": 4},
    "tpuv6e": {"instance_type": "ct6e-standard-4t", "gpus": 4},
}

# Default timeout and deadline values in seconds
DEFAULT_TIMEOUT = 600
DEFAULT_DEADLINE = 2400
