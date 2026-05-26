"""Supported resources for CHS."""

MACHINE_TYPE_RESOURCE_MAP = {
    "a3-ultra": {
        "instance_type": "a3-ultragpu-8g",
        "accelerators": 8,
        "resource_name": "nvidia.com/gpu",
    },
    "a3-mega": {
        "instance_type": "a3-megagpu-8g",
        "accelerators": 8,
        "resource_name": "nvidia.com/gpu",
    },
    "a4": {
        "instance_type": "a4-highgpu-8g",
        "accelerators": 8,
        "resource_name": "nvidia.com/gpu",
    },
    "a4x": {
        "instance_type": "a4x-highgpu-4g",
        "accelerators": 4,
        "resource_name": "nvidia.com/gpu",
    },
    "a4x-max": {
        "instance_type": "a4x-maxgpu-4g-metal",
        "accelerators": 4,
        "resource_name": "nvidia.com/gpu",
    },
    "tpu7x": {
        "instance_type": "tpu7x-standard-4t",
        "accelerators": 4,
        "resource_name": "google.com/tpu",
    },
    "tpuv6e": {
        "instance_type": "ct6e-standard-4t",
        "accelerators": 4,
        "resource_name": "google.com/tpu",
    },
}

# Default timeout and deadline values in seconds
DEFAULT_TIMEOUT = 1200
DEFAULT_DEADLINE = 4200
