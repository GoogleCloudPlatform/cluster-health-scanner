# Copyright 2026 Google LLC
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

"""Supported resources for CHS."""

MACHINE_TYPE_RESOURCE_MAP = {
    "a3-ultragpu-8g": {
        "accelerators": 8,
        "resource_name": "nvidia.com/gpu",
    },
    "a3-megagpu-8g": {
        "accelerators": 8,
        "resource_name": "nvidia.com/gpu",
    },
    "a4-highgpu-8g": {
        "accelerators": 8,
        "resource_name": "nvidia.com/gpu",
    },
    "a4x-highgpu-4g": {
        "accelerators": 4,
        "resource_name": "nvidia.com/gpu",
    },
    "a4x-maxgpu-4g-metal": {
        "accelerators": 4,
        "resource_name": "nvidia.com/gpu",
    },
    "tpu7x-standard-4t": {
        "accelerators": 4,
        "resource_name": "google.com/tpu",
    },
    "ct6e-standard-4t": {
        "accelerators": 4,
        "resource_name": "google.com/tpu",
    },
}


def normalize_machine_type(raw_machine_type: str) -> str:
    """Normalizes a GCE machine type (extracts name from URL if needed)."""
    if not raw_machine_type:
        return "unknown"
    machine_name = raw_machine_type.split("/")[-1]  # Handle GCE URL
    for chs_type in MACHINE_TYPE_RESOURCE_MAP:
        if machine_name.startswith(chs_type):
            return chs_type
    return machine_name


def is_tpu(machine_type: str) -> bool:
    """Checks if the given machine type is a TPU."""
    return (
        MACHINE_TYPE_RESOURCE_MAP.get(machine_type, {}).get("resource_name")
        == "google.com/tpu"
    )


# Default timeout and deadline values in seconds
DEFAULT_TIMEOUT = 1200
DEFAULT_DEADLINE = 4200
