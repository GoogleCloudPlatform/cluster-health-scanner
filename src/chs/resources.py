"""Supported resources for CHS."""

# TODO: Add support for dynamincally fetching the list of
# supported machine types, suites, tests and schedulers.

SUPPORTED_MACHINE_TYPES = [
    "a3-ultra",
    "a4",
    "a3-mega",
    "a4x",
    "tpu7x",
    "tpuv6e",
]
SUPPORTED_TESTS = [
    "dcgm",
    "nccl-cluster",
    "nccl-pairwise",
    "nccl-single-node",
    "stream",
    "nvidia-smi",
    "gemm",
    "tpu-info",
    "hbm",
    "host-device",
]
SUPPORTED_SCHEDULERS = ["slurm", "gke"]

TESTS_FOR_MACHINE_TYPE = {
    "a3-mega": [
        "dcgm",
        "nccl-cluster",
        "nccl-pairwise",
        "nccl-single-node",
        "stream",
        "nvidia-smi",
        "gemm",
    ],
    "a3-ultra": [
        "dcgm",
        "nccl-cluster",
        "nccl-pairwise",
        "nccl-single-node",
        "stream",
        "nvidia-smi",
        "gemm",
    ],
    "a4": [
        "dcgm",
        "nccl-cluster",
        "nccl-pairwise",
        "nccl-single-node",
        "stream",
        "nvidia-smi",
        "gemm",
    ],
    "a4x": [
        "dcgm",
        "stream",
        "nvidia-smi",
        "gemm",
        "nccl-cluster",
        "nccl-pairwise",
        "nccl-single-node",
    ],
}

# These are dummy values, will be replaced as we implement the test suite.
SUPPORTED_SUITES = {
    "lite": ["nvidia-smi", "dcgm", "gemm", "nccl-single-node"],
    "standard": [
        "nvidia-smi",
        "dcgm",
        "stream",
        "nccl-cluster",
        "nccl-pairwise",
    ],
    "burn-in": ["dcgm"],
}

# TODO: Create a single source of truth for all resources.
MAP_TESTS_BY_SCHEDULER_AND_MACHINE_TYPE = {
    "slurm": {
        "a3-mega": [
            "dcgm",
            "nccl-cluster",
            "nccl-pairwise",
            "nccl-single-node",
            "stream",
            "nvidia-smi",
            "gemm",
        ],
        "a3-ultra": [
            "dcgm",
            "nccl-cluster",
            "nccl-pairwise",
            "nccl-single-node",
            "stream",
            "nvidia-smi",
            "gemm",
        ],
        "a4": [
            "dcgm",
            "nccl-cluster",
            "nccl-pairwise",
            "nccl-single-node",
            "stream",
            "nvidia-smi",
            "gemm",
        ],
        "a4x": [
            "dcgm",
            "stream",
            "nvidia-smi",
            "gemm",
            "nccl-cluster",
            "nccl-pairwise",
            "nccl-single-node",
        ],
    },
    "gke": {
        "a3-mega": [
            "dcgm",
            "nccl-cluster",
            "nccl-pairwise",
            "nccl-single-node",
            "stream",
            "nvidia-smi",
            "gemm",
        ],
        "a3-ultra": [
            "dcgm",
            "nccl-cluster",
            "nccl-pairwise",
            "nccl-single-node",
            "stream",
            "nvidia-smi",
            "gemm",
        ],
        "a4": [
            "dcgm",
            "nccl-cluster",
            "nccl-pairwise",
            "nccl-single-node",
            "stream",
            "nvidia-smi",
            "gemm",
        ],
        "a4x": [
            "dcgm",
            "nvidia-smi",
            "stream",
            "gemm",
            "nccl-cluster",
            "nccl-pairwise",
            "nccl-single-node",
        ],
        "tpu7x": [
            "tpu-info",
            "gemm",
            "hbm",
            "host-device",
        ],
        "tpuv6e": [
            "tpu-info",
            "gemm",
            "hbm",
            "host-device",
        ],
    },
}

# Default timeout and deadline values in seconds
DEFAULT_TIMEOUT = 600
DEFAULT_DEADLINE = 2400
