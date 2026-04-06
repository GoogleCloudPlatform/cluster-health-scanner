"""Supported resources for CHS."""

# TODO: Add support for dynamincally fetching the list of
# supported machine types, suites, tests and schedulers.

SUPPORTED_MACHINE_TYPES = ["a3-ultra", "a4", "a3-mega", "a4x"]
SUPPORTED_TESTS = [
    "dcgm",
    "nccl-cluster",
    "nccl-pairwise",
    "nccl-single-node",
    "stream",
    "nvidia-smi",
    "gemm",
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
        ],
    },
    "gke": {
        "a3-mega": [
            "dcgm",
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
    },
}
