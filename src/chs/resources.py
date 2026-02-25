"""Supported resources for CHS."""

# TODO: Add support for dynamincally fetching the list of
# supported machine types, suites, tests and schedulers.

SUPPORTED_MACHINE_TYPES = ["a3-ultra"]
SUPPORTED_TESTS = [
    "dcgm",
    "nccl-cluster",
    "nccl-pairwise",
]
SUPPORTED_SCHEDULERS = ["slurm", "gke"]

TESTS_FOR_MACHINE_TYPE = {
    "a3-ultra": [
        "dcgm",
        "nccl-cluster",
        "nccl-pairwise",
    ],
}

# These are dummy values, will be replaced as we implement the test suite.
SUPPORTED_SUITES = {
    "lite": ["dcgm"],
    "standard": [
        "dcgm",
        "nccl-cluster",
    ],
    "burn-in": [
        "dcgm",
        "nccl-cluster",
        "nccl-pairwise",
    ],
}
