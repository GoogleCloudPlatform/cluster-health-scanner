from ramble.appkit import *


class NvidiaSmi(ExecutableApplication):
    """NVIDIA SMI application for cluster health scanner"""

    name = "nvidia-smi"
    maintainers("sarthakag", "simrankaurb")
    tags("diagnostics", "gpu")

    executable(
        "execute",
        template=["nvidia-smi -q"],
        use_mpi=False,
    )
    workload("local", executables=["execute"])

    figure_of_merit(
        "Driver Version",
        fom_regex=r"Driver Version\s*:\s*(?P<driver_version>[0-9.]+)",
        group_name="driver_version",
        units="",
        fom_type=FomType.INFO,
    )

    figure_of_merit(
        "CUDA Version",
        fom_regex=r"CUDA Version\s*:\s*(?P<cuda_version>[0-9.]+)",
        group_name="cuda_version",
        units="",
        fom_type=FomType.INFO,
    )

    figure_of_merit(
        "GPU Count",
        fom_regex=r"Attached GPUs\s*:\s*(?P<num_gpus>\d+)",
        group_name="num_gpus",
        units="",
        fom_type=FomType.INFO,
    )
