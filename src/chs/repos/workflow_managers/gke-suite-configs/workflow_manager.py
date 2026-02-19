import sys

from ramble.repository import ObjectTypes, paths

config_dir = (
    paths[ObjectTypes.workflow_managers]
    .repo_for_obj("gke-suite-configs")
    .dirname_for_object_name("gke-suite-configs")
)
sys.path.append(config_dir)
from nccl_config import NCCLConfig
from dcgm_config import DCGMConfig
from ramble.wmkit import *
from ramble.wm.chs.gke_jobset import GkeJobset


class GkeSuiteConfigs(GkeJobset):
    """GKE workflow manager that uses JobSet CRD"""

    name = "gke-suite-configs"

    maintainers("bytetwin")

    tags("workflow", "gke", "jobset")
    variable(
        "processes_per_node",
        default="1",
        description="Number of processes per node",
    )
    variable("n_ranks", default="1", description="Number of ranks")

    variant(
        "machine_type",
        default="{machine_type}",
        description="Machine type for the suite",
    )

    variable(
        "machine_type",
        default="a3-ultra",
        description="Machine type for the suite",
        values=["a3-high", "a3-mega", "a3-ultra", "a4", "a4x", "tpuv7x"],
    )

    with when("machine_type=a3-ultra"):
        variable(
            "gpus_per_node", default="8", description="Number of GPUs per node"
        )
        variable(
            "gpu_subnet_prefix",
            default="rdma",
            description="GPU subnet prefix for RDMA secondary networks",
        )
        variable(
            "pod_annotations_yaml",
            default=(
                "networking.gke.io/default-interface: 'eth0'\n"
                "networking.gke.io/interfaces: |\n"
                "  [\n"
                '    {"interfaceName":"eth0","network":"default"},'
                '    {"interfaceName":"eth2","network":"{gpu_subnet_prefix}-0"},'
                '    {"interfaceName":"eth3","network":"{gpu_subnet_prefix}-1"},'
                '    {"interfaceName":"eth4","network":"{gpu_subnet_prefix}-2"},'
                '    {"interfaceName":"eth5","network":"{gpu_subnet_prefix}-3"},'
                '    {"interfaceName":"eth6","network":"{gpu_subnet_prefix}-4"},'
                '    {"interfaceName":"eth7","network":"{gpu_subnet_prefix}-5"},'
                '    {"interfaceName":"eth8","network":"{gpu_subnet_prefix}-6"},'
                '    {"interfaceName":"eth9","network":"{gpu_subnet_prefix}-7"}'
                "  ]"
            ),
            description="A3 Ultra Pod network annotations",
        )
        variable(
            "node_selector_yaml",
            default="cloud.google.com/gke-nodepool: "
            "a3-ultragpu-8g-a3-ultragpu-pool",
            description="Node pool selector",
        )
        variable(
            "tolerations_yaml",
            default=(
                '- key: "nvidia.com/gpu"\n  operator: "Equal"\n'
                '  value: "present"\n  effect: "NoSchedule"'
            ),
            description="A3 Ultra tolerations",
        )
        variable(
            "extra_volumes_yaml",
            default='- name: shared-memory\n  emptyDir:\n    medium: "Memory"',
            description="Extra volumes for NCCL/A3",
        )
        variable(
            "extra_volume_mounts_yaml",
            default="- name: shared-memory\n  mountPath: /dev/shm",
            description="Extra mounts for NCCL/A3",
        )
        variable(
            "container_resources_yaml",
            default="limits:\n  nvidia.com/gpu: 8",
            description="GPU limits",
        )
        variable(
            "container_security_context_yaml",
            default="privileged: true",
            description="Privileged setup for NCCL on GKE",
        )

        with when("application_name=nvidia-dcgm"):
            DCGMConfig.standard()

        with when("application_name=nccl-tests"):
            NCCLConfig.a3_ultra()

        with when("application_name=chs-gke"):
            variable("mpi_command", default="", description="Empty MPI command")
            variable(
                "jobset_template",
                default="jobset_standard.yaml.tpl",
                description="Jobset Template for CHS",
            )
            variable(
                "container_image",
                default="ubuntu:22.04",
                description="Basic container image",
            )
