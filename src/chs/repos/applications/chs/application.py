import sys
import os
import spack.util.spack_yaml as yaml
from ramble.repository import ObjectTypes, paths

config_dir = (
    paths[ObjectTypes.applications]
    .repo_for_obj("chs")
    .dirname_for_object_name("chs")
)
sys.path.append(config_dir)

from test_helper import TestHelper
from ramble.appkit import *


class Chs(ExecutableApplication):
    """Cluster health scanner for running diagnostics"""

    name = "chs"
    maintainers("bytetwin")

    tags("chs-diagnostics")

    variable(
        "scheduler",
        default="",
        description="Scheduler for the suite",
        values=["slurm", "gke"],
    )

    variable(
        "machine_type",
        default="",
        description="Machine type for the suite",
        values=["a3-high", "a3-ultra", "a4"],
    )

    executable(
        "cluster_health",
        template=[
            "echo 'Initiating cluster health diagnostics for {machine_type}'"
        ],
    )

    workload("cluster_health", executables=["cluster_health"])

    def _deep_merge(self, dest, src):
        for key, value in src.items():
            if (
                isinstance(value, dict)
                and key in dest
                and isinstance(dest[key], dict)
            ):
                self._deep_merge(dest[key], value)
            else:
                dest[key] = value
        return dest

    def create_experiment_chain(self, workspace):
        configs_dir = os.path.join(config_dir, "configs")
        configs = {}
        scheduler = self.expander.expand_var_name("scheduler", typed=True)
        common_cfg_path = os.path.join(configs_dir, "common.yaml")
        if os.path.exists(common_cfg_path):
            with open(common_cfg_path, "r") as f:
                configs = yaml.load(f) or {}

        if scheduler == "gke":
            gke_cfg_path = os.path.join(configs_dir, "gke.yaml")
            if os.path.exists(gke_cfg_path):
                with open(gke_cfg_path, "r") as f:
                    gke_configs = yaml.load(f) or {}
                self._deep_merge(configs, gke_configs)

        elif scheduler == "slurm":
            slurm_cfg_path = os.path.join(configs_dir, "slurm.yaml")
            if os.path.exists(slurm_cfg_path):
                with open(slurm_cfg_path, "r") as f:
                    slurm_configs = yaml.load(f) or {}
                self._deep_merge(configs, slurm_configs)

        tests = self.expander.expand_var_name("tests", typed=True)
        if not tests:
            logger.warn(
                f"No tests defined for {self.name}. Skipping experiments generation."
            )
            return

        for test in tests:
            test_vars = self.non_reserved_variables(
                workspace,
                remove_keys={
                    "n_ranks",
                    "processes_per_node",
                    "n_nodes",
                    "jobset_template",
                    "mpi_command",
                    "container_image",
                },
            )

            # Filter out any keys starting with "application::" as they are reserved
            # by Ramble and can cause validation errors in sub-experiments.
            test_vars = {
                k: v
                for k, v in test_vars.items()
                if not k.startswith("application::")
            }
            if test in ["nccl-cluster", "nccl-pairwise"]:
                workloads = ["all-gather", "all-reduce"]
                application_name = "nccl-tests"

                for workload in workloads:
                    experiment_name = (
                        f"{self.expander.experiment_name}-{test}-{workload}"
                    )
                    contexts = TestHelper.nccl_test(
                        test_vars, experiment_name, configs, test
                    )
                    for context in contexts:
                        self.experiment_set.render_chained_experiments(
                            application_name, workload, context
                        )
            elif test == "dcgm":
                application_name = "nvidia-dcgm"
                experiment_name = self.expander.experiment_name

                context = TestHelper.dcgm_test(
                    test_vars, experiment_name, configs
                )
                self.experiment_set.render_chained_experiments(
                    application_name, "diag", context
                )
            else:
                logger.warn(f"Unknown test {test} requested. Skipping.")
