from copy import deepcopy
import logging
import sys
import os
import json
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

logger = logging.getLogger(__name__)


class TestModel:
    def __init__(self, name, workloads=None, variables=None):
        self.name = name
        self.workloads = workloads or []
        self.variables = variables or {}


class TestsModel:
    def __init__(self, tests):
        self.tests = tests


class CommentedMap(dict):
    pass


class CommentedSeq(list):
    pass


def parse_tests_config(raw_input) -> TestsModel:
    data = raw_input
    if isinstance(raw_input, str):
        context = {
            "__builtins__": {},
            "CommentedMap": CommentedMap,
            "CommentedSeq": CommentedSeq,
        }
        data = eval(raw_input, context)

    if isinstance(data, list):
        cleaned_data = [
            TestModel(
                name=item.get("name"),
                workloads=item.get("workloads"),
                variables=item.get("variables"),
            )
            for item in data
        ]
        return TestsModel(tests=cleaned_data)

    return TestsModel(tests=[])


def is_tpu(machine_type: str) -> bool:
    """Returns True if the machine type is a TPU."""
    return machine_type.startswith("tpu") if machine_type else False


class Chs(ExecutableApplication):
    """Cluster health scanner for running diagnostics"""

    name = "chs"
    maintainers("bytetwin")

    variable("tests", default=[], description="Specific tests to run")

    tags("chs-diagnostics")

    variable(
        "scheduler",
        default="slurm",
        description="Scheduler for the suite",
        values=["slurm", "gke"],
    )

    variable(
        "machine_type",
        default="",
        description="Machine type to test",
        values=[
            "a3-high",
            "a3-ultra",
            "a4",
            "a3-mega",
            "a4x",
            "tpu7x",
            "tpuv6e",
        ],
    )

    executable(
        "cluster_health",
        template=[
            "echo 'Initiating cluster health diagnostics for {machine_type}'"
        ],
    )

    workload("cluster_health", executables=["cluster_health"])

    def _deep_merge(self, dest, src):
        if not isinstance(src, dict) or not isinstance(dest, dict):
            raise ValueError(
                f"Expected both source and destination to be dictionaries, "
                f"got src={type(src).__name__} and dest={type(dest).__name__}"
            )
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

        tests_raw = self.variables.get("tests", [])
        tests_expanded = self.expander.expand_var(tests_raw, typed=True)

        if not tests_expanded:
            logger.warn(
                f"No tests defined for {self.name}. Skipping experiments generation."
            )
            return

        try:
            tests_config = parse_tests_config(tests_expanded)
        except Exception as e:
            logger.error(f"Failed to parse tests configuration: {e}")
            raise ChsExecutionException(
                f"Failed to parse tests configuration: {e}"
            ) from e

        for test in tests_config.tests:
            test_vars = self.non_reserved_variables(
                workspace,
                remove_keys={
                    "n_ranks",
                    "processes_per_node",
                    "n_nodes",
                    "jobset_template",
                    "mpi_command",
                    "container_image",
                    "srun_args",
                },
            )

            if test.variables:
                test_vars.update(test.variables)

            if test.name in [
                "nccl-cluster",
                "nccl-pairwise",
                "nccl-single-node",
            ]:
                topology = None
                if test.name == "nccl-pairwise":
                    workspace_dir = self.expander.expand_var_name(
                        "workspace_dir"
                    )
                    topo_path = os.path.join(workspace_dir, "topology.json")

                    if os.path.exists(topo_path):
                        try:
                            with open(topo_path, "r") as f:
                                topology = json.load(f)
                        except json.JSONDecodeError as e:
                            logger.warning(
                                "Failed to parse topology JSON from %s: %s",
                                topo_path,
                                e,
                            )
                        except IOError as e:
                            logger.warning(
                                "Failed to read topology file %s: %s",
                                topo_path,
                                e,
                            )

                workloads = (
                    test.workloads
                    if test.workloads
                    else ["all-gather", "all-reduce"]
                )
                application_name = "nccl-tests"

                for workload in workloads:
                    experiment_name = f"{self.expander.experiment_name}-{test.name}-{workload}"
                    local_vars = deepcopy(test_vars)
                    local_vars["job_name"] = experiment_name
                    contexts = TestHelper.nccl_test(
                        local_vars,
                        experiment_name,
                        configs,
                        test.name,
                        topology,
                    )
                    for context in contexts:
                        self.experiment_set.render_chained_experiments(
                            application_name, workload, context
                        )
            elif test.name == "dcgm":
                application_name = "nvidia-dcgm"
                experiment_name = f"{self.expander.experiment_name}-{test.name}-level-{{diag_level}}"

                local_vars = deepcopy(test_vars)
                local_vars["job_name"] = experiment_name

                context = TestHelper.dcgm_test(
                    local_vars, experiment_name, deepcopy(configs)
                )
                self.experiment_set.render_chained_experiments(
                    application_name, "diag", context
                )
            elif test.name == "stream":
                application_name = "nvidia-stream"
                experiment_name = f"{self.expander.experiment_name}-{test.name}"

                local_vars = deepcopy(test_vars)
                local_vars["job_name"] = experiment_name

                context = TestHelper.stream_test(
                    local_vars, experiment_name, deepcopy(configs)
                )
                self.experiment_set.render_chained_experiments(
                    application_name, "standard", context
                )
            elif test.name == "nvidia-smi":
                application_name = "nvidia-smi"
                experiment_name = f"{self.expander.experiment_name}-{test.name}"

                local_vars = deepcopy(test_vars)
                local_vars["job_name"] = experiment_name

                context = TestHelper.nvidia_smi_test(
                    local_vars, experiment_name, deepcopy(configs)
                )
                self.experiment_set.render_chained_experiments(
                    application_name, "local", context
                )
            elif test.name == "gemm":
                machine_type = self.expander.expand_var_name(
                    "machine_type", typed=True
                )
                is_tpu_machine = is_tpu(machine_type)
                application_name = (
                    "tpu-gemm" if is_tpu_machine else "nvidia-gemm"
                )
                workloads = test.workloads if test.workloads else ["check"]

                for workload in workloads:
                    experiment_name = f"{self.expander.experiment_name}-{application_name}-{workload}"
                    local_vars = deepcopy(test_vars)
                    local_vars["job_name"] = experiment_name
                    if is_tpu_machine:
                        context = TestHelper.tpu_gemm_test(
                            local_vars, experiment_name, deepcopy(configs)
                        )
                    else:
                        context = TestHelper.gemm_test(
                            local_vars, experiment_name, deepcopy(configs)
                        )
                    self.experiment_set.render_chained_experiments(
                        application_name, workload, context
                    )
            elif test.name == "hbm":
                application_name = "tpu-hbm"
                experiment_name = (
                    f"{self.expander.experiment_name}-{application_name}"
                )

                local_vars = deepcopy(test_vars)
                local_vars["job_name"] = experiment_name

                context = TestHelper.tpu_hbm_test(
                    local_vars, experiment_name, deepcopy(configs)
                )
                self.experiment_set.render_chained_experiments(
                    application_name, "check", context
                )
            elif test.name == "host-device":
                application_name = "tpu-host-device"
                experiment_name = (
                    f"{self.expander.experiment_name}-{application_name}"
                )

                local_vars = deepcopy(test_vars)
                local_vars["job_name"] = experiment_name

                context = TestHelper.tpu_host_device_test(
                    local_vars, experiment_name, deepcopy(configs)
                )
                self.experiment_set.render_chained_experiments(
                    application_name, "check", context
                )
            elif test.name == "collectives":
                application_name = "tpu-collectives"
                workloads = (
                    test.workloads
                    if test.workloads
                    else ["all_gather", "psum", "all_to_all"]
                )

                for workload in workloads:
                    experiment_name = f"{self.expander.experiment_name}-{application_name}-{workload}"
                    local_vars = deepcopy(test_vars)
                    local_vars["job_name"] = experiment_name

                    config_file = f"collectives_{workload}.yaml"
                    local_vars["benchmarks_yaml_file"] = config_file

                    context = TestHelper.tpu_collectives_test(
                        local_vars, experiment_name, deepcopy(configs)
                    )
                    self.experiment_set.render_chained_experiments(
                        application_name, workload, context
                    )
            elif test.name == "collectives-single-node":
                application_name = "tpu-collectives"
                workloads = (
                    test.workloads
                    if test.workloads
                    else ["all_gather", "psum", "all_to_all"]
                )

                for workload in workloads:
                    experiment_name = f"{self.expander.experiment_name}-{application_name}-{workload}"
                    local_vars = deepcopy(test_vars)
                    local_vars["job_name"] = experiment_name

                    config_file = f"collectives_{workload}.yaml"
                    local_vars["benchmarks_yaml_file"] = config_file

                    context = TestHelper.tpu_collectives_single_node_test(
                        local_vars, experiment_name, deepcopy(configs)
                    )
                    self.experiment_set.render_chained_experiments(
                        application_name, workload, context
                    )

            elif test.name == "tpu-info":
                application_name = "tpu-info"
                experiment_name = f"{self.expander.experiment_name}-{test.name}"

                local_vars = deepcopy(test_vars)
                local_vars["job_name"] = experiment_name

                context = TestHelper.tpu_info_test(
                    local_vars, experiment_name, deepcopy(configs)
                )
                self.experiment_set.render_chained_experiments(
                    application_name, "local", context
                )
            else:
                logger.warn(f"Unknown test {test.name} requested. Skipping.")
