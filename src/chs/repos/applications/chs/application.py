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
from ramble.appkit import *


class TestModel:
    def __init__(self, name, workloads=None, variables=None):
        self.name = name
        self.workloads = workloads or []
        self.variables = variables or {}


class TestsModel:
    def __init__(self, tests):
        self.tests = tests


# Helper classes to support Ramble's CommentedMap/Seq in eval
class CommentedMap(dict):
    pass


class CommentedSeq(list):
    pass


def parse_tests_config(raw_input) -> TestsModel:
    try:
        data = raw_input
        if isinstance(raw_input, str):
            # Basic safety check: if it looks like a list, try eval
            # This is still risky but matches previous logic without ruamel dependency complexity
            # ideally we should use json.loads if we ensure input is json
            try:
                import json

                data = json.loads(raw_input)
            except json.JSONDecodeError:
                # Fallback to eval if it's a python string representation (e.g. from Ramble)
                context = {
                    "__builtins__": {},
                    "CommentedMap": CommentedMap,
                    "CommentedSeq": CommentedSeq,
                }
                data = eval(raw_input, context)

        # If data is a list of strings, convert to list of dicts
        if isinstance(data, list):
            cleaned_data = []
            for item in data:
                if isinstance(item, str):
                    cleaned_data.append(TestModel(name=item))
                elif isinstance(item, dict):
                    cleaned_data.append(
                        TestModel(
                            name=item.get("name"),
                            workloads=item.get("workloads"),
                            variables=item.get("variables"),
                        )
                    )
            return TestsModel(tests=cleaned_data)

        return TestsModel(tests=[])

    except Exception as e:
        logger.error(f"Failed to parse tests configuration: {e}")
        # Re-raise to be caught by caller
        raise e


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

        # Try to expand tests variable
        tests_str = self.expander.expand_var_name("tests", typed=True)
        # Handle empty list from Ramble
        if hasattr(tests_str, "__len__") and len(tests_str) == 0:
            tests_str = None
        if isinstance(tests_str, str) and tests_str.strip() == "[]":
            tests_str = None

        if not tests_str:
            logger.warn(
                f"No tests defined for {self.name}. Skipping experiments generation."
            )
            return

        try:
            tests_config = parse_tests_config(
                str(tests_str) if isinstance(tests_str, str) else tests_str
            )
        except Exception as e:
            logger.error(f"Failed to parse tests configuration: {e}")
            return

        for test in tests_config.tests:
            # Merge per-test variables with workspace variables, giving precedence to test variables
            # Create a context with merged variables
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

            # Update test_vars with per-test variables
            if test.variables:
                test_vars.update(test.variables)

            if test.name in ["nccl-cluster", "nccl-pairwise"]:
                workloads = (
                    test.workloads
                    if test.workloads
                    else ["all-gather", "all-reduce"]
                )
                application_name = "nccl-tests"

                for workload in workloads:
                    experiment_name = f"{self.expander.experiment_name}-{test.name}-{workload}"
                    contexts = TestHelper.nccl_test(
                        test_vars, experiment_name, configs, test.name
                    )
                    for context in contexts:
                        self.experiment_set.render_chained_experiments(
                            application_name, workload, context
                        )
            elif test.name == "dcgm":
                application_name = "nvidia-dcgm"
                experiment_name = self.expander.experiment_name

                context = TestHelper.dcgm_test(
                    test_vars, experiment_name, configs
                )
                self.experiment_set.render_chained_experiments(
                    application_name, "diag", context
                )
            elif test.name == "stream":
                application_name = "nvidia-stream"
                experiment_name = self.expander.experiment_name

                context = TestHelper.stream_test(
                    test_vars, experiment_name, configs
                )
                self.experiment_set.render_chained_experiments(
                    application_name, "standard", context
                )
            else:
                logger.warn(f"Unknown test {test.name} requested. Skipping.")
