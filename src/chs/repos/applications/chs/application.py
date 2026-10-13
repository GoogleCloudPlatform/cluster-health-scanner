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

import logging

from chs.exceptions import ChsExecutionException
from chs.testkit import build_test_request, load_plugin

from ramble.appkit import *  # isort: skip # noqa: E402, F403

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
        values=["slurm", "kubernetes"],
    )

    variable(
        "machine_type",
        default="",
        description="Machine type to test",
    )

    executable(
        "cluster_health",
        template=[
            "echo 'Initiating cluster health diagnostics for {machine_type}'"
        ],
    )

    workload("cluster_health", executables=["cluster_health"])

    def create_experiment_chain(self):
        tests_str = self.expander.expand_var_name("tests", typed=True)
        if not tests_str:
            logger.warning(
                f"No tests defined for {self.name}. Skipping experiments generation."
            )
            return

        try:
            tests_config = parse_tests_config(tests_str)
        except Exception as e:
            logger.error(
                f"Failed to parse the test suite configuration file. Error: {e}"
            )
            raise ChsExecutionException(
                f"Failed to parse tests configuration: {e}"
            ) from e

        chs_run_id = self.expander.expand_var_name("chs_run_id", typed=True)
        base_prefix = (
            f"chs-{chs_run_id}" if chs_run_id else self.expander.experiment_name
        )
        suite_deadline = self.expander.expand_var_name("deadline", typed=True)
        base_vars = self.non_reserved_variables()

        for test in tests_config.tests:
            plugin = load_plugin(test.name)
            if plugin is None:
                logger.warning(f"Unknown test {test.name} requested. Skipping.")
                continue

            request = build_test_request(
                base_vars=base_vars,
                test_model=test,
                base_prefix=base_prefix,
                workspace_root=self.workspace.root,
                suite_deadline=suite_deadline,
            )

            for spec in plugin.build(request):
                self.experiment_set.render_experiment_set(
                    spec.application,
                    spec.workload,
                    spec.to_ramble_context(),
                    chained=True,
                )
