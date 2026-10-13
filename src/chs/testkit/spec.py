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

"""The data a plugin is given and the data it hands back."""

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class TestRequest:
    variables: dict[str, Any]
    base_prefix: str
    workloads: tuple[str, ...] = ()
    workspace_root: str | None = None

    # Prevent pytest from collecting this dataclass as a test case.
    __test__ = False

    @property
    def scheduler(self) -> str:
        return self.variables.get("scheduler", "kubernetes")

    @property
    def machine_type(self) -> str:
        return self.variables.get("machine_type", "")

    @property
    def nodes(self) -> list[str]:
        nodes = self.variables.get("nodes")
        return nodes if isinstance(nodes, list) else []


@dataclasses.dataclass(frozen=True)
class NodeGrouping:
    groups: list[list[str]]
    node_locations: dict[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class ExperimentSpec:
    application: str
    workload: str
    name: str
    context: dict[str, Any]

    def to_ramble_context(self):
        from ramble.context import create_context_from_dict

        return create_context_from_dict(self.name, self.context)


STRIPPED_WORKSPACE_KEYS = frozenset(
    {
        "n_ranks",
        "processes_per_node",
        "n_nodes",
        "jobset_template",
        "mpi_command",
        "container_image",
        "srun_args",
    }
)


def build_test_request(
    base_vars: dict[str, Any],
    test_model: Any,
    base_prefix: str,
    workspace_root: str | None = None,
    suite_deadline: Any = None,
) -> TestRequest:
    variables = {
        key: value
        for key, value in base_vars.items()
        if key not in STRIPPED_WORKSPACE_KEYS
    }
    variables["chs_test_name"] = test_model.name
    if test_model.variables:
        variables.update(test_model.variables)

    if suite_deadline is None:
        suite_deadline = base_vars.get("deadline")
    timeout = _effective_timeout(suite_deadline, variables.get("timeout"))
    if timeout is not None:
        variables["timeout"] = timeout

    deadline = _as_int(suite_deadline) if suite_deadline else None
    if deadline is not None and "deadline" not in (test_model.variables or {}):
        variables["deadline"] = deadline

    workloads = test_model.workloads or ()
    if isinstance(workloads, str):
        workloads = (workloads,)

    return TestRequest(
        variables=variables,
        base_prefix=base_prefix,
        workloads=tuple(workloads),
        workspace_root=workspace_root,
    )


def _effective_timeout(suite_deadline: Any, test_timeout: Any) -> int | None:
    deadline = _as_int(suite_deadline) if suite_deadline else None
    if not test_timeout:
        return deadline
    timeout = _as_int(test_timeout)
    if timeout is None:
        return deadline
    if deadline is None:
        return timeout
    return min(deadline, timeout)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
