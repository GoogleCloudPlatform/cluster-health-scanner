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

"""Base plugin classes for CHS diagnostic tests."""

import abc
import copy
import logging
import os
from typing import Any, ClassVar, Iterable

from chs.testkit import config as chs_config
from chs.testkit.context import (
    build_cluster_context,
    build_node_context,
    build_slice_contexts,
)
from chs.testkit.spec import ExperimentSpec, NodeGrouping, TestRequest

logger = logging.getLogger(__name__)


class BaseTestPlugin(abc.ABC):
    name: ClassVar[str] = ""
    application: ClassVar[str] = ""
    default_workloads: ClassVar[tuple[str, ...]] = ()
    experiment_label: ClassVar[str] = ""
    experiment_suffix: ClassVar[str] = ""
    workload_in_experiment_name: ClassVar[bool] = True
    extra_context: ClassVar[dict[str, Any] | None] = None
    extra_sbatch_headers: ClassVar[dict[str, Any] | None] = None
    benchmark_config: ClassVar[str | None] = None
    min_nodes_to_run: ClassVar[int] = 0
    category: ClassVar[str] = ""
    description: ClassVar[str] = ""
    long_description: ClassVar[str] = ""

    def __init__(self):
        if not self.name:
            raise ValueError(f"{type(self).__name__} must declare a test name.")
        if not self.application:
            raise ValueError(
                f"Test plugin '{self.name}' must declare a Ramble application."
            )
        if not self.default_workloads:
            raise ValueError(
                f"Test plugin '{self.name}' must declare at least one default "
                "workload."
            )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"

    @property
    def benchmarks_dir(self) -> str:
        return os.path.join(
            chs_config.TESTS_ROOT, self.application, "benchmarks"
        )

    @property
    def label(self) -> str:
        return self.experiment_label or self.name

    def resolve_workloads(self, request: TestRequest) -> tuple[str, ...]:
        workloads = request.workloads
        if not workloads or not self.workload_in_experiment_name:
            return tuple(self.default_workloads)
        if isinstance(workloads, str):
            return (workloads,)
        return tuple(workloads)

    def experiment_name(self, base_prefix: str, workload: str) -> str:
        """Builds the experiment name for one workload."""
        name = f"{base_prefix}-{self.label}"
        if self.workload_in_experiment_name:
            name = f"{name}-{workload}"
        return f"{name}{self.experiment_suffix}"

    def build(self, request: TestRequest) -> list[ExperimentSpec]:
        if not self._has_enough_nodes(request):
            return []

        specs: list[ExperimentSpec] = []
        for workload in self.resolve_workloads(request):
            experiment_name = self.experiment_name(
                request.base_prefix, workload
            )
            variables = copy.deepcopy(request.variables)
            variables["job_name"] = experiment_name
            variables["chs_test_name"] = self.name
            self.prepare_variables(variables, workload, request)

            for name, context in self.build_contexts(
                variables, experiment_name, request
            ):
                self.finalize_context(context, workload, request)
                specs.append(
                    ExperimentSpec(
                        application=self.application,
                        workload=workload,
                        name=name,
                        context=context,
                    )
                )
        return specs

    def _has_enough_nodes(self, request: TestRequest) -> bool:
        if not self.min_nodes_to_run:
            return True
        available = len(request.nodes)
        if available < self.min_nodes_to_run:
            logger.warning(
                "The '%s' diagnostic requires a minimum of %d nodes, but only "
                "%d are available. Skipping this test.",
                self.name,
                self.min_nodes_to_run,
                available,
            )
            return False
        return True

    def prepare_variables(
        self, variables: dict, workload: str, request: TestRequest
    ) -> None:
        """Adjusts the variables before any context is built."""

    def finalize_context(
        self, context: dict, workload: str, request: TestRequest
    ) -> None:
        """Adjusts a context after it has been built."""

    @abc.abstractmethod
    def build_contexts(
        self,
        variables: dict,
        experiment_name: str,
        request: TestRequest,
    ) -> Iterable[tuple[str, dict]]:
        """Builds the (experiment_name, context_dict) pairs for one workload."""


class SingleNodePlugin(BaseTestPlugin):
    """A test that runs on each node independently.

    Produces a single context templated with ``{node}`` that Ramble expands
    across all nodes in the allocation.
    """

    def build_contexts(self, variables, experiment_name, request):
        context = build_node_context(
            variables,
            extra_context=self.extra_context,
            extra_sbatch_headers=self.extra_sbatch_headers,
            benchmark_config=self.benchmark_config,
            benchmarks_dir=self.benchmarks_dir,
        )
        return [(f"{experiment_name}-{{node}}", context)]


class MultiNodePlugin(BaseTestPlugin):
    """A test that runs across a group of nodes at once.

    By default the group is the whole allocation. Tests that measure several
    groups, such as pairs of nodes, override :meth:`grouping` and get one
    experiment per group.
    """

    def grouping(self, request: TestRequest) -> NodeGrouping:
        return NodeGrouping(groups=[request.nodes])

    def group_suffix(self, nodes: list[str], grouping: NodeGrouping) -> str:
        return ""

    def group_domain_type(
        self, nodes: list[str], grouping: NodeGrouping
    ) -> str | None:
        return None

    def build_contexts(self, variables, experiment_name, request):
        grouping = self.grouping(request)
        contexts = []
        for nodes in grouping.groups:
            context = build_cluster_context(
                variables,
                nodes_list=nodes,
                extra_context=self.extra_context,
                extra_sbatch_headers=self.extra_sbatch_headers,
                benchmark_config=self.benchmark_config,
                benchmarks_dir=self.benchmarks_dir,
                domain_type=self.group_domain_type(nodes, grouping),
            )
            contexts.append(
                (
                    f"{experiment_name}{self.group_suffix(nodes, grouping)}",
                    context,
                )
            )
        return contexts


class SlicePlugin(BaseTestPlugin):
    """A test that runs across a TPU slice.

    On Kubernetes this creates one experiment per slice; on Slurm it creates a
    single node-templated experiment. Multi-host plugins that need an entire
    slice set `requires_whole_slice = True`.
    """

    requires_whole_slice: ClassVar[bool] = False

    def build_contexts(self, variables, experiment_name, request):
        return build_slice_contexts(
            variables,
            experiment_name,
            extra_context=self.extra_context,
            extra_sbatch_headers=self.extra_sbatch_headers,
            benchmark_config=self.benchmark_config,
            benchmarks_dir=self.benchmarks_dir,
        )
