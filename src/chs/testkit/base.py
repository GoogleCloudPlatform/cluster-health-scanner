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

"""The contract every CHS test implements"""

import abc
import copy
import logging
from typing import Any, ClassVar, Iterable

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
    config_key: ClassVar[str] = ""
    default_workloads: ClassVar[tuple[str, ...]] = ()
    experiment_label: ClassVar[str] = ""
    experiment_suffix: ClassVar[str] = ""
    workload_in_experiment_name: ClassVar[bool] = True
    extra_context: ClassVar[dict[str, Any] | None] = None
    extra_sbatch_headers: ClassVar[dict[str, Any] | None] = None
    benchmark_config: ClassVar[str | None] = None
    minimum_nodes: ClassVar[int] = 0
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
    def config_name(self) -> str:
        return self.config_key or self.application

    @property
    def label(self) -> str:
        return self.experiment_label or self.name

    @property
    def accepts_workload_override(self) -> bool:
        return self.workload_in_experiment_name

    def resolve_workloads(self, request: TestRequest) -> tuple[str, ...]:
        workloads = request.workloads
        if not workloads or not self.accepts_workload_override:
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

    def machine_configs(self, request: TestRequest) -> dict[str, Any] | None:
        return copy.deepcopy(request.configs)

    def prepare_run(self, request: TestRequest) -> Any:
        del request
        return None

    def build(self, request: TestRequest) -> list[ExperimentSpec]:
        if not self._has_enough_nodes(request):
            return []

        prepared = self.prepare_run(request)
        specs: list[ExperimentSpec] = []
        for workload in self.resolve_workloads(request):
            experiment_name = self.experiment_name(
                request.base_prefix, workload
            )
            variables = copy.deepcopy(request.variables)
            # The job name identifies the submitted job and stays common to all
            # experiments of this workload, even where the experiment names
            # below are further qualified by node, pair or slice.
            variables["job_name"] = experiment_name
            variables["chs_test_name"] = self.name
            self.prepare_variables(variables, workload, request)

            for name, context in self.build_contexts(
                variables, experiment_name, workload, request, prepared
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
        if not self.minimum_nodes:
            return True
        available = len(request.nodes)
        if available < self.minimum_nodes:
            logger.warning(
                "The '%s' diagnostic requires a minimum of %d nodes, but only "
                "%d are available. Skipping this test.",
                self.name,
                self.minimum_nodes,
                available,
            )
            return False
        return True

    def prepare_variables(
        self, variables: dict, workload: str, request: TestRequest
    ) -> None:
        """Adjusts the variables before any context is built.

        Override to derive values that depend on the workload. Called once per
        workload, on a copy that no other workload shares.
        """

    def finalize_context(
        self, context: dict, workload: str, request: TestRequest
    ) -> None:
        """Adjusts a context after it has been built.

        Override for changes that need the merged configuration in place, such
        as substituting values into content the machine configuration supplied.
        Called once per produced context.
        """

    @abc.abstractmethod
    def build_contexts(
        self,
        variables: dict,
        experiment_name: str,
        workload: str,
        request: TestRequest,
        prepared: Any = None,
    ) -> Iterable[tuple[str, dict]]:
        """Builds the contexts for one workload.

        Returns pairs of experiment name and context dictionary. The number of
        pairs is the test's own decision: one per node group, one per slice, or
        a single context that a matrix expands later. ``prepared`` is whatever
        :meth:`prepare_run` returned for this run.
        """


class NodeTestPlugin(BaseTestPlugin):
    """A test that runs on each node independently.

    One context is produced whose experiment name ends in ``{node}``. Ramble
    matrixes it over the node list, so the number of runs follows the size of
    the cluster without the plugin counting anything.
    """

    def build_contexts(
        self, variables, experiment_name, workload, request, prepared=None
    ):
        del workload, prepared
        context = build_node_context(
            variables,
            self.machine_configs(request),
            self.config_name,
            extra_context=self.extra_context,
            extra_sbatch_headers=self.extra_sbatch_headers,
            benchmark_config=self.benchmark_config,
        )
        return [(f"{experiment_name}-{{node}}", context)]


class ClusterTestPlugin(BaseTestPlugin):
    """A test that runs across a group of nodes at once.

    By default the group is the whole allocation. Tests that measure several
    groups, such as pairs of nodes, override :meth:`grouping` and get one
    experiment per group.
    """

    def grouping(self, request: TestRequest) -> NodeGrouping:
        return NodeGrouping(groups=[request.nodes])

    def prepare_run(self, request: TestRequest) -> NodeGrouping:
        return self.grouping(request)

    def group_suffix(self, nodes: list[str]) -> str:
        del nodes
        return ""

    def group_domain_type(
        self, nodes: list[str], grouping: NodeGrouping
    ) -> str | None:
        del nodes, grouping
        return None

    def build_contexts(
        self, variables, experiment_name, workload, request, prepared=None
    ):
        del workload
        grouping = prepared if prepared is not None else self.grouping(request)
        contexts = []
        for nodes in grouping.groups:
            context = build_cluster_context(
                variables,
                self.machine_configs(request),
                self.config_name,
                nodes_list=nodes,
                extra_context=self.extra_context,
                extra_sbatch_headers=self.extra_sbatch_headers,
                benchmark_config=self.benchmark_config,
                domain_type=self.group_domain_type(nodes, grouping),
            )
            contexts.append(
                (f"{experiment_name}{self.group_suffix(nodes)}", context)
            )
        return contexts


class SliceTestPlugin(BaseTestPlugin):
    """A test that runs across a whole TPU slice at once.

    On Kubernetes this is one experiment per slice, because a slice is the unit
    the inter-chip interconnect is wired across. On Slurm it collapses to a
    single node-matrix experiment.
    """

    def build_contexts(
        self, variables, experiment_name, workload, request, prepared=None
    ):
        del workload, prepared
        return build_slice_contexts(
            variables,
            experiment_name,
            self.machine_configs(request),
            self.config_name,
            extra_context=self.extra_context,
            extra_sbatch_headers=self.extra_sbatch_headers,
            benchmark_config=self.benchmark_config,
        )
