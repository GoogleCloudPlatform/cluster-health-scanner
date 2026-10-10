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

"""Gang admission for Kubernetes JobSets."""

from collections.abc import Sequence

_HOSTNAME_LABEL = "kubernetes.io/hostname"


def _is_finished(jobset: dict) -> bool:
    conditions = jobset.get("status", {}).get("conditions", [])
    return any(
        c.get("type") in ("Completed", "Failed") and c.get("status") == "True"
        for c in conditions
    )


def _is_suspended(jobset: dict) -> bool:
    return bool(jobset.get("spec", {}).get("suspend"))


def _target_nodes(jobset: dict) -> frozenset[str] | None:
    nodes = set()
    for replicated_job in jobset.get("spec", {}).get("replicatedJobs", []):
        pod_spec = (
            replicated_job.get("template", {})
            .get("spec", {})
            .get("template", {})
            .get("spec", {})
        )
        required = (
            pod_spec.get("affinity", {})
            .get("nodeAffinity", {})
            .get("requiredDuringSchedulingIgnoredDuringExecution", {})
        )
        for term in required.get("nodeSelectorTerms", []):
            for expression in term.get("matchExpressions", []):
                if (
                    expression.get("key") == _HOSTNAME_LABEL
                    and expression.get("operator") == "In"
                ):
                    nodes.update(expression.get("values", []))
    return frozenset(nodes) if nodes else None


def plan_admissions(jobsets: list[dict], queue: Sequence[str]) -> list[str]:
    """Returns the names of suspended JobSets that can be started now."""
    queue_position = {name: index for index, name in enumerate(queue)}
    active = [j for j in jobsets if not _is_finished(j)]
    used_nodes = set()
    all_nodes_used = False
    for jobset in active:
        if _is_suspended(jobset):
            continue
        nodes = _target_nodes(jobset)
        if nodes is None:
            all_nodes_used = True
        else:
            used_nodes |= nodes

    waiting = sorted(
        (
            j
            for j in active
            if _is_suspended(j) and j["metadata"]["name"] in queue_position
        ),
        key=lambda j: queue_position[j["metadata"]["name"]],
    )

    admitted = []
    for jobset in waiting:
        nodes = _target_nodes(jobset)
        if nodes is None:
            can_start = not all_nodes_used and not used_nodes
            all_nodes_used = True
        else:
            can_start = not all_nodes_used and used_nodes.isdisjoint(nodes)
            used_nodes |= nodes
        if can_start:
            admitted.append(jobset["metadata"]["name"])
    return admitted
