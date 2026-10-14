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

"""Gang admission for Kubernetes Jobs."""

from collections.abc import Iterable, Sequence

_HOSTNAME_LABEL = "kubernetes.io/hostname"
_FINISHED_CONDITIONS = ("Complete", "Failed")


def _name(job: dict) -> str:
    return job["metadata"]["name"]


def _pod_spec(job: dict) -> dict:
    return job.get("spec", {}).get("template", {}).get("spec", {})


def _is_finished(job: dict) -> bool:
    conditions = job.get("status", {}).get("conditions") or []
    return any(
        c.get("type") in _FINISHED_CONDITIONS and c.get("status") == "True"
        for c in conditions
    )


def _is_suspended(job: dict) -> bool:
    return bool(job.get("spec", {}).get("suspend"))


def _required_hostnames(pod_spec: dict) -> frozenset[str] | None:
    required = (
        pod_spec.get("affinity", {})
        .get("nodeAffinity", {})
        .get("requiredDuringSchedulingIgnoredDuringExecution", {})
    )
    nodes = set()
    for term in required.get("nodeSelectorTerms", []):
        for expression in term.get("matchExpressions", []):
            if (
                expression.get("key") == _HOSTNAME_LABEL
                and expression.get("operator") == "In"
            ):
                nodes.update(expression.get("values", []))
    return frozenset(nodes) if nodes else None


def _target_nodes(job: dict) -> frozenset[str] | None:
    return _required_hostnames(_pod_spec(job))


def plan_admissions(jobs: list[dict], queue: Sequence[str]) -> list[str]:
    """Returns the names of suspended Jobs that can be started now."""
    queue_position = {name: index for index, name in enumerate(queue)}
    active = [j for j in jobs if not _is_finished(j)]
    used_nodes = set()
    all_nodes_used = False
    for job in active:
        if _is_suspended(job):
            continue
        nodes = _target_nodes(job)
        if nodes is None:
            all_nodes_used = True
        else:
            used_nodes |= nodes

    waiting = sorted(
        (j for j in active if _is_suspended(j) and _name(j) in queue_position),
        key=lambda j: queue_position[_name(j)],
    )

    admitted = []
    for job in waiting:
        nodes = _target_nodes(job)
        if nodes is None:
            can_start = not all_nodes_used and not used_nodes
            all_nodes_used = True
        else:
            can_start = not all_nodes_used and used_nodes.isdisjoint(nodes)
            used_nodes |= nodes
        if can_start:
            admitted.append(_name(job))
    return admitted


def pinned_nodes(
    jobs: list[dict], names: Iterable[str]
) -> dict[str, frozenset[str]]:
    """Returns the nodes that each named Job is pinned to."""
    wanted = set(names)
    return {
        _name(job): _target_nodes(job) or frozenset()
        for job in jobs
        if _name(job) in wanted
    }


def pod_nodes(pod: dict) -> frozenset[str]:
    """Returns the nodes that a pod is pinned to."""
    return _required_hostnames(pod.get("spec") or {}) or frozenset()
