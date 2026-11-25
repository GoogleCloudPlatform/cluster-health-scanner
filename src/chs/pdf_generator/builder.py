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

"""Turns a Bill of Health into the PDF report model.

Nothing is judged here: host and test statuses come from
report_data.node_health (the HTML report's judgement), values and thresholds
from the plugin-filtered validation rows, and events from
aggregate_boh_summary. The builder arranges them into physical group sections
(CHS Subblock, CHS Block, CHS Slice) and ungrouped Hosts.
"""

import collections

from chs.core import report_data
from chs.core.plugin_manager import PluginManager
from chs.core.summary_aggregator import aggregate_boh_summary
from chs.pdf_generator import hardware_ids as hardware_ids_lib
from chs.pdf_generator import labels as labels_lib
from chs.pdf_generator import metrics
from chs.pdf_generator import model
from chs.pdf_generator.labels import TestKey
from chs.protos import bill_of_health_pb2 as pb
from chs.testkit import requires_whole_slice

_HOST_STATUS = {
    report_data.NodeStatus.HEALTHY: metrics.HEALTHY,
    report_data.NodeStatus.DEGRADED: metrics.DEGRADED,
    report_data.NodeStatus.POTENTIALLY_DEGRADED: metrics.POTENTIALLY_DEGRADED,
    report_data.NodeStatus.INCOMPLETE: metrics.INCOMPLETE,
}
_EVENT_TYPES = {"FAIL": "Error", "WARNING": "Warning"}  # aggregator's types
_PHYSICAL_GROUP_RULES = (
    (("tpu7x", "v7x"), "4x4x4", "Subblock"),
    (("v6e", "ct6e"), "16x16", "Block"),
)
_GROUP_SCOPES = {
    "Subblock": hardware_ids_lib.SUBBLOCK_SCOPE,
    "Block": hardware_ids_lib.BLOCK_SCOPE,
}
_SYSTEM_TYPE_SCHEDULERS = {
    "SLURM_CLUSTER": ("Slurm", "Slurm Host", "Partition"),
    "KUBERNETES_CLUSTER": ("GKE", "GKE Node", "Nodepool"),
    "STANDALONE_CLUSTER": ("Standalone", "VM Host", ""),
}
# (overview label, system_info.metadata key), shown only when reported.
_SOFTWARE_VERSIONS = (
    ("GKE version", "gke_version"),
    ("JAX version", "jax_version"),
    ("libtpu version", "libtpu_version"),
)


def _group_kind(machine_type: str, topology: str) -> str:
    machine = machine_type.lower()
    for tokens, expected_topo, kind in _PHYSICAL_GROUP_RULES:
        if topology == expected_topo and any(tok in machine for tok in tokens):
            return kind
    return "Slice"


def _collect_instance_ids(
    boh: pb.BillOfHealth, host_meta: dict[str, dict[str, str]]
) -> dict[str, str]:
    ids: dict[str, str] = {}
    for r in boh.test_results:
        for t in r.target_resources:
            if t.node_name and t.instance_id and t.instance_id != t.node_name:
                ids.setdefault(t.node_name, t.instance_id)
    for node, meta in host_meta.items():
        inst = meta.get("instance_id", "")
        if inst and inst != node:
            ids.setdefault(node, inst)
    return ids


def _claim_slices(
    tpu_slices: list[dict],
) -> tuple[list[tuple[str, list[str]]], set[str]]:
    slices, claimed = [], set()
    for tpu_slice in tpu_slices:
        unique = dict.fromkeys(tpu_slice["nodes"])
        nodes = [n for n in unique if n not in claimed]
        claimed.update(nodes)
        if nodes:
            slices.append((tpu_slice["topology"], nodes))
    return slices, claimed


def _sorted_nodes(ids: hardware_ids_lib.AcceleratorIds, nodes) -> list[str]:
    return sorted(
        nodes,
        key=lambda n: (hardware_ids_lib.sort_key(ids.host_id(n)), n),
    )


def _add_host_entity(
    target_list,
    node: str,
    ids: hardware_ids_lib.AcceleratorIds,
    host_meta: dict[str, dict[str, str]],
    instance_ids: dict[str, str],
) -> None:
    host = target_list.add()
    host.id = ids.host_id(node)
    host.type = "CHS Host"
    host.metadata.fields["node_name"].string_value = node
    meta = host_meta.get(node, {})
    for key in ("nodepool", "partition"):
        if val := meta.get(key):
            host.metadata.fields[key].string_value = val
    if inst := instance_ids.get(node, ""):
        host.metadata.fields["instance_id"].string_value = inst


def populate_system_entities(
    boh: pb.BillOfHealth,
    hardware_ids: dict[str, list[str]],
    tpu_slices: list[dict],
    host_metadata: dict[str, dict[str, str]] | None = None,
    machine_type: str = "",
) -> None:
    """Populates CHS Host, Subblock and Block IDs into system_info.entities."""
    boh.system_info.ClearField("entities")
    host_meta = host_metadata or {}
    field = boh.system_info.metadata.fields.get("machine_type")
    machine = machine_type or (field.string_value if field else "")
    ids = hardware_ids_lib.AcceleratorIds(hardware_ids)
    instance_ids = _collect_instance_ids(boh, host_meta)
    slices, _ = _claim_slices(tpu_slices)

    grouped: set[str] = set()
    for topology, nodes in slices:
        kind = _group_kind(machine, topology)
        scope = _GROUP_SCOPES.get(kind)
        if not scope:
            continue
        ordered = _sorted_nodes(ids, nodes)
        grouped.update(ordered)
        entity = boh.system_info.entities.add()
        entity.id = ids.group_id(scope, ordered)
        entity.type = f"CHS {kind}"
        entity.metadata.fields["topology"].string_value = topology
        pools = sorted(
            {p for n in ordered if (p := host_meta.get(n, {}).get("nodepool"))}
        )
        if len(pools) == 1:
            entity.metadata.fields["nodepool"].string_value = pools[0]
        for node in ordered:
            _add_host_entity(
                entity.sub_entities, node, ids, host_meta, instance_ids
            )

    tested = {t.node_name for r in boh.test_results for t in r.target_resources}
    ungrouped = (tested | ids.nodes | set(host_meta)) - grouped
    for node in _sorted_nodes(ids, ungrouped):
        _add_host_entity(
            boh.system_info.entities, node, ids, host_meta, instance_ids
        )


def build_report(
    boh: pb.BillOfHealth,
    hardware_ids: dict[str, list[str]],
    tpu_slices: list[dict],
    host_metadata: dict[str, dict[str, str]] | None = None,
) -> model.Report:
    """Builds the report; fails closed (like HTML) without a report_id."""
    return _Builder(boh, hardware_ids, tpu_slices, host_metadata).build()


class _Builder:
    def __init__(self, boh, hardware_ids, tpu_slices, host_metadata=None):
        self._boh = boh
        self._host_meta = host_metadata or {}
        self._pm = PluginManager()
        self._summary = aggregate_boh_summary(boh, plugin_manager=self._pm)
        if self._summary is None:
            raise ValueError("The Bill of Health has no report_id.")
        self._tiering = report_data.load_tiering()
        self._health = {
            h.node: h
            for h in report_data.node_health(boh, self._pm, self._tiering)
        }
        keys = [self._test_key(r) for r in boh.test_results]
        self._tests = list(dict.fromkeys(keys))
        self._multi_host = {
            k
            for k, r in zip(keys, boh.test_results)
            if len(r.target_resources) > 1
        }
        self._pairwise = {
            (s.canonical_name, s.test_level)
            for s in self._summary.test_summaries
            if s.entity_noun == "pairs"
        }
        self._descriptions = report_data.test_descriptions()
        cell_events: dict[tuple[str, str], set[str]] = collections.defaultdict(
            set
        )
        for e in self._summary.errors_and_warnings:
            if e.type in _EVENT_TYPES:
                cell_events[(e.node, e.test)].add(_EVENT_TYPES[e.type])
        self._slices, claimed = _claim_slices(tpu_slices)
        self._metrics = metrics.Metrics(
            boh,
            self._pm,
            self._health,
            self._multi_host,
            cell_events,
            topologies={topo for topo, _ in self._slices},
        )
        field = boh.system_info.metadata.fields.get("machine_type")
        self._machine = field.string_value if field else ""
        self._ids = hardware_ids_lib.AcceleratorIds(hardware_ids)
        self._instance_ids = _collect_instance_ids(boh, self._host_meta)
        (
            self._scheduler,
            self._host_col_label,
            self._pool_col_label,
        ) = self._detect_scheduler(tpu_slices)

        tested = {
            t.node_name for r in boh.test_results for t in r.target_resources
        }
        self._unclaimed = (
            tested | self._ids.nodes | set(self._host_meta)
        ) - claimed

    def _detect_scheduler(self, tpu_slices) -> tuple[str, str, str]:
        sys_type = self._boh.system_info.type.upper()
        if sys_type in _SYSTEM_TYPE_SCHEDULERS:
            return _SYSTEM_TYPE_SCHEDULERS[sys_type]
        if any(m.get("partition") for m in self._host_meta.values()):
            return _SYSTEM_TYPE_SCHEDULERS["SLURM_CLUSTER"]
        if any(m.get("nodepool") for m in self._host_meta.values()) or (
            tpu_slices
        ):
            return _SYSTEM_TYPE_SCHEDULERS["KUBERNETES_CLUSTER"]
        if self._instance_ids:
            return _SYSTEM_TYPE_SCHEDULERS["STANDALONE_CLUSTER"]
        return "", "Node", ""

    def _pool_of(self, node: str) -> str:
        meta = self._host_meta.get(node, {})
        return meta.get("nodepool") or meta.get("partition") or ""

    def _pool_list(self, nodes) -> list[str]:
        return sorted({p for n in nodes if (p := self._pool_of(n))})

    def _pool_summary(self, nodes, limit: int = 3) -> str:
        pools = self._pool_list(nodes)
        if len(pools) <= limit:
            return ", ".join(pools)
        shown = ", ".join(pools[: limit - 1])
        return f"{shown} (+{len(pools) - (limit - 1)} more)"

    def _test_key(self, result: pb.TestResult) -> TestKey:
        return self._pm.resolve_test_name(result.test_name), result.test_level

    def _primary_groups(self) -> list[tuple]:
        """Returns [(title, id_label, topology, nodes, scope), ...]."""
        groups = []
        kinds = [
            _group_kind(self._machine, topology) for topology, _ in self._slices
        ]
        totals, seen = collections.Counter(kinds), collections.Counter()
        for kind, (topology, nodes) in zip(kinds, self._slices, strict=True):
            seen[kind] += 1
            title = f"CHS {kind}"
            if totals[kind] > 1:
                title += f" {seen[kind]} of {totals[kind]}"
            if kind == "Slice":
                title += f" ({topology})"
            scope = _GROUP_SCOPES.get(kind)
            groups.append(
                (
                    title,
                    f"CHS {kind} ID" if scope else None,
                    topology,
                    nodes,
                    scope,
                )
            )

        others = sorted(self._unclaimed)
        if others:
            groups.append(("Hosts", None, None, others, None))
        return groups

    def build(self) -> model.Report:
        ordered_groups = [
            (
                title,
                id_lbl,
                topo,
                _sorted_nodes(self._ids, nodes),
                scope,
            )
            for title, id_lbl, topo, nodes, scope in self._primary_groups()
        ]
        owner: dict[str, int] = {}
        for g_idx, (_, _, _, nodes, _) in enumerate(ordered_groups):
            for node in nodes:
                owner[node] = g_idx

        intra_ids: list[set[int]] = [set() for _ in ordered_groups]
        for r in self._boh.test_results:
            targets = [t.node_name for t in r.target_resources if t.node_name]
            sec_owners = {owner[n] for n in targets if n in owner}
            if len(sec_owners) == 1:
                intra_ids[next(iter(sec_owners))].add(id(r))

        sections = [
            self._section(
                title,
                id_lbl,
                topo,
                nodes,
                scope,
                result_ids=intra_ids[g_idx],
            )
            for g_idx, (title, id_lbl, topo, nodes, scope) in enumerate(
                ordered_groups
            )
        ]

        return model.Report(
            title=f"{self._title()} Bill of Health".strip(),
            run_info=self._run_info(sections),
            tests=[self._test_row(key) for key in self._tests],
            sections=sections,
            id_footer=hardware_ids_lib.ID_FOOTER,
        )

    def _title(self) -> str:
        return self._machine

    def _run_info(self, sections: list[model.Section]) -> list[tuple[str, str]]:
        boh = self._boh
        created = (
            boh.created_at.ToDatetime().strftime("%Y-%m-%d %H:%M:%S UTC")
            if boh.HasField("created_at")
            else hardware_ids_lib.UNAVAILABLE
        )
        total_hosts = len({r.name for s in sections for r in s.hosts})
        info = [
            ("Run ID", self._summary.run_id or boh.report_id),
            ("Machine type", self._machine or hardware_ids_lib.UNAVAILABLE),
            ("CHS version", boh.chs_version or hardware_ids_lib.UNAVAILABLE),
            ("Execution time", created),
            ("Hosts", str(total_hosts)),
        ]
        if self._scheduler:
            info.append(("Scheduler", self._scheduler))
        fields = boh.system_info.metadata.fields
        for label, key in _SOFTWARE_VERSIONS:
            if key in fields and fields[key].string_value:
                info.append((label, fields[key].string_value))
        return info

    def _test_row(self, key: TestKey) -> model.TestRow:
        if key in self._pairwise:
            scope = "Host Pair"
        elif requires_whole_slice(key[0]) or (
            key in self._multi_host and key[0].startswith("tpu")
        ):
            scope = "Slice-Wide"
        elif key not in self._multi_host:
            scope = "Single-Host"
        else:
            scope = "Multi-Host"
        return model.TestRow(
            name=metrics.display(key),
            scope=scope,
            description=self._descriptions.get(key[0], ""),
            thresholds=self._metrics.thresholds(key),
        )

    def _row_status(self, node: str) -> str:
        health = self._health.get(node)
        return _HOST_STATUS[health.status] if health else metrics.INCOMPLETE

    def _section(
        self,
        title,
        id_label,
        topology,
        nodes,
        scope=None,
        *,
        result_ids: set[int] | None = None,
    ):
        labels = labels_lib.Labels(nodes, self._pairwise)
        run_here = {
            metrics.test_key(t)
            for n in nodes
            for t in self._metrics.node_tests(n, result_ids)
        }
        columns = [k for k in self._tests if k in run_here]
        col_specs = [
            (key, header, metric_name)
            for key in columns
            for header, metric_name in self._metrics.column_specs(key)
        ]
        return model.Section(
            title=title,
            group_id_label=id_label,
            group_id=self._ids.group_id(scope, nodes) if scope else None,
            topology=topology,
            columns=[header for _, header, _ in col_specs],
            hosts=[
                self._host_row(n, col_specs, labels, result_ids) for n in nodes
            ],
            charts=self._metrics.charts(
                nodes,
                columns,
                labels,
                result_ids=result_ids,
            ),
            events=self._events(title, labels),
            pool_label=self._pool_col_label,
            host_label=self._host_col_label,
            pool_summary=self._pool_summary(nodes) if scope else "",
        )

    def _host_row(
        self,
        node,
        col_specs,
        labels,
        result_ids,
    ) -> model.Row:
        status = self._row_status(node)
        cells = [
            self._metrics.cell(
                node,
                key,
                labels,
                result_ids,
                metric=metric,
            )
            for key, _, metric in col_specs
        ]
        return model.Row(
            labels[node],
            node,
            self._ids.host_id(node),
            status,
            cells,
            pool=self._pool_of(node),
            instance_id=self._instance_ids.get(node, ""),
        )

    def _events(
        self,
        section,
        labels,
    ) -> list[model.Event]:
        """This section's potential errors and warnings, one per message."""
        merged = collections.defaultdict(list)
        for e in self._summary.errors_and_warnings:
            if e.node in labels.index:
                merged[e.test, e.type, e.category, e.message].append(e.node)
        ordered = sorted(
            merged.items(),
            key=lambda item: (labels.order(item[1]), item[0][0], item[0][2]),
        )
        return [
            model.Event(
                hosts=labels.hosts_text(nodes),
                section=section,
                test=test,
                type=_EVENT_TYPES[kind],
                category=category,
                message=message,
                count=len(nodes),
            )
            for (test, kind, category, message), nodes in ordered
        ]
