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

"""Per-host cells and charts: validation rows and node outcomes as Points.

Values, units and thresholds are the plugin-filtered validation rows of
extract_validation_data; outcomes are report_data.node_health's.
"""

import collections
import functools
import math

from chs.core import report_data
from chs.core.plugin_manager import PluginManager
from chs.core.utils.validation_analyzer import (
    ValidationRowData,
    extract_validation_data,
)
from chs.pdf_generator import model
from chs.pdf_generator.labels import TestKey
from chs.protos import bill_of_health_pb2 as pb
from chs.testkit.thresholds import build_thresholds

HEALTHY = report_data.NodeStatus.HEALTHY.value
DEGRADED = report_data.NodeStatus.DEGRADED.value
POTENTIALLY_DEGRADED = report_data.NodeStatus.POTENTIALLY_DEGRADED.value
INCOMPLETE = report_data.NodeStatus.INCOMPLETE.value
_Outcome = report_data.TestOutcome
HEALTHY_OUTCOMES = {_Outcome.MET, _Outcome.NO_THRESHOLD}
REASONS = {
    o: o.value
    for o in (
        _Outcome.TIMEOUT,
        _Outcome.DEADLINE,
        _Outcome.CANCELLED,
        _Outcome.FAILED_TO_EXECUTE,
        _Outcome.MISSING_METRICS,
    )
}

ChartSpec = tuple[str, str, float]  # (metric, unit, threshold)


def display(key: TestKey) -> str:
    return f"{key[0]} ({key[1]})" if key[1] else key[0]


def test_key(test: report_data.NodeTest) -> TestKey:
    return test.canonical_name, test.result.test_level


def hosts_of(result: pb.TestResult) -> tuple[str, ...]:
    return tuple(t.node_name for t in result.target_resources)


def _is_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _spec(row: ValidationRowData) -> ChartSpec | None:
    """The chart a row is drawn in: numeric >= criteria only."""
    if row.comparator != pb.COMPARATOR_GTE or row.expected is None:
        return None
    return (
        (row.metric, row.unit, row.expected) if _is_number(row.value) else None
    )


@functools.lru_cache(maxsize=1)
def _raw_thresholds() -> dict:
    try:
        return build_thresholds()
    except OSError:
        return {}


def _fmt(value) -> str:
    if not _is_number(value):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def _worst(rows: list[ValidationRowData], label: str) -> model.Point:
    row = min(rows, key=lambda r: r.value)
    return model.Point(
        label, row.value, _fmt(row.value), row.status == "NOT MET"
    )


def _other_failure(row: ValidationRowData) -> str:
    if _is_number(row.value):
        return f"{row.metric} {_fmt(row.value)}"
    if row.value is not None and not isinstance(row.value, float):
        return f"{row.metric}: {row.value}"
    return f"{row.metric}: {DEGRADED}"


def incomplete(label: str) -> model.Point:
    return model.Point(label, None, INCOMPLETE, False)


_MAX_BARS_PER_CHART = 32


def _chart_chunks(points: list[model.Point], base_title: str):
    if len(points) <= _MAX_BARS_PER_CHART:
        return [(base_title, points)]
    total = len(points)
    chunks = []
    for start in range(0, total, _MAX_BARS_PER_CHART):
        end = min(total, start + _MAX_BARS_PER_CHART)
        chunks.append(
            (f"{base_title} ({start + 1}–{end} of {total})", points[start:end])
        )
    return chunks


class Metrics:
    """Validation rows and node outcomes, indexed by test and target hosts."""

    def __init__(
        self,
        boh: pb.BillOfHealth,
        pm: PluginManager,
        health: dict[str, report_data.NodeHealth],
        multi_host: set[TestKey],
        cell_events: dict[tuple[str, str], set[str]] | None = None,
        topologies: set[str] | None = None,
    ):
        self._pm = pm
        self._health = health
        self._multi_host = multi_host
        self._cell_events = cell_events or {}
        self._topologies = topologies or set()
        machine_field = boh.system_info.metadata.fields.get("machine_type")
        self._machine = machine_field.string_value if machine_field else ""
        skip_field = boh.metadata.fields.get("skip_validation")
        self._skip_validation = bool(skip_field and skip_field.bool_value)
        self._rows = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        self._specs = collections.defaultdict(dict)
        self._thresholds = collections.defaultdict(dict)
        for group in extract_validation_data(
            list(boh.test_results), plugin_manager=pm
        ):
            key = (
                pm.resolve_test_name(group.test_name),
                group.test_level,
            )
            for row in group.rows:
                self._rows[key][frozenset(row.target_nodes)].append(row)
                threshold = f"{row.metric} {row.threshold} {row.unit}"
                self._thresholds[key][threshold.strip()] = None
                if spec := _spec(row):
                    self._specs[key][spec] = None

    def _result_rows(self, key, test) -> list[ValidationRowData]:
        return self._rows[key].get(frozenset(hosts_of(test.result)), [])

    def _single_column_label(self, key: TestKey) -> str:
        """Unit or threshold name to show in a single-column test header."""
        rows = [r for node_rows in self._rows[key].values() for r in node_rows]
        units = list(dict.fromkeys(r.unit for r in rows if r.unit))
        if len(units) == 1:
            return units[0]
        numeric_metrics = list(
            dict.fromkeys(r.metric for r in rows if _is_number(r.value))
        )
        if len(numeric_metrics) == 1 and all(
            r.metric == numeric_metrics[0] for r in rows
        ):
            return numeric_metrics[0]
        return ""

    def _metric_columns(self, key: TestKey) -> list[tuple[str, str]]:
        """Distinct (metric, unit) pairs for chart-backed host table columns."""
        return list(dict.fromkeys((m, u) for m, u, _ in self._specs[key]))

    def column_specs(self, key: TestKey) -> list[tuple[str, str | None]]:
        """Returns [(column header, metric filter or None), ...] for a test."""
        cols = self._metric_columns(key)
        if len(cols) > 1:
            return [
                (
                    f"{display(key)} · {m} ({u})"
                    if u
                    else f"{display(key)} · {m}",
                    m,
                )
                for m, u in cols
            ]
        suffix = (
            (cols[0][1] or cols[0][0])
            if cols
            else self._single_column_label(key)
        )
        header = f"{display(key)} ({suffix})" if suffix else display(key)
        return [(header, None)]

    def thresholds(self, key: TestKey) -> list[str]:
        """Distinct evaluated thresholds of a test, as 'metric >= X unit'."""
        if self._thresholds[key]:
            return list(self._thresholds[key])
        if self._skip_validation:
            return []
        return self._configured_thresholds(key)

    def _configured_thresholds(self, key: TestKey) -> list[str]:
        """Falls back to thresholds.yaml when no host emits metrics."""
        raw = _raw_thresholds()
        default_cfg = raw.get("default", {}).get(key[0], {})
        machine_cfg = (
            raw.get(self._machine, {}).get(key[0], {}) if self._machine else {}
        )
        if not default_cfg and not machine_cfg:
            return []
        comp_type = machine_cfg.get("type") or default_cfg.get("type")
        if not comp_type:
            return []
        merged = {
            **default_cfg.get("thresholds", {}),
            **machine_cfg.get("thresholds", {}),
        }
        plugin = self._pm.get_plugin(key[0])
        aliases = plugin.get_fom_aliases() if plugin else {}
        seen: dict[str, None] = {}
        for metric_key, cfg in merged.items():
            entries = cfg if isinstance(cfg, list) else [cfg]
            for entry in entries:
                if isinstance(entry, dict):
                    val = entry.get("value")
                    conds = entry.get("conditions") or {}
                else:
                    val, conds = entry, {}
                if (
                    "workload_name" in conds
                    and str(conds["workload_name"]) != key[1]
                ):
                    continue
                if (
                    "tpu_topology" in conds
                    and str(conds["tpu_topology"]) not in self._topologies
                ):
                    continue
                if val is None:
                    continue
                if _is_number(val) and comp_type == "min":
                    val = round(float(val) * 0.95)
                op = (
                    "=="
                    if isinstance(val, bool) or comp_type == "scalar"
                    else ">="
                )
                name = (
                    metric_key.title()
                    if comp_type == "relative"
                    else aliases.get(metric_key, metric_key)
                )
                seen[f"{name} {op} {val}"] = None
        return list(seen)

    def node_tests(
        self, node: str, result_ids: set[int] | None = None
    ) -> list[report_data.NodeTest]:
        health = self._health.get(node)
        tests = health.tests if health else []
        if result_ids is None:
            return tests
        return [t for t in tests if id(t.result) in result_ids]

    def cell(
        self,
        node,
        key,
        labels,
        result_ids: set[int] | None = None,
        metric: str | None = None,
    ) -> list[model.Point]:
        """One line per result of the test/metric the host took part in."""
        tests = [
            t for t in self.node_tests(node, result_ids) if test_key(t) == key
        ]
        tests.sort(key=lambda t: labels.order(hosts_of(t.result)))
        points = []
        for t in tests:
            targets = hosts_of(t.result)
            wrap = functools.partial(
                labels.cell_text, key=key, targets=targets, viewer=node
            )
            points += self.points(
                key, t, labels.entity(key, targets), wrap, metric=metric
            )
        events = self._cell_events.get((node, display(key)), set())
        if "Error" in events:
            points.append(model.Point(labels[node], None, "Error logged", True))
        if "Warning" in events:
            points.append(
                model.Point(labels[node], None, "Warning logged", False)
            )
        return points or [incomplete(labels[node])]

    def points(
        self, key, test, label, wrap=str, *, metric: str | None = None
    ) -> list[model.Point]:
        """A result's metric values, or outcome/status when none."""
        if test.outcome in REASONS:
            return [model.Point(label, None, wrap(REASONS[test.outcome]), True)]
        rows = self._result_rows(key, test)
        if metric is not None:
            rows = [r for r in rows if r.metric == metric]
            named = False
        else:
            named = len(self._metric_columns(key)) > 1
        charted = [r for r in rows if _spec(r)]
        others = [r for r in rows if not _spec(r)]
        points = [
            model.Point(
                label,
                r.value,
                wrap(f"{r.metric} {_fmt(r.value)}" if named else _fmt(r.value)),
                r.status == "NOT MET",
            )
            for r in charted
        ]
        failed_others = [r for r in others if r.status == "NOT MET"]
        if charted:
            for r in failed_others:
                points.append(
                    model.Point(label, None, wrap(_other_failure(r)), True)
                )
        else:
            if failed_others and len(others) > 1:
                for r in failed_others:
                    points.append(
                        model.Point(label, None, wrap(_other_failure(r)), True)
                    )
            else:
                degraded = (
                    bool(failed_others)
                    if others
                    else test.outcome == _Outcome.UNMET
                )
                status = DEGRADED if degraded else HEALTHY
                value = others[0].value if len(others) == 1 else None
                if _is_number(value):
                    points.append(
                        model.Point(
                            label,
                            value,
                            wrap(_fmt(value)),
                            degraded,
                        )
                    )
                elif value is not None and not isinstance(value, float):
                    points.append(
                        model.Point(
                            label,
                            None,
                            wrap(str(value)),
                            degraded,
                        )
                    )
                else:
                    points.append(
                        model.Point(label, None, wrap(status), degraded)
                    )
        return points

    def charts(
        self,
        nodes,
        tests,
        labels,
        result_ids: set[int] | None = None,
    ) -> list[model.Chart]:
        charts = []
        for key in tests:
            for spec in self._specs[key]:
                points = self._bars(key, spec, nodes, labels, result_ids)
                title = display(key)
                if len(self._metric_columns(key)) > 1:
                    title += f" · {spec[0]}"
                if any(p.value is not None or p.degraded for p in points):
                    for chart_title, chunk in _chart_chunks(points, title):
                        charts.append(
                            model.Chart(
                                test=display(key),
                                title=chart_title,
                                unit=spec[1],
                                threshold=spec[2],
                                points=chunk,
                            )
                        )
        return charts

    def _bars(
        self,
        key,
        spec,
        nodes,
        labels,
        result_ids: set[int] | None = None,
    ) -> list[model.Point]:
        """One bar per result inside the section; every host if per-host."""
        bars = {}
        allowed_targets = (
            {
                frozenset(hosts_of(t.result))
                for n in nodes
                for t in self.node_tests(n, result_ids)
                if test_key(t) == key
            }
            if result_ids is not None
            else None
        )
        label_set = set(labels.index)
        candidates = (
            (
                (t, self._rows[key][t])
                for t in allowed_targets
                if t in self._rows[key]
            )
            if allowed_targets is not None
            else self._rows[key].items()
        )
        for targets, rows in candidates:
            rows = [r for r in rows if _spec(r) == spec]
            if rows and targets <= label_set:
                bars[targets] = _worst(rows, labels.entity(key, targets))
        if key not in self._multi_host:
            for node in nodes:
                bars.setdefault(
                    frozenset([node]), self._unmeasured(key, node, labels[node])
                )
        return [bars[t] for t in sorted(bars, key=labels.order)]

    def _unmeasured(self, key, node, label) -> model.Point:
        outcomes = [
            t.outcome for t in self.node_tests(node) if test_key(t) == key
        ]
        bad = [o for o in outcomes if o not in HEALTHY_OUTCOMES]
        if bad:
            return model.Point(label, None, REASONS.get(bad[0], DEGRADED), True)
        return model.Point(
            label, None, HEALTHY if outcomes else INCOMPLETE, False
        )
