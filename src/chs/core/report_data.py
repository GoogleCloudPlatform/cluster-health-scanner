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

"""Report-neutral Bill of Health analysis shared by the HTML and PDF reports.

Both reports judge each node from these functions, so for every host with
results the HTML and PDF reports agree on whether it is healthy.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
import dataclasses
import enum
import functools
import logging
import math
from typing import Any

from chs.boh_generator import NodeContextMatcher
from chs.core.plugin_manager import PluginManager
from chs.protos import bill_of_health_pb2 as pb
from chs.testkit.thresholds import build_thresholds
from chs.utils.config import load_yaml_config
from chs.utils.lifecycle import incomplete_label
from chs.utils.tests import get_all_supported_tests_details

logger = logging.getLogger(__name__)


class TestOutcome(enum.Enum):
    """How one test result judged one of its target nodes."""

    MET = "MET THRESHOLD"
    NO_THRESHOLD = "EXECUTED (No thresholds set)"
    UNMET = "DID NOT MEET THRESHOLD"
    TIMEOUT = "TIMEOUT"
    DEADLINE = "DEADLINE"
    CANCELLED = "CANCELLED"
    FAILED_TO_EXECUTE = "FAILED TO EXECUTE"
    MISSING_METRICS = "MISSING METRICS"


class NodeStatus(enum.Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    POTENTIALLY_DEGRADED = "POTENTIALLY DEGRADED"
    INCOMPLETE = "INCOMPLETE"


_EXECUTION_ERRORS = {
    pb.RESULT_STATUS_TIMEOUT: TestOutcome.TIMEOUT,
    pb.RESULT_STATUS_DEADLINE: TestOutcome.DEADLINE,
    pb.RESULT_STATUS_CANCELLED: TestOutcome.CANCELLED,
}


@dataclasses.dataclass(frozen=True)
class NodeTest:
    """The outcome of one test result for one of its target nodes."""

    result: pb.TestResult
    canonical_name: str
    display_name: str
    outcome: TestOutcome
    status_detail: str = ""


@dataclasses.dataclass(frozen=True)
class NodeHealth:
    node: str
    status: NodeStatus
    has_threshold_breach: bool
    tests: list[NodeTest]


def is_event_metric(metric: pb.Metric) -> bool:
    """Error-detector metrics report events, never measurements."""
    key = metric.key
    return (
        key.endswith("Error Detected")
        or key.endswith("Failure Detected")
        or "Warning" in key
    )


def load_tiering() -> dict[str, list[str]]:
    """Maps each tier-1 test to the tier-2 tests that can clear it."""
    try:
        raw_tiering = load_yaml_config("tiering.yaml").get("tiering", {})
        return {
            k.lower(): [v.lower() for v in vals]
            for k, vals in raw_tiering.items()
        }
    except Exception as e:
        logger.debug(
            "Failed to safely extract tier2_mappings: %s", e, exc_info=True
        )
        return {}


def test_descriptions() -> dict[str, str]:
    """Long description of every supported test, by name and plugin name."""
    descriptions = {}
    for t in get_all_supported_tests_details():
        long_desc = t.get("long_description", "") or t.get("description", "")
        if t.get("name"):
            descriptions[t["name"]] = long_desc
        if t.get("plugin_name"):
            descriptions[t["plugin_name"]] = long_desc
    return descriptions


@dataclasses.dataclass(frozen=True)
class _NodeTestEntry:
    canonical_name: str
    display_name: str
    result: pb.TestResult
    eval_tr: pb.TestResult
    has_unmet_metric: bool
    has_val_criteria: bool
    has_metrics: bool


def _summarize_metrics(
    metrics: Iterable[pb.Metric],
) -> tuple[bool, bool, bool]:
    """Returns (has_unmet_metric, has_val_criteria, has_metrics) in one pass."""
    has_unmet_metric = False
    has_val_criteria = False
    has_metrics = False
    for m in metrics:
        if m.HasField("validation_criteria"):
            if m.validation_criteria.result_status == pb.RESULT_STATUS_FAILED:
                has_unmet_metric = True
            if m.validation_criteria.name:
                has_val_criteria = True
        if not is_event_metric(m):
            has_metrics = True
        if has_unmet_metric and has_val_criteria and has_metrics:
            break
    return has_unmet_metric, has_val_criteria, has_metrics


def node_health(
    boh: pb.BillOfHealth | Sequence[pb.TestResult],
    plugin_manager: PluginManager,
    tiering: dict[str, list[str]],
) -> list[NodeHealth]:
    """Judges every target node of the BoH or test results, sorted by node name.

    A node failing only tier-1 tests is POTENTIALLY_DEGRADED, and HEALTHY
    when a tier-2 test for each of them passed on it.
    """
    results = boh.test_results if isinstance(boh, pb.BillOfHealth) else boh
    resolved_name_cache: dict[str, str] = {}
    plugin_cache: dict[str, Any] = {}
    by_node: dict[str, list[_NodeTestEntry]] = defaultdict(list)

    for tr in results:
        target_nodes = {t.node_name for t in tr.target_resources}
        if not target_nodes:
            continue

        if tr.test_name not in resolved_name_cache:
            resolved_name_cache[tr.test_name] = (
                plugin_manager.resolve_test_name(tr.test_name) or tr.test_name
            )
        canonical_name = resolved_name_cache[tr.test_name]

        if canonical_name not in plugin_cache:
            plugin_cache[canonical_name] = plugin_manager.get_plugin(
                canonical_name
            )
        plugin = plugin_cache[canonical_name]
        has_custom_filter = bool(
            plugin and hasattr(plugin, "_filter_validation_results")
        )

        eval_tr = tr
        if has_custom_filter:
            sorted_nodes = sorted(target_nodes)
            filtered = plugin._filter_validation_results([tr], sorted_nodes)
            if filtered:
                eval_tr = filtered[0]

        display_name = (
            f"{canonical_name} ({tr.test_level})"
            if tr.test_level
            else canonical_name
        )

        if has_custom_filter or len(target_nodes) == 1:
            has_unmet, has_val, has_m = _summarize_metrics(eval_tr.metrics)
            entry = _NodeTestEntry(
                canonical_name=canonical_name,
                display_name=display_name,
                result=tr,
                eval_tr=eval_tr,
                has_unmet_metric=has_unmet,
                has_val_criteria=has_val,
                has_metrics=has_m,
            )
            for node_name in target_nodes:
                by_node[node_name].append(entry)
        else:
            matcher = NodeContextMatcher(target_nodes)
            per_node_metrics: dict[str, list[pb.Metric]] = {
                n: [] for n in target_nodes
            }
            shared_metrics: list[pb.Metric] = []
            for m in eval_tr.metrics:
                ctx = m.metadata.fields.get("context")
                ctx_str = ctx.string_value if ctx and ctx.string_value else ""
                if ctx_str:
                    matched_nodes = matcher.match(ctx_str)
                    if matched_nodes:
                        for t_node in matched_nodes:
                            per_node_metrics[t_node].append(m)
                        continue
                shared_metrics.append(m)

            shared_unmet, shared_val, shared_has_m = _summarize_metrics(
                shared_metrics
            )
            for node_name in target_nodes:
                n_unmet, n_val, n_has_m = _summarize_metrics(
                    per_node_metrics[node_name]
                )
                by_node[node_name].append(
                    _NodeTestEntry(
                        canonical_name=canonical_name,
                        display_name=display_name,
                        result=tr,
                        eval_tr=eval_tr,
                        has_unmet_metric=(shared_unmet or n_unmet),
                        has_val_criteria=(shared_val or n_val),
                        has_metrics=(shared_has_m or n_has_m),
                    )
                )

    return [
        _judge_node(node, by_node[node], tiering) for node in sorted(by_node)
    ]


def _judge_node(
    node: str,
    entries: list[_NodeTestEntry],
    tiering: dict[str, list[str]],
) -> NodeHealth:
    tests = []
    failed_tests = set()
    passed_tests = set()
    for entry in entries:
        canonical_name = entry.canonical_name
        eval_tr = entry.eval_tr
        if entry.has_unmet_metric:
            failed_tests.add(canonical_name.lower())
        outcome = _outcome(eval_tr, entry)
        if outcome == TestOutcome.MET:
            passed_tests.add(canonical_name.lower())
        status_detail = (
            incomplete_label(eval_tr)
            if eval_tr.result_status == pb.RESULT_STATUS_DEADLINE
            else ""
        )
        tests.append(
            NodeTest(
                entry.result,
                canonical_name,
                entry.display_name,
                outcome,
                status_detail,
            )
        )

    cleared_tier1 = {
        tier1
        for tier1, tier2_tests in tiering.items()
        if any(t in passed_tests for t in tier2_tests)
    }
    has_breach = bool(failed_tests) and not failed_tests <= cleared_tier1
    incomplete = any(
        t.outcome
        not in (TestOutcome.MET, TestOutcome.NO_THRESHOLD, TestOutcome.UNMET)
        for t in tests
    )
    if has_breach and failed_tests <= set(tiering):
        status = NodeStatus.POTENTIALLY_DEGRADED
    elif has_breach:
        status = NodeStatus.DEGRADED
    elif incomplete:
        status = NodeStatus.INCOMPLETE
    else:
        status = NodeStatus.HEALTHY
    return NodeHealth(node, status, has_breach, tests)


def _outcome(judged: pb.TestResult, entry: _NodeTestEntry) -> TestOutcome:
    if judged.result_status in _EXECUTION_ERRORS:
        return _EXECUTION_ERRORS[judged.result_status]
    if entry.has_unmet_metric:
        return TestOutcome.UNMET
    if judged.result_status == pb.RESULT_STATUS_FAILED:
        return TestOutcome.FAILED_TO_EXECUTE
    if not entry.has_metrics:
        return TestOutcome.MISSING_METRICS
    return (
        TestOutcome.MET if entry.has_val_criteria else TestOutcome.NO_THRESHOLD
    )


@dataclasses.dataclass(frozen=True)
class ThresholdSpec:
    """One resolved threshold rule with its metric label and canonical unit."""

    name: str
    op: str
    value: str
    unit: str
    is_number: bool

    def format(self) -> str:
        return f"{self.name} {self.op} {self.value} {self.unit}".strip()


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


_TOPOLOGY_KEYS = ("topology", "tpu_topology")
# Threshold conditions the reports can check against the Bill of Health.
_RESOLVED_CONDITIONS = frozenset({"workload_name", "tpu_topology"})
# Metric-level selectors evaluated within a single run's own output.
_METRIC_FILTERS = frozenset({"context", "elements"})


def collect_host_topologies(
    boh: pb.BillOfHealth,
    tpu_slices: list[dict] | None = None,
    host_metadata: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    """Node name -> TPU topology the run tested it as.

    Reads the host entities on boh.system_info (a host inherits its
    group's topology), plus the raw inventory when the caller has it, so
    the HTML and PDF reports resolve the same topology for each host.
    """
    topologies: dict[str, str] = {}

    def walk(entities: Iterable[pb.ResourceEntity], inherited: str) -> None:
        for entity in entities:
            fields = entity.metadata.fields
            topo = inherited
            for key in _TOPOLOGY_KEYS:
                if key in fields and fields[key].string_value.strip():
                    topo = fields[key].string_value.strip()
            if "node_name" in fields and topo:
                topologies.setdefault(fields["node_name"].string_value, topo)
            walk(entity.sub_entities, topo)

    walk(boh.system_info.entities, "")
    for s in tpu_slices or []:
        if topo := s.get("topology"):
            for node in s.get("nodes") or []:
                topologies.setdefault(node, topo)
    for node, meta in (host_metadata or {}).items():
        if topo := meta.get("tpu_topology"):
            topologies.setdefault(node, topo)
    return topologies


def topologies_of(
    host_topologies: dict[str, str], nodes: Iterable[str]
) -> set[str]:
    """TPU topologies of the given hosts."""
    return {host_topologies[n] for n in nodes if n in host_topologies}


@functools.lru_cache(maxsize=1)
def _raw_thresholds() -> dict[str, Any]:
    try:
        return build_thresholds()
    except OSError:
        return {}


def _unit_lookup(plugin: Any) -> dict[str, str]:
    """Lower-cased metric name -> canonical unit declared by the plugin."""
    if not plugin:
        return {}
    return {k.lower(): v for k, v in plugin.get_metric_units().items()}


def configured_threshold_specs(
    test_name: str,
    test_level: str,
    *,
    machine_type: str = "",
    topologies: set[str] | None = None,
    plugin_manager: PluginManager | None = None,
) -> list[ThresholdSpec]:
    """Configured thresholds from thresholds.yaml with canonical FOM units."""
    raw = _raw_thresholds()
    default_cfg = raw.get("default", {}).get(test_name, {})
    machine_cfg = (
        raw.get(machine_type, {}).get(test_name, {}) if machine_type else {}
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
    pm = plugin_manager or PluginManager()
    plugin = pm.get_plugin(test_name)
    aliases = plugin.get_fom_aliases() if plugin else {}
    unit_map = _unit_lookup(plugin)
    active_topos = topologies or set()
    seen: dict[ThresholdSpec, None] = {}
    # Single-run metric filters (e.g. elements, context) may select one rule
    # per metric name; any other unresolvable condition (e.g. domain_type)
    # is skipped because the report cannot verify whether the run matched it.
    filtered: dict[str, dict[ThresholdSpec, None]] = defaultdict(dict)
    for metric_key, cfg in merged.items():
        entries = cfg if isinstance(cfg, list) else [cfg]
        for entry in entries:
            if isinstance(entry, dict):
                val = entry.get("value")
                conds = entry.get("conditions") or {}
            else:
                val, conds = entry, {}
            extra_conds = set(conds) - _RESOLVED_CONDITIONS
            if extra_conds - _METRIC_FILTERS:
                continue
            if (
                "workload_name" in conds
                and str(conds["workload_name"]) != test_level
            ):
                continue
            if (
                "tpu_topology" in conds
                and str(conds["tpu_topology"]) not in active_topos
            ):
                continue
            if val is None:
                continue
            is_num = is_finite_number(val)
            if is_num and comp_type == "min":
                val = round(float(val) * 0.95)
            elif is_num and isinstance(val, float) and val.is_integer():
                val = int(val)
            op = (
                "==" if isinstance(val, bool) or comp_type == "scalar" else ">="
            )
            is_relative = comp_type == "relative"
            name = (
                metric_key.title()
                if is_relative
                else aliases.get(metric_key, metric_key)
            )
            unit = (
                ""
                if is_relative
                else (
                    unit_map.get(name.lower())
                    or unit_map.get(metric_key.lower())
                    or ""
                )
            )
            spec = ThresholdSpec(name, op, str(val), unit, is_num)
            if extra_conds:
                filtered[name][spec] = None
            else:
                seen[spec] = None
    resolved_names = {s.name for s in seen}
    for name, specs in filtered.items():
        if len(specs) == 1 and name not in resolved_names:
            seen.update(specs)
    return list(seen)


def configured_thresholds(
    test_name: str,
    test_level: str,
    *,
    machine_type: str = "",
    topologies: set[str] | None = None,
    plugin_manager: PluginManager | None = None,
) -> list[str]:
    """Formatted threshold strings for fallback when no host emits metrics."""
    specs = configured_threshold_specs(
        test_name,
        test_level,
        machine_type=machine_type,
        topologies=topologies,
        plugin_manager=plugin_manager,
    )
    return list(dict.fromkeys(s.format() for s in specs))
