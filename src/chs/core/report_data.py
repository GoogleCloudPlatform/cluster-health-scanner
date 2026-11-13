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
import logging
from typing import Any

from chs.boh_generator import NodeContextMatcher
from chs.core.plugin_manager import PluginManager
from chs.protos import bill_of_health_pb2 as pb
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
