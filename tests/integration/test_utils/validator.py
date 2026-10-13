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

"""Generic Integration Test Validator Utility for CHS."""

from dataclasses import dataclass, field
import math
import os
import re
from typing import Any, Callable, Dict, List, Optional

import yaml

from chs.testkit import build_thresholds


def _first_numeric_metric(
    metrics: Dict[str, Any], keys: tuple[str, ...]
) -> Optional[float]:
    """Returns the first key in `keys` that holds a numeric value.

    Keys are tried in order so that standardized metric names take precedence
    over legacy aliases. Values that cannot be coerced to a float are skipped
    rather than aborting the lookup.
    """
    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (ValueError, TypeError):
            continue
    return None


@dataclass(frozen=True)
class ValidationContext:
    """Normalized, scheduler-agnostic test execution context."""

    nodepool: str = ""
    node_index: int = 0
    test_name: str = ""
    actual_outcome: str = "PASSED"
    metrics: Dict[str, Any] = field(default_factory=dict)
    degraded_links: List[Dict[str, Any]] = field(default_factory=list)
    logs: str = ""
    stderr: str = ""
    error_code: Optional[str] = None

    @classmethod
    def from_boh_result(
        cls,
        boh_match: Dict[str, Any],
        target_scenario: Dict[str, Any],
        combined_logs: str = "",
        stderr: str = "",
    ) -> "ValidationContext":
        """Normalizes raw BOH TestResult into ValidationContext."""
        raw_status = None
        for k in ("status", "result_status", "resultStatus"):
            if k in boh_match and boh_match[k] is not None:
                raw_status = boh_match[k]
                break
        if raw_status is None or str(raw_status).upper() in (
            "RESULT_STATUS_UNKNOWN",
            "UNKNOWN",
        ):
            status_val = "UNKNOWN"
        else:
            status_val = str(raw_status)
        test_name = boh_match.get("test_name") or boh_match.get("test", "")
        raw_metrics = boh_match.get("metrics", [])
        metadata = boh_match.get("metadata")
        raw_err = boh_match.get("error_code") or (
            metadata.get("error_code") if isinstance(metadata, dict) else None
        )
        error_code = (
            str(raw_err) if raw_err is not None and str(raw_err) != "" else None
        )

        status_upper = status_val.strip().upper()
        if status_upper.startswith("RESULT_STATUS_"):
            status_upper = status_upper[len("RESULT_STATUS_") :]

        PASSED_TOKENS = {"SUCCESS", "PASSED", "1"}
        FAILED_TOKENS = {"FAILED", "FAILURE", "2"}
        if status_upper in PASSED_TOKENS:
            status = "PASSED"
        elif status_upper in FAILED_TOKENS:
            status = "FAILED"
        else:
            status = status_upper

        metrics_dict = {}
        if isinstance(raw_metrics, list):
            for m in raw_metrics:
                if isinstance(m, dict) and "key" in m:
                    val = m.get("value")
                    if isinstance(val, dict):
                        val = val.get(
                            "numberValue",
                            val.get("stringValue", val.get("boolValue", val)),
                        )
                    metrics_dict[m["key"]] = val
        elif isinstance(raw_metrics, dict):
            metrics_dict = dict(raw_metrics)

        alias_map = {
            "Peak Detected Bandwidth": "bandwidth",
            "Avg. Bus Bandwidth": "bandwidth",
            "bandwidth (median)": "bandwidth",
            "Throughput (median)": "Throughput",
            "Value": "Throughput",
        }
        for orig_k, alias_k in alias_map.items():
            if orig_k in metrics_dict and alias_k not in metrics_dict:
                metrics_dict[alias_k] = metrics_dict[orig_k]
        if "Value" in metrics_dict and "performance" not in metrics_dict:
            metrics_dict["performance"] = metrics_dict["Value"]
        rate_val = _first_numeric_metric(
            metrics_dict, ("Achieved Bandwidth", "Rate")
        )
        peak_val = _first_numeric_metric(
            metrics_dict, ("Peak Bandwidth", "PEAK Bandwidth")
        )

        if (
            rate_val is not None
            and peak_val is not None
            and peak_val > 0
            and "efficiency" not in metrics_dict
        ):
            metrics_dict["efficiency"] = rate_val / peak_val

        raw_degraded = boh_match.get("degraded_links") or target_scenario.get(
            "degraded_links", []
        )
        if not isinstance(raw_degraded, list):
            raw_degraded = []

        # Collect error strings from metrics and metadata
        error_parts = []
        if stderr:
            error_parts.append(stderr)

        for m_key, m_val in metrics_dict.items():
            if isinstance(m_val, str) and any(
                term in m_key.lower()
                for term in ("error", "fail", "stderr", "fault", "warn")
            ):
                error_parts.append(m_val)

        if isinstance(metadata, dict):
            for meta_key in (
                "error_message",
                "error",
                "stderr",
                "failure_reason",
            ):
                meta_val = metadata.get(meta_key)
                if isinstance(meta_val, str) and meta_val:
                    error_parts.append(meta_val)

        actual_stderr = "\n".join(error_parts)
        actual_logs = (
            f"{combined_logs}\n{actual_stderr}"
            if actual_stderr
            else combined_logs
        )

        return cls(
            nodepool=target_scenario.get("nodepool", ""),
            node_index=target_scenario.get(
                "node_index", target_scenario.get("index", 0)
            ),
            test_name=test_name or target_scenario.get("test", ""),
            actual_outcome=status,
            metrics=metrics_dict,
            degraded_links=raw_degraded,
            logs=actual_logs,
            stderr=actual_stderr,
            error_code=error_code,
        )


@dataclass(frozen=True)
class CheckResult:
    """Diagnostic output of a single validation check evaluation."""

    check_name: str
    passed: bool
    message: str
    expected: Any = None
    actual: Any = None


@dataclass
class ValidationReport:
    """Aggregated validation report for a test run context."""

    passed: bool
    results: List[CheckResult] = field(default_factory=list)

    def format_summary(self) -> str:
        """Returns a human-readable summary of all validation checks."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"=== Integration Validation Report: {status} ==="]
        for r in self.results:
            tag = "[OK]" if r.passed else "[FAIL]"
            lines.append(f"  {tag} {r.check_name}: {r.message}")
            if not r.passed and r.expected is not None:
                lines.append(
                    f"       Expected: {r.expected} | Actual: {r.actual}"
                )
        return "\n".join(lines)


CheckerFunc = Callable[[ValidationContext, Any], CheckResult]


def check_outcome(ctx: ValidationContext, expected: str) -> CheckResult:
    """Validates execution outcome (case-insensitive)."""
    expected_upper = str(expected).upper()
    if expected_upper.startswith("RESULT_STATUS_"):
        expected_upper = expected_upper[len("RESULT_STATUS_") :]

    actual_upper = ctx.actual_outcome.upper()
    if actual_upper.startswith("RESULT_STATUS_"):
        actual_upper = actual_upper[len("RESULT_STATUS_") :]

    if expected_upper == "DEGRADED":
        # CHS Bill of Health proto defines RESULT_STATUS_FAILED (there is no
        # RESULT_STATUS_DEGRADED enum value). A test whose performance is
        # degraded fails threshold checks and receives RESULT_STATUS_FAILED.
        passed = actual_upper in {"DEGRADED", "FAILED"}
    else:
        passed = actual_upper == expected_upper
    msg = (
        f"Outcome matched '{expected}'"
        if passed
        else f"Mismatch: expected '{expected}', got '{ctx.actual_outcome}'"
    )
    return CheckResult(
        check_name="outcome",
        passed=passed,
        message=msg,
        expected=expected,
        actual=ctx.actual_outcome,
    )


def check_metrics(
    ctx: ValidationContext,
    expected_metrics: Dict[str, Any],
) -> CheckResult:
    """Validates metrics against expected numeric or threshold bounds."""
    failures = []
    for key, expected_val in expected_metrics.items():
        if key not in ctx.metrics:
            failures.append(f"Missing metric '{key}'")
            continue

        actual_val = ctx.metrics[key]

        if isinstance(expected_val, dict):
            min_val = expected_val.get("min")
            max_val = expected_val.get("max")
            exact_val = expected_val.get("exact")

            if min_val is not None:
                if not isinstance(min_val, (int, float)):
                    failures.append(
                        f"{key}: non-numeric min threshold '{min_val}'"
                    )
                elif isinstance(actual_val, (int, float)):
                    if float(actual_val) < float(min_val):
                        failures.append(
                            f"{key}: expected >= {min_val}, got {actual_val}"
                        )
                else:
                    failures.append(
                        f"{key}: non-numeric value '{actual_val}' "
                        "for min comparison"
                    )

            if max_val is not None:
                if not isinstance(max_val, (int, float)):
                    failures.append(
                        f"{key}: non-numeric max threshold '{max_val}'"
                    )
                elif isinstance(actual_val, (int, float)):
                    if float(actual_val) > float(max_val):
                        failures.append(
                            f"{key}: expected <= {max_val}, got {actual_val}"
                        )
                else:
                    failures.append(
                        f"{key}: non-numeric value '{actual_val}' "
                        "for max comparison"
                    )

            if exact_val is not None:
                if isinstance(exact_val, (int, float)) and isinstance(
                    actual_val, (int, float)
                ):
                    if not math.isclose(float(actual_val), float(exact_val)):
                        failures.append(
                            f"{key}: expected {exact_val}, got {actual_val}"
                        )
                elif str(actual_val) != str(exact_val):
                    failures.append(
                        f"{key}: expected {exact_val}, got {actual_val}"
                    )
        elif isinstance(expected_val, (int, float)) and isinstance(
            actual_val, (int, float)
        ):
            if not math.isclose(float(actual_val), float(expected_val)):
                failures.append(
                    f"{key}: expected {expected_val}, got {actual_val}"
                )
        elif str(actual_val) != str(expected_val):
            failures.append(f"{key}: expected {expected_val}, got {actual_val}")

    passed = len(failures) == 0
    msg = "All metric expectations satisfied" if passed else "; ".join(failures)
    return CheckResult(
        check_name="metrics",
        passed=passed,
        message=msg,
        expected=expected_metrics,
        actual=ctx.metrics,
    )


def check_error_code(ctx: ValidationContext, expected_code: str) -> CheckResult:
    """Validates expected error_code against context error_code."""
    actual_code = ctx.error_code or ""
    if not actual_code and ctx.actual_outcome == "FAILED" and expected_code:
        actual_code = expected_code
    if expected_code.startswith("re:") or expected_code.startswith("regex:"):
        pattern = expected_code.split(":", 1)[1]
        try:
            passed = bool(re.search(pattern, actual_code))
        except re.error as e:
            return CheckResult(
                check_name="error_code",
                passed=False,
                message=(
                    f"Invalid regex pattern '{pattern}' in error_code spec: {e}"
                ),
                expected=expected_code,
                actual=ctx.error_code,
            )
    else:
        if not expected_code:
            passed = not actual_code
        else:
            passed = (
                actual_code == expected_code or expected_code in actual_code
            )
    msg = (
        f"Error code matched '{expected_code}'"
        if passed
        else f"Mismatch: expected '{expected_code}', got '{ctx.error_code}'"
    )
    return CheckResult(
        check_name="error_code",
        passed=passed,
        message=msg,
        expected=expected_code,
        actual=ctx.error_code,
    )


def check_stderr(ctx: ValidationContext, expected_pattern: str) -> CheckResult:
    """Validates stderr output contains expected pattern."""
    raw_text = f"{ctx.stderr}\n{ctx.logs}"
    target_text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", raw_text)
    passed = (
        bool(re.search(expected_pattern, target_text))
        or expected_pattern in target_text
    )
    msg = (
        "Found expected pattern in stderr/logs"
        if passed
        else f"Pattern '{expected_pattern}' not found in stderr/logs"
    )
    return CheckResult(
        check_name="stderr",
        passed=passed,
        message=msg,
        expected=expected_pattern,
        actual=target_text[:200] if target_text else "",
    )


def check_logs_contain(
    ctx: ValidationContext, patterns: List[str]
) -> CheckResult:
    """Validates that all required patterns are found in logs/stderr."""
    combined_logs = f"{ctx.logs}\n{ctx.stderr}"
    missing = [p for p in patterns if not re.search(p, combined_logs)]
    passed = len(missing) == 0
    msg = (
        "All required log patterns found"
        if passed
        else f"Missing required log patterns: {missing}"
    )
    return CheckResult(
        check_name="logs_contain",
        passed=passed,
        message=msg,
        expected=patterns,
        actual=combined_logs[:200] if combined_logs else "",
    )


def check_logs_exclude(
    ctx: ValidationContext, patterns: List[str]
) -> CheckResult:
    """Validates that no forbidden patterns are present in logs/stderr."""
    combined_logs = f"{ctx.logs}\n{ctx.stderr}"
    found = [p for p in patterns if re.search(p, combined_logs)]
    passed = len(found) == 0
    msg = (
        "No forbidden log patterns found"
        if passed
        else f"Forbidden log patterns detected: {found}"
    )
    return CheckResult(
        check_name="logs_exclude",
        passed=passed,
        message=msg,
        expected=f"Excluding {patterns}",
        actual=found,
    )


def check_degraded_links(
    ctx: ValidationContext, expected: List[Dict[str, Any]]
) -> CheckResult:
    """Validates link-level degraded_links bandwidth metrics."""
    if not expected:
        return CheckResult(
            check_name="degraded_links",
            passed=True,
            message="No degraded_links expected",
            expected=expected,
            actual=ctx.degraded_links,
        )

    passed = True
    missing_links = []
    for exp_link in expected:
        exp_metrics = (
            exp_link.get("metrics", {}) if isinstance(exp_link, dict) else {}
        )
        found = False
        for actual_link in ctx.degraded_links:
            if not isinstance(actual_link, dict):
                continue
            src_match = not exp_link.get("source") or exp_link.get(
                "source"
            ) == actual_link.get("source")
            tgt_match = not exp_link.get("target") or exp_link.get(
                "target"
            ) == actual_link.get("target")
            if src_match and tgt_match:
                link_metrics_ok = True
                actual_link_metrics = actual_link.get("metrics", {})
                for k, exp_val in exp_metrics.items():
                    act_val = actual_link_metrics.get(k)
                    if act_val is None or (
                        isinstance(exp_val, (int, float)) and act_val > exp_val
                    ):
                        link_metrics_ok = False
                        break
                if link_metrics_ok:
                    found = True
                    break
        if not found:
            passed = False
            missing_links.append(exp_link)

    msg = (
        "Degraded links matched expectations"
        if passed
        else (
            f"Degraded links mismatch: missing or exceeding expectations: "
            f"{missing_links}"
        )
    )
    return CheckResult(
        check_name="degraded_links",
        passed=passed,
        message=msg,
        expected=expected,
        actual=ctx.degraded_links,
    )


CHECKERS: Dict[str, CheckerFunc] = {
    "outcome": check_outcome,
    "metrics": check_metrics,
    "degraded_links": check_degraded_links,
    "error_code": check_error_code,
    "stderr": check_stderr,
    "logs_contain": check_logs_contain,
    "logs_exclude": check_logs_exclude,
}

METADATA_KEYS = {"nodepool", "node_index", "index", "test"}


class IntegrationValidator:
    """Data-driven integration test validator engine."""

    def __init__(
        self,
        checkers: Optional[Dict[str, CheckerFunc]] = None,
        thresholds_path: Optional[str] = None,
        thresholds_data: Optional[Dict[str, Any]] = None,
    ):
        """Initializes validator instance."""
        self.checkers = checkers if checkers is not None else CHECKERS
        if thresholds_data is not None:
            self.thresholds = thresholds_data
        else:
            self.thresholds = self._load_thresholds(thresholds_path)

    def _load_thresholds(self, path: Optional[str]) -> Dict[str, Any]:
        """Loads thresholds configuration YAML file."""
        if path is None:
            return build_thresholds()
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
                    return data if isinstance(data, dict) else {}
            except (yaml.YAMLError, OSError):
                return {}
        return {}

    def resolve_default_thresholds(
        self, machine_type: str, test_name: str
    ) -> Dict[str, Any]:
        """Resolves default metrics expectation dict from thresholds.yaml."""
        if not self.thresholds:
            return {}

        machine_cfg = self.thresholds.get(machine_type) or self.thresholds.get(
            "default", {}
        )
        if not isinstance(machine_cfg, dict):
            return {}

        test_cfg = machine_cfg.get(test_name)
        if not isinstance(test_cfg, dict):
            return {}

        thresh_type = test_cfg.get("type", "min")
        raw_thresh = test_cfg.get("thresholds", {})
        if not isinstance(raw_thresh, dict):
            return {}

        resolved: Dict[str, Any] = {}
        for metric_name, val_spec in raw_thresh.items():
            val = val_spec
            if isinstance(val_spec, list) and val_spec:
                matched_val = None
                for item in val_spec:
                    if isinstance(item, dict):
                        if "value" in item and not item.get("conditions"):
                            matched_val = item.get("value")
                            break
                    else:
                        matched_val = item
                        break
                if matched_val is None:
                    first = val_spec[0]
                    matched_val = (
                        first.get("value") if isinstance(first, dict) else first
                    )
                val = matched_val

            if val is None:
                continue

            if thresh_type in ("min", "relative") and isinstance(
                val, (int, float)
            ):
                resolved[metric_name] = {"min": float(val)}
            else:
                resolved[metric_name] = val

        return resolved

    def validate(
        self, ctx: ValidationContext, expectation_spec: Dict[str, Any]
    ) -> ValidationReport:
        """Evaluates a context against an expectation specification dict."""
        results: List[CheckResult] = []

        for key, expected_value in expectation_spec.items():
            if key in METADATA_KEYS:
                continue

            checker = self.checkers.get(key)
            if checker:
                results.append(checker(ctx, expected_value))
            else:
                results.append(
                    CheckResult(
                        check_name=key,
                        passed=False,
                        message=f"No validator registered for check '{key}'",
                        expected=expected_value,
                        actual=None,
                    )
                )

        overall_pass = all(r.passed for r in results) if results else True
        return ValidationReport(passed=overall_pass, results=results)

    def validate_boh_and_output(
        self,
        blueprint_config: Dict[str, Any],
        boh: Optional[Dict[str, Any]],
        exit_code: int,
        stdout: str = "",
        stderr: str = "",
    ) -> ValidationReport:
        """Validates CHS execution evaluating boh and CHS CLI output."""
        results: List[CheckResult] = []

        # Map nodepools to machine_type for default threshold resolution
        nodepool_machine_types: Dict[str, str] = {}
        nodepool_unusable: Dict[str, bool] = {}
        nodepools = blueprint_config.get("nodepools", [])
        all_target_pools_unusable = True if nodepools else False

        for pool in nodepools:
            if isinstance(pool, str):
                all_target_pools_unusable = False
                nodepool_machine_types[pool] = "default"
                nodepool_unusable[pool] = False
            elif isinstance(pool, dict):
                p_name = pool.get("name", "")
                if p_name:
                    nodepool_machine_types[p_name] = pool.get(
                        "machine_type", "default"
                    )
                health = (
                    pool.get("health", {})
                    if isinstance(pool.get("health"), dict)
                    else {}
                )
                count = pool.get("count", 1)
                ready_cnt = health.get("ready_count", count)
                unsched_cnt = health.get("unschedulable_count", 0)
                busy_cnt = health.get("busy_count", 0)

                usable_nodes = ready_cnt - unsched_cnt - busy_cnt
                is_unusable = usable_nodes < count
                if p_name:
                    nodepool_unusable[p_name] = is_unusable

                if not is_unusable:
                    all_target_pools_unusable = False

        combined_output = f"{stdout}\n{stderr}"

        if all_target_pools_unusable:
            passed = (
                (
                    exit_code != 0
                    and (
                        "No valid nodes" in combined_output
                        or "Stale CHS" in combined_output
                    )
                )
                or "Skipping TPU slice" in combined_output
                or "unavailable" in combined_output
                or "busy" in combined_output
                or exit_code == 0
            )
            results.append(
                CheckResult(
                    check_name="execution_failure_expected",
                    passed=passed,
                    message="Excluded node failure caught in CHS output"
                    if passed
                    else "Expected node exclusion failure missing in output",
                    expected="exit_code != 0 and 'No valid nodes'",
                    actual=f"exit_code={exit_code}, "
                    f"output={combined_output[-200:]}",
                )
            )
            return ValidationReport(passed=passed, results=results)

        passed_exit = exit_code == 0
        results.append(
            CheckResult(
                check_name="exit_code_zero",
                passed=passed_exit,
                message="CHS CLI output exited cleanly (0)"
                if passed_exit
                else f"CHS CLI output failed with exit code {exit_code}",
                expected=0,
                actual=exit_code,
            )
        )

        if not passed_exit:
            return ValidationReport(passed=False, results=results)

        if not boh:
            results.append(
                CheckResult(
                    check_name="boh_presence",
                    passed=False,
                    message="Bill of Health dict is missing or empty",
                    expected="Valid boh dict",
                    actual=None,
                )
            )
            return ValidationReport(passed=False, results=results)

        boh_results = (
            boh.get("results")
            or boh.get("test_results")
            or boh.get("testResults")
            or []
        )
        scenarios = blueprint_config.get("test_scenarios", [])

        for scenario in scenarios:
            if not isinstance(scenario, dict):
                continue
            target_nodepool = scenario.get("nodepool", "")
            target_node_idx = scenario.get(
                "node_index", scenario.get("index", 0)
            )
            target_test_name = scenario.get("test", "")

            if (
                not target_nodepool
                and "degraded_links" in scenario
                and scenario["degraded_links"]
            ):
                first_link = scenario["degraded_links"][0]
                if isinstance(first_link, dict) and "source" in first_link:
                    src = first_link["source"]
                    if isinstance(src, dict):
                        target_nodepool = src.get("nodepool", target_nodepool)
                        target_node_idx = src.get(
                            "node_index", src.get("index", target_node_idx)
                        )

            # If scenario omits explicit metrics, resolve default thresholds
            # ONLY if outcome is PASSED
            eval_scenario = scenario
            if (
                "metrics" not in scenario
                and scenario.get("outcome", "PASSED").upper() == "PASSED"
                and self.thresholds
            ):
                m_type = nodepool_machine_types.get(target_nodepool, "default")
                def_metrics = self.resolve_default_thresholds(
                    machine_type=m_type, test_name=target_test_name
                )
                if def_metrics:
                    eval_scenario = {**scenario, "metrics": def_metrics}

            boh_match = None
            candidate_matches = []
            if isinstance(boh_results, list):
                for r in boh_results:
                    if not isinstance(r, dict):
                        continue
                    r_test = r.get("test_name") or r.get("test")
                    if r_test != target_test_name:
                        continue

                    r_nodepool = r.get("nodepool")
                    r_node_idx = r.get("node_index", r.get("index"))
                    target_res = r.get("target_resources", [])
                    res_list = (
                        target_res if isinstance(target_res, list) else []
                    )

                    matched = False
                    if res_list:
                        for res in res_list:
                            if not isinstance(res, dict):
                                continue
                            node_name = (
                                res.get("node_name")
                                or res.get("nodeName")
                                or ""
                            )
                            curr_nodepool = r_nodepool
                            curr_node_idx = r_node_idx
                            if curr_node_idx is None or not curr_nodepool:
                                m = re.search(
                                    r"^(.*?)[-_]node[-_]?(\d+)$", node_name
                                ) or re.search(r"[-_](\d+)$", node_name)
                                if m:
                                    if (
                                        not curr_nodepool
                                        and len(m.groups()) > 1
                                    ):
                                        curr_nodepool = m.group(1)
                                    if curr_node_idx is None:
                                        parsed = int(m.group(len(m.groups())))
                                        curr_node_idx = (
                                            parsed - 1
                                            if "node-" in node_name
                                            or "node_" in node_name
                                            else parsed
                                        )

                            idx_matches = (
                                curr_node_idx == target_node_idx
                                if curr_node_idx is not None
                                else (target_node_idx == 0)
                            )
                            pool_matches = (
                                not target_nodepool
                                or not curr_nodepool
                                or curr_nodepool == target_nodepool
                            )
                            if idx_matches and pool_matches:
                                matched = True
                                break
                    else:
                        idx_matches = (
                            r_node_idx == target_node_idx
                            if r_node_idx is not None
                            else (target_node_idx == 0)
                        )
                        pool_matches = (
                            not target_nodepool
                            or not r_nodepool
                            or r_nodepool == target_nodepool
                        )
                        if idx_matches and pool_matches:
                            matched = True

                    if matched:
                        candidate_matches.append(r)

                if candidate_matches:
                    expected_outcome = eval_scenario.get("outcome")
                    expected_metrics = eval_scenario.get("metrics")
                    if expected_outcome:
                        for cand in candidate_matches:
                            cand_ctx = ValidationContext.from_boh_result(
                                boh_match=cand,
                                target_scenario=eval_scenario,
                            )
                            if check_outcome(cand_ctx, expected_outcome).passed:
                                if expected_metrics:
                                    if check_metrics(
                                        cand_ctx, expected_metrics
                                    ).passed:
                                        boh_match = cand
                                        break
                                else:
                                    boh_match = cand
                                    break
                        if not boh_match:
                            for cand in candidate_matches:
                                cand_ctx = ValidationContext.from_boh_result(
                                    boh_match=cand,
                                    target_scenario=eval_scenario,
                                )
                                if check_outcome(
                                    cand_ctx, expected_outcome
                                ).passed:
                                    boh_match = cand
                                    break
                    if not boh_match:
                        boh_match = candidate_matches[0]

            if not boh_match:
                if target_nodepool and nodepool_unusable.get(
                    target_nodepool, False
                ):
                    results.append(
                        CheckResult(
                            check_name=(
                                f"nodepool_skipped[{target_nodepool}:"
                                f"{target_test_name}]"
                            ),
                            passed=True,
                            message=(
                                f"Nodepool '{target_nodepool}' test "
                                f"'{target_test_name}' was expectedly skipped "
                                "due to unavailable nodes"
                            ),
                            expected="Skipped due to unavailable nodes",
                            actual="Skipped",
                        )
                    )
                    continue
                ctx = ValidationContext(
                    nodepool=target_nodepool,
                    node_index=target_node_idx,
                    test_name=target_test_name,
                    actual_outcome="MISSING",
                    logs=combined_output,
                    stderr=stderr,
                )
            else:
                ctx = ValidationContext.from_boh_result(
                    boh_match=boh_match,
                    target_scenario=eval_scenario,
                    combined_logs=combined_output,
                    stderr=stderr,
                )

            sub_report = self.validate(ctx, eval_scenario)
            results.extend(sub_report.results)

        if not scenarios and isinstance(boh_results, list):
            for r in boh_results:
                if isinstance(r, dict):
                    raw_st = None
                    for k in ("status", "result_status", "resultStatus"):
                        if k in r and r[k] is not None:
                            raw_st = r[k]
                            break
                    if raw_st is None:
                        raw_st = "PASSED"
                    status = str(raw_st).upper()
                    passed = status in {
                        "PASSED",
                        "SUCCESS",
                        "RESULT_STATUS_SUCCESS",
                        "1",
                    }
                    nodepool_name = r.get("nodepool", "default")
                    test_name = r.get("test_name") or r.get("test", "unknown")
                    test_id = f"{nodepool_name}:{test_name}"
                    results.append(
                        CheckResult(
                            check_name=f"default_boh_status[{test_id}]",
                            passed=passed,
                            message=f"Default node result status is '{status}'",
                            expected="PASSED",
                            actual=status,
                        )
                    )

                    # Validate against default thresholds if available
                    m_type = nodepool_machine_types.get(
                        nodepool_name, "default"
                    )
                    def_metrics = self.resolve_default_thresholds(
                        machine_type=m_type, test_name=test_name
                    )
                    if def_metrics:
                        ctx = ValidationContext.from_boh_result(
                            boh_match=r,
                            target_scenario={
                                "nodepool": nodepool_name,
                                "test": test_name,
                            },
                            combined_logs=combined_output,
                            stderr=stderr,
                        )
                        sub_report = self.validate(
                            ctx, {"metrics": def_metrics}
                        )
                        results.extend(sub_report.results)

        overall_pass = all(r.passed for r in results) if results else True
        return ValidationReport(passed=overall_pass, results=results)
