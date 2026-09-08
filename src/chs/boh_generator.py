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

"""Generates Bill of Health from Ramble JSON results."""

from datetime import datetime, timezone
import json
import logging
import os
import re
import chs.core.plugin_manager
from typing import Any
import uuid
import yaml
from chs.protos import bill_of_health_pb2 as pb
from chs.tier_manager import TestStatus
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf import json_format
from chs.utils.app_data import ChsAppData
from chs.utils import lifecycle


logger = logging.getLogger(__name__)


class BOHGenerator:
    """Generates Bill of Health from Ramble JSON results."""

    def __init__(
        self,
        file_path: str,
        base_log_dir: str | None = None,
        cluster_type: str = "slurm",
        success_criteria_mapping: (
            dict[str, dict[str, dict[str, str]]] | None
        ) = None,
        gcs_base_path: str | None = None,
        thresholds_path: str | None = None,
        skip_validation: bool = False,
    ):
        self.file_path = file_path
        self.base_log_dir = base_log_dir
        self.cluster_type = cluster_type
        self.report = pb.BillOfHealth()
        self.success_criteria_mapping = success_criteria_mapping or {}
        self.pm = chs.core.plugin_manager.PluginManager()
        self.gcs_base_path = gcs_base_path
        self.thresholds_path = thresholds_path
        self.skip_validation = skip_validation
        self.raw_thresholds = {} if skip_validation else self._load_thresholds()
        self.app_data = ChsAppData()

    def _load_thresholds(self) -> dict:
        """Loads thresholds from workspace or default path."""
        path = self.thresholds_path
        explicit = bool(path)
        if not path and self.file_path:
            workspace_dir = os.path.dirname(self.file_path)
            path = os.path.join(workspace_dir, "thresholds.yaml")

        if path:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        logger.debug("Loading thresholds from %s", path)
                        data = yaml.safe_load(f) or {}
                        return data
                except (yaml.YAMLError, OSError) as e:
                    logger.warning(
                        "Failed to load performance thresholds "
                        "configuration file from '%s'. Error: %s",
                        path,
                        e,
                    )
                    if explicit:
                        raise ValueError(
                            f"Failed to load explicit thresholds file {path}: "
                            f"{e}"
                        ) from e
            else:
                if explicit:
                    raise FileNotFoundError(
                        f"Explicit thresholds file not found at {path}"
                    )

        internal_path = os.path.join(
            os.path.dirname(__file__), "configs", "thresholds.yaml"
        )
        if os.path.exists(internal_path):
            try:
                with open(internal_path, "r") as f:
                    logger.debug(
                        "Loading default internal thresholds from %s",
                        internal_path,
                    )
                    data = yaml.safe_load(f) or {}
                    return data
            except (yaml.YAMLError, OSError) as e:
                logger.debug("Failed to load internal thresholds: %s", e)
                logger.warning(
                    "Unable to find thresholds. The final report will only "
                    "display raw metrics without pass/fail evaluation."
                )
        return {}

    def _get_timestamp(self) -> Timestamp:
        ts = Timestamp()
        ts.FromDatetime(datetime.now(timezone.utc))
        return ts

    def _parse_foms(self, foms: list[dict[str, Any]]) -> list[pb.Metric]:
        """Extracts metrics from list of FOM objects."""
        metrics = []
        for fom in foms:
            key = fom.get("name", "unknown")
            raw_val = fom.get("value")
            unit = fom.get("units", "")

            metric = pb.Metric(key=key)
            if unit:
                metric.unit = unit

            if isinstance(raw_val, bool):
                metric.value.bool_value = raw_val
            elif isinstance(raw_val, (int, float)):
                metric.value.number_value = float(raw_val)
            elif raw_val is not None:
                val = str(raw_val)
                lower_val = val.lower()
                if lower_val == "true":
                    metric.value.bool_value = True
                elif lower_val == "false":
                    metric.value.bool_value = False
                else:
                    try:
                        metric.value.number_value = float(val)
                    except ValueError:
                        metric.value.string_value = val

            metrics.append(metric)
        return metrics

    def _evaluate_metric_conditions(
        self,
        metric: pb.Metric | None,
        conditions: dict[str, Any],
        aliases: dict[str, str],
        ramble_vars: dict[str, Any] = None,
    ) -> bool:
        """Evaluates conditions against run variables and metric context."""
        if not conditions:
            return True

        metric_context = (
            metric.metadata.fields["context"].string_value
            if metric and "context" in metric.metadata.fields
            else ""
        )
        context_parts = (
            [part.strip() for part in metric_context.split(",")]
            if metric_context
            else []
        )

        for condition_name, expected_condition_value in conditions.items():
            resolved_condition_name = aliases.get(
                condition_name, condition_name
            )

            if ramble_vars and resolved_condition_name in ramble_vars:
                if str(ramble_vars[resolved_condition_name]) != str(
                    expected_condition_value
                ):
                    return False
            elif metric is not None:
                if str(expected_condition_value) not in context_parts:
                    return False
        return True

    def _split_test_result_per_node(
        self, test_result: pb.TestResult
    ) -> list[pb.TestResult]:
        split_results = []
        for target in test_result.target_resources:
            node_result = pb.TestResult()
            node_result.CopyFrom(test_result)

            node_result.ClearField("target_resources")
            node_result.target_resources.add(
                node_name=target.node_name, instance_id=target.instance_id
            )

            node_metrics = []
            target_name = target.node_name.strip()
            pattern = (
                r"(?<![a-zA-Z0-9_-])"
                + re.escape(target_name)
                + r"(?![a-zA-Z0-9_-])"
            )

            for metric in node_result.metrics:
                ctx = ""
                if "context" in metric.metadata.fields:
                    ctx = metric.metadata.fields["context"].string_value.strip()

                if "node" in metric.metadata.fields:
                    metric_node = metric.metadata.fields[
                        "node"
                    ].string_value.strip()
                    if metric_node == target_name:
                        node_metrics.append(metric)
                elif not ctx or ctx == "null":
                    node_metrics.append(metric)
                elif re.search(pattern, ctx):
                    node_metrics.append(metric)

            node_result.ClearField("metrics")
            node_result.metrics.extend(node_metrics)

            split_results.append(node_result)
        return split_results

    def _evaluate_thresholds(
        self,
        machine_type: str,
        test_result: pb.TestResult,
        ramble_vars: dict[str, Any] = None,
    ) -> None:
        """Evaluates metrics against thresholds defined in thresholds.yaml."""
        test_name = test_result.test_name

        default_cfg = self.raw_thresholds.get("default", {}).get(test_name, {})
        machine_cfg = self.raw_thresholds.get(machine_type, {}).get(
            test_name, {}
        )

        if not default_cfg and not machine_cfg:
            test_result.result_status = pb.RESULT_STATUS_UNKNOWN
            return

        merged_thresholds = {}
        merged_thresholds.update(default_cfg.get("thresholds", {}))
        merged_thresholds.update(machine_cfg.get("thresholds", {}))

        comp_type = machine_cfg.get("type") or default_cfg.get("type")
        if not comp_type:
            logger.debug(
                "Missing comparison type configuration for test "
                "'%s'. Validation will be skipped.",
                test_name,
            )
            test_result.result_status = pb.RESULT_STATUS_UNKNOWN
            return

        resolved_name = self.pm.resolve_test_name(test_name)
        plugin = self.pm.get_plugin(resolved_name or test_name)
        aliases = plugin.get_fom_aliases()
        relative_pairs = plugin.get_relative_fom_keys()

        overall_success = True
        metrics_map = {m.key: m for m in test_result.metrics}

        flattened_thresholds = []
        for metric_name, metric_threshold_configs in merged_thresholds.items():
            threshold_config_list = (
                metric_threshold_configs
                if isinstance(metric_threshold_configs, list)
                else [metric_threshold_configs]
            )

            for threshold_config in threshold_config_list:
                if isinstance(threshold_config, dict):
                    threshold_value = threshold_config.get("value")
                    conditions = threshold_config.get("conditions", {})
                else:
                    threshold_value = threshold_config
                    conditions = {}

                # Reduce numeric TPU test thresholds by 5% at runtime
                # to account for typical performance variance and our
                # rounding logic requirements.
                if (
                    test_name.startswith("tpu-")
                    and isinstance(threshold_value, (int, float))
                    and comp_type == "min"
                ):
                    reduced_value = float(threshold_value) * 0.95
                    threshold_value = round(reduced_value)

                if not self._evaluate_metric_conditions(
                    None, conditions, aliases, ramble_vars
                ):
                    continue

                flattened_thresholds.append(
                    (metric_name, threshold_value, conditions)
                )

        if not flattened_thresholds:
            test_result.result_status = pb.RESULT_STATUS_UNKNOWN
            return

        if comp_type == "relative":
            for th_key, th_val, conditions in flattened_thresholds:
                validation = pb.ValidationCriteria(
                    comparator=pb.Comparator.COMPARATOR_GTE, name=th_key
                )
                validation.expected_value.number_value = float(th_val)

                for pair in relative_pairs:
                    ach_key, peak_key = pair
                    peak_metric = metrics_map.get(peak_key)
                    if not peak_metric:
                        logger.debug(
                            "Missing metric: Peak metric '%s' is "
                            "required for the "
                            "relative threshold but was not found",
                            peak_key,
                        )
                        overall_success = False
                        continue

                    achieved_metric_found = False
                    for m in test_result.metrics:
                        if m.key == ach_key:
                            achieved_metric_found = True
                            if not self._evaluate_metric_conditions(
                                m,
                                conditions,
                                aliases,
                                ramble_vars,
                            ):
                                continue
                            ach_val = (
                                m.value.number_value
                                if m.value.HasField("number_value")
                                else 0
                            )
                            peak_val = (
                                peak_metric.value.number_value
                                if peak_metric.value.HasField("number_value")
                                else 0
                            )

                            if peak_val > 0:
                                ratio = ach_val / peak_val
                                is_pass = ratio >= float(th_val)
                            else:
                                ratio = 0
                                is_pass = False

                            m.metadata.fields["relative"].number_value = ratio
                            m.metadata.fields[
                                "relative_formula"
                            ].string_value = f"{ach_key} / {peak_key}"

                            m_validation = pb.ValidationCriteria()
                            m_validation.CopyFrom(validation)
                            m_validation.result_status = (
                                pb.RESULT_STATUS_SUCCESS
                                if is_pass
                                else pb.RESULT_STATUS_FAILED
                            )
                            m.validation_criteria.CopyFrom(m_validation)

                            if not is_pass:
                                overall_success = False

                    if not achieved_metric_found:
                        overall_success = False

        else:
            for th_key, th_val, conditions in flattened_thresholds:
                raw_fom_name = aliases.get(th_key, th_key)

                found_metric = False
                for metric in test_result.metrics:
                    if metric.key == raw_fom_name:
                        found_metric = True
                        if not self._evaluate_metric_conditions(
                            metric,
                            conditions,
                            aliases,
                            ramble_vars,
                        ):
                            continue
                        validation = pb.ValidationCriteria(name=th_key)
                        if comp_type == "min":
                            validation.comparator = pb.Comparator.COMPARATOR_GTE
                        elif comp_type == "scalar":
                            validation.comparator = pb.Comparator.COMPARATOR_EQ

                        if isinstance(th_val, bool):
                            validation.expected_value.bool_value = th_val
                        elif isinstance(th_val, (int, float)):
                            validation.expected_value.number_value = float(
                                th_val
                            )
                        else:
                            validation.expected_value.string_value = str(th_val)

                        is_pass = False
                        if metric.value.HasField("number_value") and isinstance(
                            th_val, (int, float)
                        ):
                            obs_val = metric.value.number_value
                            if comp_type == "min":
                                is_pass = obs_val >= float(th_val)
                            elif comp_type == "scalar":
                                is_pass = obs_val == float(th_val)
                        elif metric.value.HasField("bool_value") and isinstance(
                            th_val, bool
                        ):
                            obs_val = metric.value.bool_value
                            is_pass = obs_val == th_val
                        elif metric.value.HasField("string_value"):
                            obs_val = metric.value.string_value
                            is_pass = obs_val.lower() == str(th_val).lower()

                        validation.result_status = (
                            pb.RESULT_STATUS_SUCCESS
                            if is_pass
                            else pb.RESULT_STATUS_FAILED
                        )
                        metric.validation_criteria.CopyFrom(validation)

                        if not is_pass:
                            overall_success = False

                if not found_metric:
                    overall_success = False

        if overall_success and not any(
            m.HasField("validation_criteria") for m in test_result.metrics
        ):
            test_result.result_status = pb.RESULT_STATUS_UNKNOWN
            return

        test_result.result_status = (
            pb.RESULT_STATUS_SUCCESS
            if overall_success
            else pb.RESULT_STATUS_FAILED
        )

    def _extract_test_info(self, exp: dict[str, Any]) -> tuple[str, str, str]:
        """Extracts test name and level, delegating to plugins if available."""
        exp_name = exp.get("experiment_name", "test_name_unidentified")
        app_name = exp.get("application_name", "")

        resolved_name = self.pm.resolve_test_name(exp_name)
        if resolved_name is None and app_name:
            resolved_name = self.pm.resolve_test_name(app_name)

        plugin = self.pm.get_plugin(resolved_name or exp_name)

        return plugin.extract_test_info(exp)

    def _get_result_status(self, success_criteria: dict[str, str]) -> int:
        """Maps experiment success criteria to ResultStatus."""
        if not success_criteria:
            return pb.RESULT_STATUS_UNKNOWN

        statuses = {s.upper() for s in success_criteria.values()}

        if "FAILED" in statuses:
            return pb.RESULT_STATUS_FAILED
        if "TIMEOUT" in statuses:
            return pb.RESULT_STATUS_TIMEOUT
        if "CANCELLED" in statuses:
            return pb.RESULT_STATUS_CANCELLED
        if statuses == {"PASSED"}:
            return pb.RESULT_STATUS_SUCCESS

        return pb.RESULT_STATUS_UNKNOWN

    def _add_system_info_from_app_data(self) -> None:
        """Adds system information to the report from ChsAppData."""
        try:
            run_data = self.app_data.get_run(self.report.report_id)
            if run_data:
                command_statement = run_data.get("command_statement", "")
                if command_statement:
                    self.report.system_info.metadata.fields[
                        "command_statement"
                    ].string_value = command_statement
        except Exception as e:
            logger.debug(
                "Failed to retrieve system information from "
                "application data cache. Error: %s",
                e,
            )

    def generate(self) -> pb.BillOfHealth:
        """Generates the Bill of Health report."""
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.debug("Error reading JSON results: %s", e)
            logger.error(
                "Failed to read raw diagnostic results. The final Bill "
                "of Health report cannot be generated."
            )
            return self.report

        self.report.report_id = data.get("workspace_name", str(uuid.uuid4()))
        self.report.chs_version = "1.0.0"
        self.report.created_at.CopyFrom(self._get_timestamp())

        self.report.system_info.type = f"{self.cluster_type.upper()}_CLUSTER"
        if self.skip_validation:
            self.report.metadata.fields["skip_validation"].bool_value = True

        self._add_system_info_from_app_data()

        run_data = self.app_data.get_run(self.report.report_id) or {}
        job_entries = run_data.get("job_entries", {})
        executed_exp_dirs = {
            job.get("experiment_dir")
            for job in job_entries.values()
            if job.get("experiment_dir")
        }

        for exp in data.get("experiments", []):
            exp_name = exp.get("experiment_name", "")
            # Experiment chs is used to inject other experiments;
            # hardcoding the skip for now.
            if exp_name == "chs":
                continue

            # Skip experiments that were never submitted for execution
            exp_dir = exp.get("RAMBLE_VARIABLES", {}).get("experiment_run_dir")
            if (
                executed_exp_dirs
                and exp_dir
                and exp_dir not in executed_exp_dirs
            ):
                continue

            test_result = pb.TestResult()
            desc = ""
            (
                test_result.test_name,
                test_result.test_level,
                desc,
            ) = self._extract_test_info(exp)

            if desc:
                test_result.metadata.fields["description"].string_value = desc
            metrics_map = {}

            tags = exp.get("TAGS", [])
            if tags:
                test_result.metadata.fields["tags"].list_value.extend(tags)

            ramble_vars = exp.get("RAMBLE_VARIABLES", {})
            ramble_raw_vars = exp.get("RAMBLE_RAW_VARIABLES", {})

            log_file = ramble_vars.get("log_file")
            # TODO: Update log file link to point to the actual log file in
            # bucket. Currently, it points to log files inside workspace. I'll
            # update it after the followup PR on uploading logs to bucket.
            if log_file:
                if self.base_log_dir:
                    log_file = os.path.relpath(log_file, self.base_log_dir)
                test_result.metadata.fields["log_file"].string_value = log_file
            else:
                test_result.metadata.fields["log_file"].string_value = ""
            if self.gcs_base_path:
                gcs_path = f"{self.gcs_base_path}/{log_file}"
                test_result.metadata.fields[
                    "gcs_log_file"
                ].string_value = gcs_path

            run_data = self.app_data.get_run(self.report.report_id) or {}
            node_mapping = run_data.get("node_to_instance_id", {})

            node = ramble_vars.get("node")
            if node:
                inst_id = node_mapping.get(node, "")
                test_result.target_resources.add(
                    node_name=node, instance_id=inst_id
                )

            else:
                nodes = ramble_raw_vars.get("nodes", [])
                for n in nodes:
                    inst_id = node_mapping.get(n, "")
                    test_result.target_resources.add(
                        node_name=n, instance_id=inst_id
                    )

            resolved_name = self.pm.resolve_test_name(test_result.test_name)
            plugin = self.pm.get_plugin(resolved_name)

            contexts = exp.get("CONTEXTS", [])
            if contexts:
                contexts_to_process = plugin.get_required_contexts(contexts)

                for ctx in contexts_to_process:
                    foms = ctx.get("foms", [])
                    metrics = self._parse_foms(foms)

                    ctx_name = ctx.get("name", "unknown")
                    parsed_metadata = (
                        plugin.parse_context_metadata(ctx_name) or {}
                    )
                    context = parsed_metadata.get("context", ctx_name)
                    if context in ("null", "unknown", None):
                        context = ""
                    parsed_metadata["context"] = context

                    for metric in metrics:
                        for k, v in parsed_metadata.items():
                            if isinstance(v, bool):
                                metric.metadata.fields[k].bool_value = v
                            elif isinstance(v, (int, float)):
                                metric.metadata.fields[k].number_value = float(
                                    v
                                )
                            else:
                                metric.metadata.fields[k].string_value = str(v)

                    test_result.metrics.extend(metrics)

            for metric in test_result.metrics:
                metrics_map[metric.key] = metric

            job_entries = run_data.get("job_entries", {})
            matching_entry = next(
                (
                    entry
                    for entry in job_entries.values()
                    if os.path.basename(entry.get("experiment_dir", ""))
                    == exp_name
                ),
                {},
            )
            job_status = matching_entry.get("status", "UNKNOWN").upper()
            machine_type = ramble_vars.get("machine_type", "default")

            if job_status in lifecycle.TIME_EXCEEDED_PB_STATUSES:
                test_result.result_status = lifecycle.TIME_EXCEEDED_PB_STATUSES[
                    job_status
                ]
            elif self.skip_validation:
                test_result.result_status = pb.RESULT_STATUS_UNKNOWN

            needs_evaluation = not (
                job_status in lifecycle.TIME_EXCEEDED_PB_STATUSES
                or self.skip_validation
            )

            results_to_add = [test_result]
            if (
                plugin.creates_per_node_results()
                and len(test_result.target_resources) > 1
            ):
                results_to_add = self._split_test_result_per_node(test_result)

            for result in results_to_add:
                if hasattr(plugin, "post_process_test_result"):
                    plugin.post_process_test_result(result)

                if needs_evaluation:
                    self._evaluate_thresholds(
                        machine_type, result, exp | ramble_vars
                    )
                self.report.test_results.append(result)

        return self.report


def parse_tier_results_from_boh(
    boh: pb.BillOfHealth | str, expected_tests: set[str]
) -> tuple[dict[str, TestStatus], bool]:
    """
    Parses the protobuf into a dictionary of explicit TestStatus enums.
    Missing data defaults to ERROR.
    """
    if isinstance(boh, str):
        with open(boh, "r") as f:
            boh_json = f.read()
        boh_pb = pb.BillOfHealth()
        json_format.Parse(boh_json, boh_pb)
        boh = boh_pb

    tier_results = {test: TestStatus.ERROR for test in expected_tests}
    has_hard_failure = False

    for result in boh.test_results:
        test = result.test_name
        if test in expected_tests:
            if result.result_status == pb.ResultStatus.RESULT_STATUS_FAILED:
                tier_results[test] = TestStatus.DEGRADED
            elif result.result_status == pb.ResultStatus.RESULT_STATUS_SUCCESS:
                if tier_results[test] != TestStatus.DEGRADED:
                    tier_results[test] = TestStatus.HEALTHY

        for m in result.metrics:
            if (
                m.key.endswith("Error Detected")
                or m.key.endswith("Failure Detected")
                or "Warning" in m.key
            ):
                has_hard_failure = True

    return tier_results, has_hard_failure
