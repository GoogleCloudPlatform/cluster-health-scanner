"""Generates Bill of Health from Ramble JSON results."""

from datetime import datetime, timezone
import json
import logging
import os
import chs.core.plugin_manager
from typing import Any
import uuid
import yaml
from chs.protos import bill_of_health_pb2 as pb
from google.protobuf.timestamp_pb2 import Timestamp
from chs.utils.app_data import ChsAppData

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
                        logger.info("Loading thresholds from %s", path)
                        data = yaml.safe_load(f) or {}
                        return data
                except (yaml.YAMLError, OSError) as e:
                    logger.warning(
                        "Failed to load thresholds from %s: %s", path, e
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
                logger.warning("Thresholds file not found at %s", path)

        internal_path = os.path.join(
            os.path.dirname(__file__), "configs", "thresholds.yaml"
        )
        if os.path.exists(internal_path):
            try:
                with open(internal_path, "r") as f:
                    logger.info(
                        "Loading default internal thresholds from %s",
                        internal_path,
                    )
                    data = yaml.safe_load(f) or {}
                    return data
            except (yaml.YAMLError, OSError) as e:
                logger.warning("Failed to load internal thresholds: %s", e)
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

    def _evaluate_thresholds(
        self, machine_type: str, test_result: pb.TestResult
    ) -> None:
        """Evaluates metrics against thresholds defined in thresholds.yaml."""
        test_name = test_result.test_name

        default_cfg = self.raw_thresholds.get("default", {}).get(test_name, {})
        machine_cfg = self.raw_thresholds.get(machine_type, {}).get(
            test_name, {}
        )

        if not default_cfg and not machine_cfg:
            self._fallback_string_match(test_result)
            return

        merged_thresholds = {}
        merged_thresholds.update(default_cfg.get("thresholds", {}))
        merged_thresholds.update(machine_cfg.get("thresholds", {}))

        comp_type = machine_cfg.get("type") or default_cfg.get("type")
        if not comp_type:
            logger.warning("No comparison type defined for test %s", test_name)
            test_result.result_status = pb.RESULT_STATUS_UNKNOWN
            return

        resolved_name = self.pm.resolve_test_name(test_name)
        plugin = self.pm.get_plugin(resolved_name or test_name)
        aliases = plugin.get_fom_aliases()
        relative_pairs = plugin.get_relative_fom_keys()

        overall_success = True
        metrics_map = {m.key: m for m in test_result.metrics}

        if comp_type == "relative":
            for th_key, th_val in merged_thresholds.items():
                validation = pb.ValidationCriteria(
                    comparator=pb.Comparator.COMPARATOR_GTE, name=th_key
                )
                validation.expected_value.number_value = float(th_val)

                for pair in relative_pairs:
                    ach_key, peak_key = pair
                    peak_metric = metrics_map.get(peak_key)
                    if not peak_metric:
                        logger.warning(
                            "Peak metric %s not found for relative check",
                            peak_key,
                        )
                        continue

                    for m in test_result.metrics:
                        if m.key == ach_key:
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

        else:
            for th_key, th_val in merged_thresholds.items():
                raw_fom_name = aliases.get(th_key, th_key)

                found_metric = False
                for metric in test_result.metrics:
                    if metric.key == raw_fom_name:
                        found_metric = True
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
                    logger.warning(
                        "Metric %s (alias for %s) not found in results",
                        raw_fom_name,
                        th_key,
                    )

        test_result.result_status = (
            pb.RESULT_STATUS_SUCCESS
            if overall_success
            else pb.RESULT_STATUS_FAILED
        )

    def _fallback_string_match(self, test_result: pb.TestResult) -> None:
        """Fallback: search string metrics for 'PASS' (case-insensitive)."""
        overall_success = False
        found_string_metric = False

        for m in test_result.metrics:
            if m.value.HasField("string_value"):
                found_string_metric = True
                val = m.value.string_value
                if "pass" in val.lower():
                    overall_success = True
                    validation = pb.ValidationCriteria(
                        comparator=pb.Comparator.COMPARATOR_CONTAINS,
                        result_status=pb.RESULT_STATUS_SUCCESS,
                        name="default_fallback",
                    )
                    validation.expected_value.string_value = "PASS"
                    m.validation_criteria.CopyFrom(validation)
                elif "fail" in val.lower():
                    validation = pb.ValidationCriteria(
                        comparator=pb.Comparator.COMPARATOR_CONTAINS,
                        result_status=pb.RESULT_STATUS_FAILED,
                        name="default_fallback",
                    )
                    validation.expected_value.string_value = "PASS"
                    m.validation_criteria.CopyFrom(validation)
                    overall_success = False
                    break

        if found_string_metric:
            test_result.result_status = (
                pb.RESULT_STATUS_SUCCESS
                if overall_success
                else pb.RESULT_STATUS_FAILED
            )
        else:
            test_result.result_status = pb.RESULT_STATUS_SUCCESS

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
            logger.warning("Failed to add system info from app_data: %s", e)

    def generate(self) -> pb.BillOfHealth:
        """Generates the Bill of Health report."""
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error("Error reading JSON results: %s", e)
            return self.report

        self.report.report_id = data.get("workspace_name", str(uuid.uuid4()))
        self.report.chs_version = "1.0.0"
        self.report.created_at.CopyFrom(self._get_timestamp())

        self.report.system_info.type = f"{self.cluster_type.upper()}_CLUSTER"

        self._add_system_info_from_app_data()

        for exp in data.get("experiments", []):
            exp_name = exp.get("experiment_name", "")
            # Experiment chs is used to inject other experiments;
            # hardcoding the skip for now.
            if exp_name == "chs":
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

            node = ramble_vars.get("node")
            if node:
                test_result.target_resource_ids.append(node)
            else:
                nodes = ramble_raw_vars.get("nodes", [])
                test_result.target_resource_ids.extend(nodes)

            contexts = exp.get("CONTEXTS", [])
            if contexts:
                resolved_name = self.pm.resolve_test_name(test_result.test_name)
                plugin = self.pm.get_plugin(resolved_name)

                contexts_to_process = plugin.get_required_contexts(contexts)

                for ctx in contexts_to_process:
                    foms = ctx.get("foms", [])
                    metrics = self._parse_foms(foms)

                    ctx_name = ctx.get("name", "unknown")
                    for metric in metrics:
                        metric.metadata.fields[
                            "context"
                        ].string_value = ctx_name

                    test_result.metrics.extend(metrics)

            for metric in test_result.metrics:
                metrics_map[metric.key] = metric

            machine_type = ramble_vars.get("machine_type", "default")
            if not self.skip_validation:
                self._evaluate_thresholds(machine_type, test_result)
            else:
                test_result.result_status = pb.RESULT_STATUS_UNKNOWN

            self.report.test_results.append(test_result)

        return self.report
