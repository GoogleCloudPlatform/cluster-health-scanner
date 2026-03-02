"""Generates Bill of Health from Ramble JSON results."""

from datetime import datetime, timezone
import json
import os
from typing import Any
import uuid

from chs.protos import bill_of_health_pb2 as pb
from google.protobuf.timestamp_pb2 import Timestamp


class BOHGenerator:
    """Generates Bill of Health from Ramble JSON results."""

    def __init__(
        self,
        file_path: str,
        base_log_dir: str | None = None,
        cluster_type: str = "slurm",
    ):
        self.file_path = file_path
        self.base_log_dir = base_log_dir
        self.cluster_type = cluster_type
        self.report = pb.BillOfHealth()
        self._extractors = {
            "nccl-tests": self._extract_nccl_info,
            "nvidia-dcgm": self._extract_dcgm_info,
        }

    def _get_timestamp(self) -> Timestamp:
        ts = Timestamp()
        ts.FromDatetime(datetime.now(timezone.utc))
        return ts

    def _parse_foms(self, foms: list[dict[str, Any]]) -> list[pb.Metric]:
        """Extracts metrics from list of FOM objects."""
        metrics = []
        for fom in foms:
            key = fom.get("name", "unknown")
            val = str(fom.get("value", ""))
            unit = fom.get("units", "")

            metric = pb.Metric(key=key)
            if unit:
                metric.unit = unit

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

    def _extract_nccl_info(
        self, exp_name: str, workload_name: str
    ) -> tuple[str, str]:
        """Extractor for NCCL tests."""
        if "nccl-cluster" in exp_name:
            return "nccl-cluster", workload_name
        if "nccl-pairwise" in exp_name:
            return "nccl-pairwise", workload_name
        return exp_name, workload_name

    def _extract_dcgm_info(
        self, exp_name: str, workload_name: str
    ) -> tuple[str, str]:
        """Extractor for DCGM tests."""
        parts = exp_name.split("-")
        if len(parts) > 3 and parts[2] == "level":
            return "dcgm", f"level-{parts[3]}"
        return "dcgm", workload_name

    def _extract_test_info(self, exp: dict[str, Any]) -> tuple[str, str]:
        """Extracts test name and level from experiment data."""
        exp_name = exp.get("experiment_name", "test_name_unidentified")
        app_name = exp.get("application_name", "")
        workload_name = exp.get("workload_name", "")

        extractor = self._extractors.get(app_name)
        if extractor:
            return extractor(exp_name, workload_name)

        return exp_name, workload_name

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

    def generate(self) -> pb.BillOfHealth:
        """Generates the Bill of Health report."""
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            # TODO: Replace print statements with proper logging statements
            print(f"Error reading JSON results: {e}")
            return self.report

        self.report.report_id = data.get("workspace_hash", str(uuid.uuid4()))
        self.report.chs_version = "1.0.0"
        self.report.created_at.CopyFrom(self._get_timestamp())
        # TODO: Fetch and add system info
        self.report.system_info.type = f"{self.cluster_type.upper()}_CLUSTER"

        for exp in data.get("experiments", []):
            exp_name = exp.get("experiment_name", "")
            # Experiment cluster_health_diagnostics is used to inject other
            # experiments, not required for BoH; hardcoding the skip for now.
            if exp_name == "cluster_health_diagnostics":
                continue

            test_result = pb.TestResult()
            test_result.test_name, test_result.test_level = (
                self._extract_test_info(exp)
            )
            test_result.result_status = self._get_result_status(
                exp.get("SUCCESS_CRITERIA", {})
            )

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
                    workspace_dir = os.path.dirname(self.file_path)
                    rel_path = os.path.relpath(log_file, workspace_dir)
                    log_file = os.path.join(self.base_log_dir, rel_path)
                test_result.metadata.fields["log_file"].string_value = log_file
            else:
                test_result.metadata.fields["log_file"].string_value = ""

            node = ramble_vars.get("node")
            if node:
                test_result.target_resource_ids.append(node)
            else:
                nodes = ramble_raw_vars.get("nodes", [])
                test_result.target_resource_ids.extend(nodes)

            # Only including default (null) and the last context.
            contexts = exp.get("CONTEXTS", [])
            if contexts:
                # Default Context
                default_ctx = next(
                    (c for c in contexts if c.get("name") == "null"),
                    None,
                )

                # Last Context (if different from default)
                last_ctx = contexts[-1]

                contexts_to_process = []
                if default_ctx:
                    contexts_to_process.append(default_ctx)

                if last_ctx and last_ctx != default_ctx:
                    contexts_to_process.append(last_ctx)

                for ctx in contexts_to_process:
                    foms = ctx.get("foms", [])
                    metrics = self._parse_foms(foms)
                    test_result.metrics.extend(metrics)

            self.report.test_results.append(test_result)

        return self.report
