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

  def __init__(self, file_path: str, base_log_dir: str | None = None):
    self.file_path = file_path
    self.base_log_dir = base_log_dir
    self.report = pb.BillOfHealth()

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
    self.report.system_info.type = "SLURM_CLUSTER"

    experiments = data.get("experiments", [])
    for exp in experiments:
      test_result = pb.TestResult()
      test_result.name = exp.get("experiment_name", "test_name_unidentified")

      # Experiment cluster_health_diagnostics is used to inject other
      # experiments, not required for BoH, hardcoding the skip for now.
      # TODO: Make this more robust once other experiments (gke, custom, etc)
      # are added.
      if test_result.name == "cluster_health_diagnostics":
        continue

      status_str = exp.get("EXPERIMENT_STATUS", "UNSPECIFIED").upper()
      if status_str == "SUCCESS":
        test_result.result_status = pb.RESULT_STATUS_SUCCESS
      elif status_str == "FAILED":
        test_result.result_status = pb.RESULT_STATUS_FAILED
      elif status_str == "TIMEOUT":
        test_result.result_status = pb.RESULT_STATUS_TIMEOUT
      elif status_str == "CANCELLED":
        test_result.result_status = pb.RESULT_STATUS_CANCELLED
      else:
        test_result.result_status = pb.RESULT_STATUS_UNKNOWN

      tags = exp.get("TAGS", [])
      if tags:
        test_result.metadata.fields["tags"].list_value.extend(tags)

      ramble_vars = exp.get("RAMBLE_VARIABLES", {})
      ramble_raw_vars = exp.get("RAMBLE_RAW_VARIABLES", {})

      log_file = ramble_vars.get("log_file")
      # TODO: Update log file link to point to the actual log file in bucket
      # Currently, it points to log files inside workspace. I'll update it
      # after the followup PR on uploading logs to bucket.
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
