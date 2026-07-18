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

"""HTML Bill of Health Report Generator for Cluster Health Scanner (CHS)."""

import json
import os
from typing import Any

from chs.core.summary_aggregator import aggregate_boh_summary
from chs.protos import bill_of_health_pb2 as pb

DEFAULT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "templates", "boh_template.html"
)


class HtmlReportGenerator:
    """Generates standalone, interactive HTML Bill of Health reports."""

    def __init__(self, template_path: str | None = None):
        self.template_path = template_path or DEFAULT_TEMPLATE_PATH

    def generate_report_data(
        self,
        boh: pb.BillOfHealth,
        run_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Converts BillOfHealth protobuf into pure-Python JSON-serializable
        dictionary."""
        run_info = run_info or {}
        summary = aggregate_boh_summary(boh)

        all_nodes = set()
        for tr in boh.test_results:
            for t in tr.target_resources:
                all_nodes.add(t.node_name)

        degraded_nodes = set()
        missing_metrics_nodes = set()
        for ts in summary.test_summaries:
            if (
                not ts.group_has_no_thresholds
                and not ts.explicit_skip_validation
            ):
                degraded_nodes.update(ts.below_threshold_resources)
                missing_metrics_nodes.update(ts.missing_metrics_resources)
                missing_metrics_nodes.update(ts.failed_to_execute_resources)

        missing_metrics_only = missing_metrics_nodes - degraded_nodes

        total_nodes = len(all_nodes)
        degraded_count = len(degraded_nodes)
        missing_metrics_count = len(missing_metrics_only)
        healthy_count = total_nodes - len(
            degraded_nodes | missing_metrics_nodes
        )
        failed_count = sum(
            1 for e in summary.errors_and_warnings if e.type == "ERROR"
        )

        report_id = (
            summary.run_id
            or (
                boh.metadata.fields["run_id"].string_value
                if "run_id" in boh.metadata.fields
                else ""
            )
            or boh.report_id
            or run_info.get("report_id", "N/A")
        )

        created_at_str = run_info.get("created_at")
        if not created_at_str and boh.HasField("created_at"):
            created_at_str = boh.created_at.ToDatetime().strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
        created_at_str = created_at_str or "N/A"

        system_type = (
            boh.metadata.fields["system_type"].string_value
            if "system_type" in boh.metadata.fields
            else run_info.get("system_type", "KUBERNETES_CLUSTER")
        )

        skip_validation = (
            boh.metadata.fields["skip_validation"].bool_value
            if "skip_validation" in boh.metadata.fields
            else False
        )

        return {
            "metadata": {
                "report_id": report_id,
                "chs_version": run_info.get("chs_version", "1.0.0"),
                "created_at": created_at_str,
                "system_type": system_type,
                "command_statement": run_info.get("command_statement", ""),
                "skip_validation": skip_validation,
            },
            "summary": {
                "total_nodes": total_nodes,
                "healthy_count": healthy_count,
                "degraded_count": degraded_count,
                "missing_metrics_count": missing_metrics_count,
                "failed_count": failed_count,
                "total_test_suites": len(summary.test_summaries),
                "potential_errors_count": len(summary.errors_and_warnings),
            },
        }

    def render_html(
        self,
        boh: pb.BillOfHealth,
        run_info: dict[str, Any] | None = None,
    ) -> str:
        """Renders HTML string by populating the template with serialized JSON
        data."""
        data_dict = self.generate_report_data(boh, run_info)
        raw_json = json.dumps(data_dict, indent=2)

        with open(self.template_path, "r", encoding="utf-8") as f:
            template_content = f.read()

        return template_content.replace(
            "__BOH_OVERVIEW_RAW_DATA_JSON__", raw_json
        ).replace("__BOH_RAW_DATA_JSON__", raw_json)

    def save_html(
        self,
        boh: pb.BillOfHealth,
        output_path: str,
        run_info: dict[str, Any] | None = None,
    ) -> str:
        """Generates and saves the HTML report to specified output path."""
        html_str = self.render_html(boh, run_info)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_str)
        return output_path
