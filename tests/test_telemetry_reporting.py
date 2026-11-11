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

"""Unit tests for chs.telemetry.reporting."""

import datetime
import json
import sys
import unittest
from unittest import mock

import click
from click.testing import CliRunner

from chs import telemetry
from chs.core import summary_aggregator
from chs.protos import bill_of_health_pb2 as pb
from chs.schedulers.base import BaseScheduler, TpuSliceSummary
from chs.schedulers.slurm import SlurmScheduler
from chs.telemetry import reporting as tr
from chs.telemetry.supported_metrics import MetricName


def _test_summary(
    name: str, level: str = "", **counts
) -> summary_aggregator.TestGroupSummary:
    fields = {
        "met_threshold": 0,
        "unmet_threshold": 0,
        "failed_to_execute": 0,
        "denominator": 0,
        "entity_noun": "nodes",
        "missing_metrics": 0,
        **counts,
    }
    return summary_aggregator.TestGroupSummary(
        canonical_name=name,
        test_level=level,
        group_has_no_thresholds=False,
        explicit_skip_validation=False,
        unmet_threshold_resources=["secret-node"],
        missing_metrics_resources=[],
        display_name=f"{name} ({level})" if level else name,
        **fields,
    )


def _issue(
    test: str, kind: str, category: str, node: str
) -> summary_aggregator.HardErrorEvent:
    return summary_aggregator.HardErrorEvent(
        type=kind,
        category=category,
        message=f"secret log line on {node}",
        node=node,
        test=test,
    )


def _summary(tests=(), issues=()) -> summary_aggregator.RunSummaryData:
    return summary_aggregator.RunSummaryData(
        run_id="secret-run",
        submission_time="",
        total_nodes=4,
        test_summaries=list(tests),
        errors_and_warnings=list(issues),
    )


def _pairwise_result(status: int, *nodes: str, issues=()) -> pb.TestResult:
    result = pb.TestResult(
        test_name="nccl-pairwise_all-reduce_acme",
        test_level="all-reduce",
        result_status=status,
    )
    for node in nodes:
        result.target_resources.add(node_name=node)
    if status == pb.RESULT_STATUS_FAILED:
        metric = result.metrics.add(key="bandwidth")
        metric.validation_criteria.result_status = pb.RESULT_STATUS_FAILED
    for key in issues:
        result.metrics.add(key=key).value.string_value = "secret log line"
    return result


def _scheduler(
    name: str = "slurm",
    project_id: str | None = "customer-project",
    node_to_zone: dict[str, str] | None = None,
) -> mock.Mock:
    scheduler = mock.Mock(spec=BaseScheduler)
    scheduler.name = name
    scheduler.project_id = project_id
    scheduler.node_to_zone = node_to_zone or {}
    scheduler.get_tpu_slice_summary.return_value = None
    return scheduler


class _ReportingTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.send = self.enterContext(
            mock.patch.object(tr.telemetry, "send_metric", autospec=True)
        )
        self.enterContext(
            mock.patch.object(telemetry, "_state", telemetry._State())
        )

    def sent(self) -> dict:
        """The last value sent for each metric."""
        return {c.args[0]: c.args[1] for c in self.send.call_args_list}


class RecordTestResultsTest(_ReportingTestCase):
    def _record(
        self, summary: summary_aggregator.RunSummaryData | None
    ) -> dict:
        with mock.patch.object(
            tr, "aggregate_boh_summary", return_value=summary
        ) as aggregate:
            boh = pb.BillOfHealth(report_id="r")
            tr.record_test_results(boh)
        aggregate.assert_called_once_with(boh)
        return self.sent()

    def test_reports_each_test_as_the_results_summary_counts_it(self):
        sent = self._record(
            _summary(
                tests=[
                    _test_summary(
                        "nccl-pairwise",
                        "all-reduce",
                        met_threshold=5,
                        unmet_threshold=1,
                        denominator=6,
                        entity_noun="pairs",
                    ),
                    _test_summary(
                        "dcgm",
                        met_threshold=3,
                        failed_to_execute=1,
                        denominator=4,
                        timeout_count=1,
                    ),
                ]
            )
        )

        self.assertEqual(
            sent[MetricName.TEST_RESULTS],
            [
                {
                    "test": "nccl-pairwise",
                    "level": "all-reduce",
                    "unit": "pairs",
                    "total": 6,
                    "met_threshold": 5,
                    "unmet_threshold": 1,
                    "missing_metrics": 0,
                    "failed_to_execute": 0,
                    "timeout": 0,
                    "deadline": 0,
                    "cancelled": 0,
                },
                {
                    "test": "dcgm",
                    "level": "default",
                    "unit": "nodes",
                    "total": 4,
                    "met_threshold": 3,
                    "unmet_threshold": 0,
                    "missing_metrics": 0,
                    "failed_to_execute": 1,
                    "timeout": 1,
                    "deadline": 0,
                    "cancelled": 0,
                },
            ],
        )

    def test_detected_errors_count_distinct_nodes_without_messages(self):
        nccl = "nccl-cluster (all-reduce)"
        sent = self._record(
            _summary(
                tests=[_test_summary("nccl-cluster", "all-reduce")],
                issues=[
                    _issue(nccl, "FAIL", "CUDA", "node-1"),
                    _issue(nccl, "FAIL", "CUDA", "node-2"),
                    _issue(nccl, "FAIL", "CUDA", "node-1"),
                    _issue(nccl, "WARNING", "Topology", "node-1"),
                ],
            )
        )

        self.assertEqual(
            sent[MetricName.DETECTED_ERRORS],
            [
                {
                    "test": "nccl-cluster",
                    "level": "all-reduce",
                    "severity": "FAIL",
                    "category": "CUDA",
                    "nodes": 2,
                },
                {
                    "test": "nccl-cluster",
                    "level": "all-reduce",
                    "severity": "WARNING",
                    "category": "Topology",
                    "nodes": 1,
                },
            ],
        )
        self.assertNotIn("secret", json.dumps(sent))
        self.assertNotIn("node-", json.dumps(sent))

    def test_missing_summary_sends_nothing(self):
        self._record(None)

        self.send.assert_not_called()

    def test_never_raises(self):
        with mock.patch.object(
            tr, "aggregate_boh_summary", side_effect=TypeError("bug")
        ):
            tr.record_test_results(pb.BillOfHealth())  # Doesn't raise.

        self.send.assert_not_called()

    def test_pairwise_counts_pairs_with_the_real_summary(self):
        self.enterContext(
            mock.patch("chs.core.summary_aggregator.ChsAppData")
        ).return_value.get_run.return_value = None
        boh = pb.BillOfHealth(report_id="r")
        boh.test_results.extend(
            [
                _pairwise_result(pb.RESULT_STATUS_SUCCESS, "n0", "n1"),
                _pairwise_result(pb.RESULT_STATUS_SUCCESS, "n2", "n3"),
                _pairwise_result(
                    pb.RESULT_STATUS_FAILED,
                    "n0",
                    "n2",
                    issues=["NVML Error Detected"],
                ),
            ]
        )

        tr.record_test_results(boh)

        sent = self.sent()
        (test,) = sent[MetricName.TEST_RESULTS]
        self.assertEqual(test["test"], "nccl-pairwise")
        self.assertEqual(test["level"], "all-reduce")
        self.assertEqual(test["unit"], "pairs")
        self.assertEqual(test["total"], 3)
        self.assertEqual(test["met_threshold"], 2)
        self.assertEqual(test["unmet_threshold"], 1)
        self.assertEqual(
            sent[MetricName.DETECTED_ERRORS],
            [
                {
                    "test": "nccl-pairwise",
                    "level": "all-reduce",
                    "severity": "FAIL",
                    "category": "NVML",
                    "nodes": 2,
                }
            ],
        )
        self.assertNotIn("secret", json.dumps(sent))

    def test_bill_of_health_without_report_id_sends_nothing(self):
        tr.record_test_results(pb.BillOfHealth())

        self.send.assert_not_called()


class RecordRunTest(_ReportingTestCase):
    def _record_run(self, scheduler, **overrides) -> dict:
        args = {
            "storage_config": {"type": "gcs", "gcs": {"bucket_name": "b"}},
            "machine_type": "a3-megagpu-8g",
            "nodes": ["n0", "n1", "n2"],
            "suite": "lite",
            "test_names": ["nvidia-smi", "dcgm"],
            **overrides,
        }
        tr.record_run(scheduler, **args)
        return self.sent()

    def test_reports_what_the_run_targets(self):
        scheduler = SlurmScheduler()
        scheduler.project_id = "customer-project"
        scheduler.node_to_zone = {
            "n0": "us-central1-b",
            "n1": "us-central1-a",
            "n2": "us-central1-b",
            "idle": "europe-west4-a",
        }

        sent = self._record_run(scheduler)

        self.assertEqual(
            sent,
            {
                MetricName.CLUSTER_TYPE: "SLURM",
                MetricName.STORAGE_TYPE: "GCS",
                MetricName.MACHINE_TYPE: "a3-megagpu-8g",
                MetricName.NODE_COUNT: 3,
                # Only zones of nodes under test, deduplicated.
                MetricName.ZONE: "us-central1-a,us-central1-b",
                MetricName.IS_TEST_DATA: "false",
                MetricName.TEST_SUITE: "lite",
                MetricName.TEST_NAME: "dcgm,nvidia-smi",
            },
        )
        self.assertEqual(telemetry._state.project_id, "customer-project")

    def test_cluster_type_comes_from_the_scheduler(self):
        for name, expected in (
            ("slurm", "SLURM"),
            ("kubernetes", "GKE"),
            ("standalone", "STANDALONE"),
            ("other", "UNKNOWN"),
        ):
            with self.subTest(name=name):
                sent = self._record_run(_scheduler(name=name))

                self.assertEqual(sent[MetricName.CLUSTER_TYPE], expected)

    def test_is_test_data(self):
        for project_id, expected in (
            ("hpc-toolkit-dev", "true"),
            ("gke-aishared-gsc-dev", "true"),
            ("customer-project", "false"),
            (None, "false"),
        ):
            with self.subTest(project_id=project_id):
                sent = self._record_run(_scheduler(project_id=project_id))

                self.assertEqual(sent[MetricName.IS_TEST_DATA], expected)

    def test_unknown_values_are_left_out(self):
        sent = self._record_run(_scheduler(), storage_config={}, suite=None)

        self.assertEqual(sent[MetricName.STORAGE_TYPE], "NONE")
        self.assertNotIn(MetricName.ZONE, sent)
        self.assertNotIn(MetricName.TEST_SUITE, sent)

    def test_local_only_run_reports_all_metrics(self):
        sent = self._record_run(_scheduler(), storage_config=None)

        self.assertEqual(sent[MetricName.STORAGE_TYPE], "NONE")
        self.assertEqual(sent[MetricName.MACHINE_TYPE], "a3-megagpu-8g")
        self.assertEqual(sent[MetricName.TEST_NAME], "dcgm,nvidia-smi")

    def test_reports_tpu_slices(self):
        scheduler = _scheduler(name="kubernetes")
        scheduler.get_tpu_slice_summary.return_value = TpuSliceSummary(
            slice_type="dynamic",
            topologies={"2x2x1": 1, "4x4x4": 2},
            nodepool_count=3,
        )

        sent = self._record_run(scheduler)

        scheduler.get_tpu_slice_summary.assert_called_once_with(
            ["n0", "n1", "n2"]
        )
        self.assertEqual(sent[MetricName.TPU_SLICE_TYPE], "dynamic")
        self.assertEqual(
            sent[MetricName.TPU_SLICE_TOPOLOGIES], {"2x2x1": 1, "4x4x4": 2}
        )
        self.assertEqual(sent[MetricName.TPU_NODEPOOL_COUNT], 3)

    def test_unknown_tpu_slice_details_are_left_out(self):
        scheduler = _scheduler(name="kubernetes")
        scheduler.get_tpu_slice_summary.return_value = TpuSliceSummary(
            slice_type="static", topologies={}, nodepool_count=None
        )

        sent = self._record_run(scheduler)

        self.assertEqual(sent[MetricName.TPU_SLICE_TYPE], "static")
        self.assertNotIn(MetricName.TPU_SLICE_TOPOLOGIES, sent)
        self.assertNotIn(MetricName.TPU_NODEPOOL_COUNT, sent)

    def test_never_raises(self):
        tr.record_run(mock.Mock(spec=[]), {}, "m", [], None, [])


class RecordTestDurationTest(_ReportingTestCase):
    def test_record_test_duration(self):
        now = datetime.datetime(
            2026, 9, 27, 12, 0, 0, tzinfo=datetime.timezone.utc
        )

        tr.record_test_duration("2026-09-27T09:05:00+00:00", now=now)

        self.assertEqual(self.sent(), {MetricName.TEST_DURATION_MS: 10_500_000})

    def test_record_test_duration_treats_naive_times_as_utc(self):
        now = datetime.datetime(
            2026, 9, 27, 12, 0, 0, tzinfo=datetime.timezone.utc
        )

        tr.record_test_duration("2026-09-27T09:05:00", now=now)

        self.assertEqual(self.sent(), {MetricName.TEST_DURATION_MS: 10_500_000})

    def test_record_test_duration_skips_bad_values(self):
        for submission_time in (None, "", "yesterday"):
            with self.subTest(submission_time=submission_time):
                tr.record_test_duration(submission_time)

                self.send.assert_not_called()


def _provider(uri: str) -> mock.Mock:
    provider = mock.Mock()
    provider.get_uri.return_value = uri
    return provider


class RecordUploadsTest(_ReportingTestCase):
    def test_reports_upload_status_and_gcs_sizes(self):
        tr.record_uploads(
            _provider("gs://bucket/"),
            {"workspace": (False, 5000), "bill_of_health": (True, 300)},
        )

        self.assertEqual(
            self.sent(),
            {
                MetricName.UPLOAD_STATUS: {
                    "bill_of_health": "success",
                    "workspace": "failed",
                },
                MetricName.ARTIFACT_SIZE_BYTES: {"bill_of_health": 300},
            },
        )

    def test_sizes_are_only_reported_for_gcs(self):
        tr.record_uploads(_provider("/data/chs/"), {"workspace": (True, 5000)})

        self.assertEqual(
            self.sent(), {MetricName.UPLOAD_STATUS: {"workspace": "success"}}
        )

    def test_never_raises(self):
        tr.record_uploads(mock.Mock(spec=[]), {"workspace": (True, 5000)})

        self.assertEqual(
            self.sent(), {MetricName.UPLOAD_STATUS: {"workspace": "success"}}
        )


class IsCommandSupportedTest(unittest.TestCase):
    def test_supported_commands(self):
        for args in (
            ["run", "--suite", "lite", "-y"],
            ["-v", "results"],
            ["list", "tests"],
            ["cancel", "-y"],
            ["history"],
        ):
            with self.subTest(args=args):
                self.assertTrue(tr.is_command_supported(args))

    def test_other_commands_are_not_supported(self):
        for args in (
            [],
            ["configure", "storage", "show"],
            ["configure", "telemetry", "disable"],
            ["-v", "configure", "telemetry", "status"],
            ["unknown-command"],
        ):
            with self.subTest(args=args):
                self.assertFalse(tr.is_command_supported(args))

    def test_help_and_version_are_not_supported(self):
        for args in (
            ["--help"],
            ["-h"],
            ["--version"],
            ["run", "--help"],
            ["results", "-h"],
        ):
            with self.subTest(args=args):
                self.assertFalse(tr.is_command_supported(args))


class TelemetryGroupTest(_ReportingTestCase):
    """Tests for the click classes, on a small stand-in CLI."""

    def setUp(self):
        super().setUp()

        @click.group(cls=tr.TelemetryGroup)
        @click.option("-v", "--verbose", is_flag=True)
        def root(verbose):
            del verbose

        @root.command()
        @click.option("--suite")
        @click.option("--nodepools")
        @click.option("--auto-approve", "-y", is_flag=True)
        @click.option("--run-id")
        def run(suite, nodepools, auto_approve, run_id):
            del suite, nodepools, auto_approve, run_id

        @root.group()
        def configure():
            pass

        @configure.command()
        @click.argument("bucket")
        @click.option("--force", is_flag=True)
        def gcs(bucket, force):
            del bucket, force

        @root.command()
        def fail():
            raise click.ClickException("nope")

        @root.command(name="quit")
        def quit_():
            raise click.exceptions.Exit(0)

        @root.command()
        def interrupted():
            # What run, cancel and results turn Ctrl+C into.
            raise click.exceptions.Exit(130)

        @root.command(name="ctrl-c")
        def ctrl_c():
            raise KeyboardInterrupt

        @root.command()
        def crash():
            raise RuntimeError("bug")

        @root.command()
        @click.argument("code", required=False)
        def stop(code):
            sys.exit(code)

        self.root = root
        self.runner = CliRunner()

    def test_reports_sorted_flag_names_without_values(self):
        result = self.runner.invoke(
            self.root,
            ["-v", "run", "-y", "--nodepools=secret-pool", "--suite", "lite"],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            self.sent()[MetricName.COMMAND_FLAGS],
            "auto-approve,nodepools,suite,verbose",
        )

    def test_nested_commands_use_telemetry_classes(self):
        result = self.runner.invoke(
            self.root, ["configure", "gcs", "secret-bucket", "--force"]
        )

        self.assertEqual(result.exit_code, 0, result.output)
        # Positional arguments are values, so they're never reported.
        self.assertEqual(self.sent()[MetricName.COMMAND_FLAGS], "force")

    def test_no_flags_sends_nothing(self):
        self.runner.invoke(self.root, ["run"])

        self.assertNotIn(MetricName.COMMAND_FLAGS, self.sent())

    def test_reports_the_top_level_command(self):
        for args, expected in (
            (["run"], "run"),
            (["-v", "configure", "gcs", "secret-bucket"], "configure"),
        ):
            with self.subTest(args=args):
                self.runner.invoke(self.root, args)

                self.assertEqual(
                    self.sent()[MetricName.PARENT_COMMAND], expected
                )

    def test_unknown_command_is_not_reported(self):
        result = self.runner.invoke(self.root, ["secret-typo"])

        self.assertEqual(result.exit_code, 2)
        self.assertNotIn(MetricName.PARENT_COMMAND, self.sent())

    def test_reports_error_category(self):
        result = self.runner.invoke(self.root, ["fail"])

        self.assertEqual(result.exit_code, 1)
        self.assertEqual(
            self.sent()[MetricName.ERROR_CATEGORY], "ClickException"
        )

    def test_clean_exit_is_not_an_error(self):
        result = self.runner.invoke(self.root, ["quit"])

        self.assertEqual(result.exit_code, 0)
        self.assertNotIn(MetricName.ERROR_CATEGORY, self.sent())

    def test_ctrl_c_is_reported_the_same_way_by_every_command(self):
        for command, exit_code in (("interrupted", 130), ("ctrl-c", 1)):
            with self.subTest(command=command):
                result = self.runner.invoke(self.root, [command])

                self.assertEqual(result.exit_code, exit_code)
                self.assertEqual(
                    self.sent()[MetricName.ERROR_CATEGORY], "KeyboardInterrupt"
                )
                self.send.assert_called_with(MetricName.EXIT_CODE, exit_code)

    def test_reports_exit_code(self):
        for args, expected in (
            (["run"], 0),
            (["stop"], 0),
            (["fail"], 1),
            (["crash"], 1),
            (["stop", "boom"], 1),
            (["nope"], 2),
        ):
            with self.subTest(args=args):
                result = self.runner.invoke(self.root, args)

                self.assertEqual(result.exit_code, expected)
                self.send.assert_called_with(MetricName.EXIT_CODE, expected)


if __name__ == "__main__":
    unittest.main()
