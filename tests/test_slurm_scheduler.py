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

import logging
import unittest
from unittest.mock import patch
import os
import subprocess
from chs import exceptions as ex
from chs.constants import SLURM_CMD_TIMEOUT_SEC
from chs.schedulers import slurm
from chs.schedulers.slurm import SlurmScheduler


class TestSlurmScheduler(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        self.scheduler = SlurmScheduler()

    @patch("subprocess.run")
    def test_get_job_statuses_success(self, mock_run):
        # Mock sacct output (JobID|State)
        mock_output = "123|RUNNING\n124|COMPLETED\n"
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=mock_output
        )

        statuses = self.scheduler.get_job_statuses(["123", "124", "125"])

        self.assertEqual(statuses["123"], "RUNNING")
        self.assertEqual(statuses["124"], "COMPLETED")
        # 125 not in output -> UNKNOWN
        self.assertEqual(statuses["125"], "UNKNOWN")

    @patch("subprocess.run")
    def test_get_job_statuses_command_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError

        statuses = self.scheduler.get_job_statuses(["123"])

        # Should fallback to UNKNOWN
        self.assertEqual(statuses["123"], "UNKNOWN")

    @patch("os.path.exists")
    def test_fetch_job_log_exists(self, mock_exists):
        mock_exists.return_value = True
        # Should not raise
        self.scheduler.fetch_job_log("123", "/path/to/log")
        mock_exists.assert_called_with("/path/to/log")

    @patch("os.path.exists")
    def test_fetch_job_log_missing(self, mock_exists):
        mock_exists.return_value = False
        # Should log warning but not raise
        self.scheduler.fetch_job_log("123", "/path/to/log")

    def test_get_extra_env(self):
        """Test get_extra_env returns expected Enroot paths."""
        env = self.scheduler.get_extra_env()
        home = os.path.expanduser("~")
        self.assertEqual(
            env["ENROOT_RUNTIME_PATH"], os.path.join(home, ".enroot", "run")
        )
        self.assertEqual(
            env["ENROOT_CACHE_PATH"], os.path.join(home, ".enroot", "cache")
        )

    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_ensure_enroot_directories(self, mock_exists, mock_makedirs):
        mock_exists.return_value = False
        self.scheduler._ensure_enroot_directories()

        # Should not create credentials directory
        credentials_path = self.scheduler._enroot_paths[
            "ENROOT_CONFIG_CREDENTIALS"
        ]
        runtime_path = self.scheduler._enroot_paths["ENROOT_RUNTIME_PATH"]
        cache_path = self.scheduler._enroot_paths["ENROOT_CACHE_PATH"]

        mock_makedirs.assert_any_call(runtime_path, exist_ok=True)
        mock_makedirs.assert_any_call(cache_path, exist_ok=True)

        # Assert that credentials_path was NOT called with os.makedirs
        for call in mock_makedirs.call_args_list:
            self.assertNotEqual(call[0][0], credentials_path)

    @patch("google.auth.default")
    @patch("google.cloud.compute_v1.InstancesClient")
    def test_get_topology(self, mock_instances_client, mock_auth_default):
        """Test get_topology uses compute API."""
        mock_auth_default.return_value = (None, "my-project")

        mock_client_instance = mock_instances_client.return_value

        class MockInstance:
            def __init__(self, name, block, subblock):
                self.name = name

                class ResourceStatus:
                    def __init__(self, block, subblock):
                        class PhysicalHostTopology:
                            def __init__(self, block, subblock):
                                self.block = block
                                self.subblock = subblock

                        self.physical_host_topology = PhysicalHostTopology(
                            block, subblock
                        )

                self.resource_status = ResourceStatus(block, subblock)

        class MockInstancesInZone:
            def __init__(self, instances):
                self.instances = instances

        mock_instances = [
            MockInstance("n1", "s0_0", "s0_0_0"),
            MockInstance("n2", "s0_0", "s0_0_0"),
            MockInstance("n3", "s0_0", "s0_1"),
            MockInstance("n4", "s0_0", "s0_1"),
        ]

        mock_aggregated_list = [
            ("zones/us-central1-a", MockInstancesInZone(mock_instances))
        ]

        mock_client_instance.aggregated_list.return_value = mock_aggregated_list

        with patch.object(
            self.scheduler,
            "get_nodes_under_test",
            return_value=["n1", "n2", "n3", "n4"],
        ):
            topology = self.scheduler.get_topology()

        self.assertEqual(len(topology), 1)
        self.assertIn("s0_0", topology)
        self.assertIn("s0_0_0", topology["s0_0"])
        self.assertIn("s0_1", topology["s0_0"])
        self.assertEqual(topology["s0_0"]["s0_0_0"], ["n1", "n2"])
        self.assertEqual(topology["s0_0"]["s0_1"], ["n3", "n4"])

    def test_allow_interdomain_pairs(self):
        self.assertFalse(
            self.scheduler.allow_interdomain_pairs("a4x-highgpu-4g")
        )
        self.assertFalse(
            self.scheduler.allow_interdomain_pairs("a4x-maxgpu-4g-metal")
        )
        self.assertTrue(self.scheduler.allow_interdomain_pairs("a3-megagpu-8g"))
        self.assertTrue(self.scheduler.allow_interdomain_pairs())

    @patch("subprocess.run")
    def test_run_slurm_cmd_passes_timeout(self, mock_run):
        self.scheduler._run_slurm_cmd(["squeue"], check=True)
        mock_run.assert_called_once_with(
            ["squeue"],
            text=True,
            capture_output=True,
            check=True,
            timeout=SLURM_CMD_TIMEOUT_SEC,
        )

    @patch("subprocess.run")
    def test_run_slurm_cmd_timeout_override(self, mock_run):
        self.scheduler._run_slurm_cmd(["sacct"], timeout=5)
        self.assertEqual(mock_run.call_args.kwargs["timeout"], 5)

    @patch("subprocess.run")
    def test_resolve_raw_nodes_sinfo_timeout_raises(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired("sinfo", 60)
        with self.assertRaises(ex.ChsSchedulerException):
            self.scheduler.resolve_raw_nodes()

    @patch("subprocess.run")
    def test_get_job_statuses_timeout_returns_unknown(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired("sacct", 60)
        statuses = self.scheduler.get_job_statuses(["123", "124"])
        self.assertEqual(statuses, {"123": "UNKNOWN", "124": "UNKNOWN"})


def _completed(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _row(requested="", allocated="", run_id=None):
    """Formats a squeue line for `%n|%N|%k`."""
    comment = f"chs-run-id={run_id}" if run_id else "(null)"
    return f"{requested}|{allocated}|{comment}"


class FakeSlurm:
    """Fakes the Slurm CLI calls behind node selection, targeting and cancel.

    Hostlists expand through `hostlists`; a missing entry expands to the
    expression itself, like a single node name.
    """

    def __init__(
        self,
        rows=(),
        hostlists=None,
        cancel_listing=(),
        config="",
        scancel=None,
        idle=(),
    ):
        self.rows = list(rows)
        self.hostlists = hostlists or {}
        self.cancel_listing = list(cancel_listing)
        self.config = config
        self.scancel = scancel or _completed()
        self.idle = list(idle)
        self.calls = []

    def __call__(self, cmd, check=False, timeout=None):
        self.calls.append(cmd)
        if cmd[:3] == ["sinfo", "-e", "-t"]:
            return _completed("".join(f"{node}\n" for node in self.idle))
        if cmd[0] == "squeue":
            lines = (
                self.rows if "--format=%n|%N|%k" in cmd else self.cancel_listing
            )
            return _completed("".join(f"{line}\n" for line in lines))
        if cmd[:3] == ["scontrol", "show", "hostnames"]:
            nodes = self.hostlists.get(cmd[3], [cmd[3]])
            return _completed("\n".join(nodes) + "\n")
        if cmd[:3] == ["scontrol", "show", "config"]:
            return _completed(self.config)
        if cmd[0] == "scancel":
            return self.scancel
        raise AssertionError(f"Unexpected command: {cmd}")

    def commands(self, *prefix):
        return [cmd for cmd in self.calls if cmd[: len(prefix)] == list(prefix)]


class TestSlurmRunsTargetingNodes(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        self.scheduler = SlurmScheduler()

    def _patch_slurm(self, fake):
        patcher = patch.object(
            self.scheduler, "_run_slurm_cmd", side_effect=fake
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake

    def _enable_logging(self):
        logging.disable(logging.NOTSET)

    def test_squeue_queries_all_users_and_partitions_untruncated(self):
        fake = self._patch_slurm(FakeSlurm())
        self.scheduler.find_runs_targeting_nodes(["n1"])
        self.assertEqual(
            fake.commands("squeue"),
            [["squeue", "--all", "--noheader", "--format=%n|%N|%k"]],
        )

    def test_pending_job_targets_requested_nodes(self):
        self._patch_slurm(
            FakeSlurm(
                [_row("n[1-2]", run_id="other-cd34")],
                hostlists={"n[1-2]": ["n1", "n2"]},
            )
        )
        targeting = self.scheduler.find_runs_targeting_nodes(["n2", "n3"])
        self.assertEqual(targeting, {"other-cd34": ["n2"]})

    def test_running_job_targets_allocated_nodes(self):
        self._patch_slurm(
            FakeSlurm(
                [_row("", "n[3-4]", "other-cd34")],
                hostlists={"n[3-4]": ["n3", "n4"]},
            )
        )
        targeting = self.scheduler.find_runs_targeting_nodes(["n1", "n4", "n3"])
        self.assertEqual(targeting, {"other-cd34": ["n3", "n4"]})

    def test_ignores_jobs_without_run_id_comment(self):
        self._patch_slurm(
            FakeSlurm([_row("n1", "n1"), "n1|n1|unrelated comment"])
        )
        self.assertEqual(self.scheduler.find_runs_targeting_nodes(["n1"]), {})

    def test_jobs_are_grouped_by_run_id_comment(self):
        self._patch_slurm(
            FakeSlurm(
                [
                    _row("n1", "n1", "my-test-x7k2"),
                    _row("n2", run_id="my-test-x7k2"),
                    _row("n3", run_id="my-test"),
                ]
            )
        )
        targeting = self.scheduler.find_runs_targeting_nodes(["n1", "n2", "n3"])
        self.assertEqual(
            targeting, {"my-test-x7k2": ["n1", "n2"], "my-test": ["n3"]}
        )

    def test_free_text_comment_is_parsed(self):
        self._patch_slurm(
            FakeSlurm(["n1|n1|weird|comment", "n2||chs-run-id=x-ab12"])
        )
        self.assertEqual(
            self.scheduler.find_runs_targeting_nodes(["n1", "n2"]),
            {"x-ab12": ["n2"]},
        )

    def test_each_distinct_hostlist_is_expanded_once(self):
        fake = self._patch_slurm(
            FakeSlurm(
                [
                    _row("n[1-2]", "n[1-2]", "a-aaaa"),
                    _row("n[1-2]", run_id="b-bbbb"),
                    _row("n3", run_id="c-cccc"),
                ],
                hostlists={"n[1-2]": ["n1", "n2"]},
            )
        )
        self.scheduler.find_runs_targeting_nodes(["n1"])
        self.assertCountEqual(
            [cmd[3] for cmd in fake.commands("scontrol", "show", "hostnames")],
            ["n[1-2]", "n3"],
        )

    def test_no_requested_nodes_skips_squeue(self):
        fake = self._patch_slurm(FakeSlurm())
        self.assertEqual(self.scheduler.find_runs_targeting_nodes([]), {})
        self.assertEqual(fake.calls, [])

    def test_squeue_failure_raises(self):
        for error in (
            subprocess.CalledProcessError(1, "squeue", stderr="down"),
            subprocess.TimeoutExpired("squeue", 60),
            FileNotFoundError("squeue"),
        ):
            with self.subTest(error=type(error).__name__):
                with patch.object(
                    self.scheduler, "_run_slurm_cmd", side_effect=error
                ):
                    with self.assertRaisesRegex(
                        ex.ChsSchedulerException, "cannot verify"
                    ):
                        self.scheduler.find_runs_targeting_nodes(["n1"])

    def test_squeue_line_without_comment_is_ignored(self):
        self._patch_slurm(FakeSlurm(["garbage"]))
        self.assertEqual(self.scheduler.find_runs_targeting_nodes(["n1"]), {})

    def test_empty_hostlists_are_not_expanded(self):
        fake = self._patch_slurm(
            FakeSlurm(
                [
                    _row("(null)", "(null)", "pending-ab12"),
                    _row("n/a", " ", "pending-cd34"),
                    _row("", "", "pending-ef56"),
                    _row("N/A", "None", "pending-gh78"),
                ]
            )
        )
        self.assertEqual(self.scheduler.find_runs_targeting_nodes(["n1"]), {})
        self.assertEqual(fake.commands("scontrol", "show", "hostnames"), [])

    def test_private_jobs_warns(self):
        self._enable_logging()
        self._patch_slurm(
            FakeSlurm(config="PrivateData             = jobs,usage\n")
        )
        with self.assertLogs(slurm.logger, level="WARNING") as logs:
            self.scheduler.find_runs_targeting_nodes(["n1"])
        self.assertEqual(len(logs.records), 1)
        self.assertIn("PrivateData=jobs", logs.output[0])

    def test_public_jobs_do_not_warn(self):
        self._enable_logging()
        self._patch_slurm(FakeSlurm(config="PrivateData = usage,users\n"))
        with self.assertNoLogs(slurm.logger, level="WARNING"):
            self.scheduler.find_runs_targeting_nodes(["n1"])

    def test_private_data_lookup_failure_does_not_fail(self):
        fake = FakeSlurm()

        def run(cmd, check=False, timeout=None):
            if cmd[:3] == ["scontrol", "show", "config"]:
                raise subprocess.TimeoutExpired(cmd, 60)
            return fake(cmd, check, timeout)

        with patch.object(self.scheduler, "_run_slurm_cmd", side_effect=run):
            self.assertEqual(
                self.scheduler.find_runs_targeting_nodes(["n1"]), {}
            )

    def test_cancel_run_selects_jobs_by_exact_run_id(self):
        fake = self._patch_slurm(
            FakeSlurm(
                cancel_listing=[
                    "10|chs-run-id=run-ab12",
                    "11|chs-run-id=run-ab12",
                    "12|chs-run-id=run-ab12-cd34",
                    "13|chs-run-id=run",
                    "14|(null)",
                ]
            )
        )
        self.assertTrue(self.scheduler.cancel_run("run-ab12"))
        self.assertEqual(
            fake.commands("squeue"),
            [["squeue", "--all", "--noheader", "--format=%A|%k"]],
        )
        self.assertEqual(fake.commands("scancel"), [["scancel", "10,11"]])

    def test_cancel_all_runs_covers_chs_jobs_of_all_users(self):
        fake = self._patch_slurm(
            FakeSlurm(
                cancel_listing=[
                    "10|chs-run-id=run-a",
                    " 11 | chs-run-id=run-b ",
                    "12|(null)",
                    "13|train",
                ]
            )
        )
        self.scheduler.cancel_all_runs()
        self.assertEqual(
            fake.commands("squeue"),
            [["squeue", "--all", "--noheader", "--format=%A|%k"]],
        )
        self.assertEqual(fake.commands("scancel"), [["scancel", "10,11"]])

    def test_cancel_all_runs_selects_jobs_by_comment_not_name(self):
        def run(cmd, check=False, timeout=None):
            listing = {
                "--format=%A|%j": ["10|chs-user-job", "11|train"],
                "--format=%A|%k": ["10|(null)", "11|chs-run-id=run-a"],
            }
            if cmd[0] == "squeue":
                return _completed(
                    "".join(f"{line}\n" for line in listing[cmd[-1]])
                )
            return fake(cmd, check, timeout)

        fake = FakeSlurm()
        with patch.object(self.scheduler, "_run_slurm_cmd", side_effect=run):
            self.scheduler.cancel_all_runs()
        self.assertEqual(fake.commands("scancel"), [["scancel", "11"]])

    def test_cancel_all_runs_warns_if_jobs_are_private(self):
        self._enable_logging()
        self._patch_slurm(
            FakeSlurm(
                cancel_listing=["10|chs-run-id=run-a"],
                config="PrivateData = jobs\n",
            )
        )
        with self.assertLogs(slurm.logger, level="WARNING") as logs:
            self.scheduler.cancel_all_runs()
        self.assertIn("only your own", logs.output[0])

    def test_cancel_run_without_matches_does_not_scancel(self):
        fake = self._patch_slurm(FakeSlurm(cancel_listing=["10|(null)"]))
        self.assertFalse(self.scheduler.cancel_run("run-ab12"))
        self.assertEqual(fake.commands("scancel"), [])

    def test_cancel_run_ignores_jobs_that_already_finished(self):
        self._patch_slurm(
            FakeSlurm(
                cancel_listing=["10|chs-run-id=run-ab12"],
                scancel=_completed(
                    returncode=1,
                    stderr="scancel: error: Kill job error on job id 10: "
                    "Job/step already completing or completed",
                ),
            )
        )
        self.scheduler.cancel_run("run-ab12")

    def test_cancel_run_permission_denied_raises(self):
        self._patch_slurm(
            FakeSlurm(
                cancel_listing=["10|chs-run-id=run-ab12"],
                scancel=_completed(
                    returncode=1,
                    stderr="scancel: error: Kill job error on job id 10: "
                    "Access/permission denied",
                ),
            )
        )
        with self.assertRaisesRegex(
            ex.ChsCancellationException, "other users.*permission denied"
        ):
            self.scheduler.cancel_run("run-ab12")

    def test_cancel_listing_failure_raises(self):
        with patch.object(
            self.scheduler,
            "_run_slurm_cmd",
            side_effect=subprocess.CalledProcessError(1, "squeue"),
        ):
            for cancel in (
                lambda: self.scheduler.cancel_run("run-ab12"),
                self.scheduler.cancel_all_runs,
            ):
                with self.assertRaises(ex.ChsCancellationException):
                    cancel()


class TestSlurmNodesUnderTest(unittest.TestCase):
    def setUp(self):
        self.scheduler = SlurmScheduler(nodes="n1,n2,n3,n4")

    def _patch_slurm(self, fake):
        self.enterContext(
            patch.object(self.scheduler, "_run_slurm_cmd", side_effect=fake)
        )

    def test_skips_busy_nodes_and_nodes_targeted_by_other_runs(self):
        self._patch_slurm(
            FakeSlurm(
                [_row("n2", run_id="other-cd34")], idle=["n1", "n2", "n3"]
            )
        )
        with self.assertLogs(slurm.logger, level="WARNING") as logs:
            nodes = self.scheduler.get_nodes_under_test()
        self.assertEqual(nodes, ["n1", "n3"])
        self.assertEqual(
            logs.output,
            [
                "WARNING:chs.schedulers.slurm:Skipping 1 node(s) targeted by "
                "CHS run 'other-cd34'."
            ],
        )

    def test_warning_counts_skipped_nodes_per_run(self):
        nodes = [f"n{i}" for i in range(1, 9)]
        self.scheduler = SlurmScheduler(nodes=",".join(nodes))
        self._patch_slurm(
            FakeSlurm(
                [_row("n[2-8]", run_id="big-ab12"), _row("n1", "n1", "ef56")],
                hostlists={"n[2-8]": nodes[1:]},
                idle=nodes,
            )
        )
        with self.assertLogs(slurm.logger, level="WARNING") as logs:
            self.assertEqual(self.scheduler.get_nodes_under_test(), [])
        self.assertEqual(
            [record.getMessage() for record in logs.records],
            [
                "Skipping 7 node(s) targeted by CHS run 'big-ab12'.",
                "Skipping 1 node(s) targeted by CHS run 'ef56'.",
            ],
        )

    def test_untargeted_idle_nodes_are_kept_without_warning(self):
        self._patch_slurm(FakeSlurm(idle=["n1", "n2", "n3", "n4"]))
        with self.assertNoLogs(slurm.logger, level="WARNING"):
            nodes = self.scheduler.get_nodes_under_test()
        self.assertEqual(nodes, ["n1", "n2", "n3", "n4"])


if __name__ == "__main__":
    unittest.main()
