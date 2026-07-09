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

"""Unit tests for RambleManager."""

import logging
import os
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from unittest import mock
from chs.exceptions import ChsArchiveException
from chs.ramble_manager import RambleManager
from chs.exceptions import ChsException


class TestRambleManager(unittest.TestCase):
    """Tests for RambleManager."""

    def setUp(self):
        super().setUp()
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = self.tmp_dir_obj.name
        self.addCleanup(self.tmp_dir_obj.cleanup)
        self.mock_run = self.enterContext(mock.patch("subprocess.run"))
        self.mock_run.return_value.stdout = ""
        self.mock_run.return_value.stderr = ""
        self.mock_run.return_value.returncode = 0
        self.mock_sysconfig = self.enterContext(
            mock.patch.object(sysconfig, "get_path", autospec=True)
        )
        self.mock_app_data_cls = self.enterContext(
            mock.patch("chs.ramble_manager.ChsAppData", autospec=True)
        )
        self.mock_app_data = self.mock_app_data_cls.return_value

        self.mock_sysconfig.return_value = self.tmp_dir
        self.real_os_path_exists = os.path.exists
        self.mock_os_path_exists = self.enterContext(
            mock.patch("os.path.exists", autospec=True)
        )
        self.mock_os_path_exists.side_effect = self.real_os_path_exists

        self.mock_os_path_abspath = self.enterContext(
            mock.patch("os.path.abspath", autospec=True)
        )
        self.mock_os_path_abspath.side_effect = lambda x: x
        setup_script_dir = os.path.join(self.tmp_dir, "share/ramble")
        os.makedirs(setup_script_dir, exist_ok=True)
        self.setup_script_path = os.path.join(setup_script_dir, "setup-env.sh")
        with open(self.setup_script_path, "w") as f:
            f.write("#!/bin/bash\n")

        logging.disable(logging.CRITICAL)

    def tearDown(self):
        super().tearDown()
        logging.disable(logging.NOTSET)

    def _get_expected_setup_script_path(self):
        """Helper to get the expected path based on mock setup."""
        return os.path.join(
            self.mock_sysconfig.return_value, "share/ramble/setup-env.sh"
        )

    def test_init_success(self):
        """Test RambleManager initializes successfully when ramble is found."""
        manager = RambleManager(
            base_dir="/tmp", workspace_name="ws1", run_command=self.mock_run
        )
        self.assertEqual(manager.path, "/tmp/ws1")
        self.mock_sysconfig.assert_called_once_with("purelib")

    def test_execute_shell_command_success(self):
        """Test _execute_shell_command runs command successfully."""
        manager = RambleManager("/tmp", "ws1", self.mock_run)
        self.mock_run.return_value = subprocess.CompletedProcess(
            args=["echo", "hi"], returncode=0
        )

        manager._execute_shell_command(["echo", "hi"], capture_output=True)

        self.mock_run.assert_called_once_with(
            ["echo", "hi"],
            check=False,
            text=True,
            cwd=None,
            capture_output=True,
            env=None,
        )

    def test_execute_shell_command_with_env(self):
        """Test _execute_shell_command passes env vars."""
        manager = RambleManager("/tmp", "ws1", self.mock_run)
        self.mock_run.return_value = subprocess.CompletedProcess(
            args=["echo", "hi"], returncode=0
        )

        manager._execute_shell_command(["echo", "hi"], env={"FOO": "bar"})

        self.mock_run.assert_called_once()
        _, kwargs = self.mock_run.call_args
        self.assertIsNotNone(kwargs.get("env"))
        self.assertEqual(kwargs["env"].get("FOO"), "bar")

    def test_execute_shell_command_failure(self):
        """Test _execute_shell_command raises ChsException on failure."""
        manager = RambleManager("/tmp", "ws1", self.mock_run)
        self.mock_run.return_value = subprocess.CompletedProcess(
            args=["bad_command"], returncode=1, stdout="", stderr="error"
        )
        # Fix: mock_run should NOT raise side_effect for check=False

        # We need to mock open explicitly for the log file?
        # setUp patches os.makedirs but not open.
        # But we are using tempdir, so it should be fine to write to it.

        with self.assertRaisesRegex(ChsException, "failed with exit code 1"):
            manager._execute_shell_command(["bad_command"])

    def test_prepare_ramble_command(self):
        """Test _prepare_ramble_command wraps command to source setup script."""
        manager = RambleManager("/tmp", "ws1", self.mock_run)
        cmd = ["ramble", "test"]
        expected_script_path = self._get_expected_setup_script_path()

        expected_cmd = [
            "bash",
            "-c",
            f'export RAMBLE_PYTHON="{sys.executable}" && '
            f'source {expected_script_path} && exec "$@"',
            "bash",
            "ramble",
            "test",
        ]

        prepared_cmd = manager._prepare_ramble_command(cmd)

        self.assertEqual(prepared_cmd, expected_cmd)

    def test_create_workspace(self):
        """Test create_workspace executes correct ramble command."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)

        manager.create_workspace()

        # New implementation uses capture_output=True, check=False
        # (check handled manually)

        # We can just verify the command args match what we expect.
        # The exact call arguments might differ slightly due to capture_output
        self.mock_run.assert_called_once()
        args, kwargs = self.mock_run.call_args
        self.assertEqual(kwargs["check"], False)
        # We check the command list
        cmd = args[0]
        self.assertIn("ramble", cmd)
        self.assertIn("workspace", cmd)
        self.assertIn("create", cmd)

    def test_setup_workspace(self):
        """Test setup_workspace executes correct ramble command."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        expected_script_path = self._get_expected_setup_script_path()

        manager.setup_workspace()

        self.mock_run.assert_called_once()
        args, kwargs = self.mock_run.call_args
        self.assertEqual(kwargs["check"], False)
        self.assertTrue(kwargs["capture_output"])
        cmd = args[0]
        self.assertIn("ramble", cmd)
        self.assertIn("workspace", cmd)
        self.assertIn("setup", cmd)
        self.assertIn(f"source {expected_script_path}", cmd[2])
        # Verify env contains TQDM_DISABLE
        self.assertEqual(kwargs["env"]["TQDM_DISABLE"], "1")
        _, kwargs = self.mock_run.call_args
        self.assertEqual(kwargs["env"].get("TQDM_DISABLE"), "1")
        self.assertEqual(kwargs["env"].get("TERM"), "dumb")

    def test_setup_workspace_with_scheduler_env(self):
        """Test setup_workspace merges extra env from scheduler."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        mock_scheduler = mock.Mock()
        mock_scheduler.get_extra_env.return_value = {"EXTRA_VAR": "value"}
        manager._scheduler = mock_scheduler

        manager.setup_workspace()

        self.mock_run.assert_called_once()
        _, kwargs = self.mock_run.call_args
        self.assertEqual(kwargs["env"].get("TQDM_DISABLE"), "1")
        self.assertEqual(kwargs["env"].get("EXTRA_VAR"), "value")

    def test_execute_experiments(self):
        """Test execute_experiments executes correct all_experiments script by
        default.
        """

        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        mock_scheduler = mock.Mock()
        mock_scheduler.name = "mock_scheduler"
        mock_scheduler.get_extra_env.return_value = {}
        mock_scheduler.extract_job_id.return_value = "123"
        manager._scheduler = mock_scheduler

        with (
            mock.patch("os.path.exists", return_value=True),
            mock.patch(
                "chs.ramble_manager.get_scheduler", return_value=mock_scheduler
            ),
            mock.patch(
                "os.walk",
                return_value=[
                    (
                        "/tmp/workspace/experiments/exp1",
                        [],
                        ["execute_experiment"],
                    )
                ],
            ),
            mock.patch.object(
                manager, "_update_registry"
            ) as mock_update_registry,
        ):
            job_ids = manager.execute_experiments()

        self.mock_run.assert_called_once()
        args, kwargs = self.mock_run.call_args
        cmd = args[0]
        self.assertEqual(
            cmd, ["bash", os.path.join(manager.path, "all_experiments")]
        )
        self.assertIsNone(kwargs.get("env"))

        mock_scheduler.extract_job_id.assert_called_with(
            "/tmp/workspace/experiments/exp1"
        )
        self.assertEqual(job_ids, ["123"])
        mock_update_registry.assert_called_once_with(
            {"123": "/tmp/workspace/experiments/exp1"}, tier=1
        )

        self.assertEqual(self.mock_app_data.update_run.call_count, 1)
        args, kwargs = self.mock_app_data.update_run.call_args
        self.assertEqual(args[0], "ws1")
        self.assertTrue(args[1].get("submission_complete"))

    def test_update_registry_preserves_custom_fields(self):
        """Test _update_registry preserves custom flags."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        mock_scheduler = mock.Mock()
        mock_scheduler.name = "kubernetes"
        manager._scheduler = mock_scheduler

        self.mock_app_data.get_run.return_value = {"kueue_supported": True}

        manager._update_registry({"job1": "/path/to/job1"})

        self.mock_app_data.update_run.assert_called_once()
        updated_run = self.mock_app_data.update_run.call_args[0][1]
        self.assertTrue(updated_run.get("kueue_supported"))
        self.assertIn("job1", updated_run.get("job_entries", {}))
        self.assertEqual(
            updated_run.get("job_entries")["job1"]["experiment_dir"],
            "/path/to/job1",
        )

    def test_configure_workspace_runs(self):
        """Test configure_workspace runs without error."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        mock_scheduler = mock.Mock()
        mock_scheduler.name = "mock_scheduler"
        self.mock_app_data.get_images_dir.return_value = "/tmp/images"

        with mock.patch("os.makedirs"):
            manager.configure_workspace(
                mock_scheduler, ["test1"], machine_type="a3-high"
            )
            manager.configure_workspace(
                mock_scheduler, None, machine_type="a3-high"
            )

        self.assertEqual(mock_scheduler.write_ramble_config.call_count, 2)
        mock_scheduler.write_ramble_config.assert_called_with(
            workspace_config_dir=mock.ANY,
            tests=None,
            machine_type="a3-high",
            timeout=None,
            deadline=None,
            run_id=None,
            images_dir="/tmp/images",
        )

    def test_repr(self):
        """Test __repr__ format."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        self.assertIn(f"{self.tmp_dir}/ws1", repr(manager))

    def test_analyze_experiments(self):
        """Test analyze_experiments runs the analyze command correctly."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        self.mock_run.return_value.stdout = "analysis complete"

        with mock.patch("os.path.exists", return_value=True):
            output_path = manager.analyze_experiments()

            for call_args in self.mock_run.call_args_list:
                args, kwargs = call_args
                cmd_list = args[0]
                if "analyze" in cmd_list:
                    self.assertIn("--fom-origin-types", cmd_list)
                    self.assertIn("modifier", cmd_list)
                    self.assertIn("application", cmd_list)
                    self.assertEqual(kwargs["env"]["TQDM_DISABLE"], "1")
                    self.assertEqual(kwargs["env"]["TERM"], "dumb")

            self.assertEqual(
                output_path, f"{self.tmp_dir}/ws1/results.latest.json"
            )

    def test_generate_boh(self):
        """Test generate_boh generates BoH and copies experiments."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        manager._scheduler = mock.Mock()
        manager._scheduler.name = "slurm"
        results_json_path = f"{self.tmp_dir}/ws1/results.latest.json"
        self.mock_app_data.get_storage_config.return_value = {
            "type": "gcs",
            "gcs": {"bucket_name": "my-bucket"},
        }

        with mock.patch("chs.ramble_manager.BOHGenerator") as mock_boh_gen:
            with mock.patch.object(manager, "cancel_experiments"):
                with mock.patch("os.getcwd", return_value="/home/user"):
                    self.mock_os_path_exists.side_effect = lambda p: True

                    # Mock PluginManager in core for Base PR isolation
                    with mock.patch("chs.core.plugin_manager.PluginManager"):
                        output_path = manager.generate_boh(results_json_path)

                        self.assertEqual(output_path, f"{self.tmp_dir}/ws1")
                        mock_boh_gen.assert_called_once_with(
                            results_json_path,
                            base_log_dir=f"{self.tmp_dir}/ws1",
                            cluster_type="slurm",
                            gcs_base_path="gs://my-bucket/workspaces/ws1",
                            thresholds_path=None,
                            skip_validation=False,
                        )
                    mock_boh_gen.return_value.generate.assert_called_once()

    def test_wait_for_jobs(self):
        """Test wait_for_jobs calls _monitor_jobs."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)

        with mock.patch.object(manager, "_monitor_jobs") as mock_monitor:
            manager.wait_for_jobs(
                self.mock_app_data, poll_interval=1, tier=None
            )
            mock_monitor.assert_called_once_with(self.mock_app_data, 1, None)

    def test_get_application_names(self):
        """Test extraction of unique application names from results."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        results_data = {
            "experiments": [
                {"application_name": "app1"},
                {"application_name": "app2"},
                {"application_name": "app1"},  # Duplicate
                {},  # Missing name
            ]
        }
        app_names = manager._get_application_names(results_data)
        self.assertEqual(app_names, {"app1", "app2"})

    def test_get_success_criteria_mapping(self):
        """Test fetching and parsing of success criteria mapping."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        mock_output = """
success_criteria # ####################
mycriteria:
  fom_name: 'metric1'
  formula: >= 10
"""
        self.mock_run.return_value.stdout = mock_output

        with mock.patch.object(manager, "_execute_ramble_command") as mock_exec:
            mock_exec.return_value.stdout = mock_output
            mock_exec.return_value.returncode = 0

            mapping = manager._get_success_criteria_mapping({"myapp"})

            self.assertIn("myapp", mapping)
            self.assertEqual(
                mapping["myapp"]["mycriteria"]["fom_name"], "metric1"
            )
            self.assertEqual(mapping["myapp"]["mycriteria"]["formula"], ">= 10")

            mock_exec.assert_called_once()
            args, kwargs = mock_exec.call_args
            self.assertEqual(kwargs["subcommand"], "info")
            self.assertIn("myapp", kwargs["args"])

    def test_get_success_criteria_mapping_formula_quoting(self):
        """Test that formulas with quotes are handled by regex replacement."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        # Raw output that needs formula quoting
        mock_output = """
success_criteria #
mycriteria:
  fom_name: 'metric1'
  formula: >= 10
"""
        with mock.patch.object(manager, "_execute_ramble_command") as mock_exec:
            mock_exec.return_value.stdout = mock_output
            mock_exec.return_value.returncode = 0
            mapping = manager._get_success_criteria_mapping({"myapp"})

            self.assertEqual(mapping["myapp"]["mycriteria"]["formula"], ">= 10")

    def test_cleanup_inputs_dir(self):
        """Test _cleanup_inputs_dir removes inputs directory."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        inputs_dir = os.path.join(manager.path, "inputs")
        os.makedirs(inputs_dir)
        self.assertTrue(os.path.exists(inputs_dir))
        manager._cleanup_inputs_dir()
        self.assertFalse(os.path.exists(inputs_dir))

    def test_cleanup_inputs_dir_not_exists(self):
        """Test _cleanup_inputs_dir does nothing if inputs dir not exists."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        inputs_dir = os.path.join(manager.path, "inputs")
        self.assertFalse(os.path.exists(inputs_dir))
        manager._cleanup_inputs_dir()
        self.assertFalse(os.path.exists(inputs_dir))

    def test_upload_boh_success(self):
        """Test _upload_boh calls provider.upload_file."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        self.mock_os_path_exists.side_effect = (
            lambda p: True
            if "bill_of_health.json" in p
            else self.real_os_path_exists(p)
        )
        mock_provider = mock.Mock()
        manager._upload_boh(mock_provider)
        mock_provider.upload_file.assert_called_once_with(
            os.path.join(manager.path, "bill_of_health.json"),
            "bills_of_health/ws1.json",
        )

    def test_upload_boh_failure(self):
        """Test _upload_boh raises ChsArchiveException on provider failure."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        self.mock_os_path_exists.side_effect = (
            lambda p: True
            if "bill_of_health.json" in p
            else self.real_os_path_exists(p)
        )
        mock_provider = mock.Mock()
        mock_provider.upload_file.side_effect = Exception("Upload failed")
        with self.assertRaises(ChsArchiveException):
            manager._upload_boh(mock_provider)

    def test_upload_workspace_success(self):
        """Test _upload_workspace archives and calls provider.upload_file."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        os.makedirs(
            os.path.join(manager.path, ".ramble-workspace"), exist_ok=True
        )
        mock_provider = mock.Mock()

        # Mock glob to return a dummy archive file
        mock_glob = mock.patch("pathlib.Path.glob").start()
        mock_archive_file = mock.Mock()
        mock_archive_file.name = "ws1.tar.gz"
        mock_glob.side_effect = (
            lambda pat: [mock_archive_file] if "tar.gz" in pat else []
        )

        with mock.patch.object(manager, "_execute_ramble_command") as mock_exec:
            manager._upload_workspace(mock_provider)

            mock_exec.assert_called_once_with(
                "workspace",
                [
                    "archive",
                    "-t",
                    "--archive-pattern",
                    "bill_of_health.json",
                    "--archive-pattern",
                    "internal.log",
                    "--archive-pattern",
                    "debug.txt",
                ],
                capture_output=True,
                error_context="Failed to archive workspace",
            )
            mock_provider.upload_file.assert_called_once_with(
                str(mock_archive_file), "workspaces/ws1.tar.gz"
            )
            mock_archive_file.unlink.assert_called_once()
        mock_glob.stop()

    def test_upload_workspace_no_archive_found(self):
        """Test _upload_workspace logs warning if no archive found."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        os.makedirs(
            os.path.join(manager.path, ".ramble-workspace"), exist_ok=True
        )
        mock_provider = mock.Mock()

        mock_glob = mock.patch("pathlib.Path.glob").start()
        mock_glob.return_value = []  # No files found

        with mock.patch.object(manager, "_execute_ramble_command") as mock_exec:
            manager._upload_workspace(mock_provider)
            mock_exec.assert_called_once()
            mock_provider.upload_file.assert_not_called()
        mock_glob.stop()

    def test_archive_success(self):
        """Test archive retrieves provider and calls upload methods."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        os.makedirs(
            os.path.join(manager.path, ".ramble-workspace"), exist_ok=True
        )
        mock_provider = mock.Mock()

        with (
            mock.patch(
                "chs.storage.StorageProviderFactory.get_active_provider",
                return_value=mock_provider,
            ),
            mock.patch.object(manager, "_upload_boh") as mock_boh,
            mock.patch.object(manager, "_upload_workspace") as mock_workspace,
        ):
            manager.archive()
            mock_boh.assert_called_once_with(mock_provider)
            mock_workspace.assert_called_once_with(mock_provider)

    def test_archive_no_provider(self):
        """Test archive skips upload if provider cannot be loaded."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        os.makedirs(
            os.path.join(manager.path, ".ramble-workspace"), exist_ok=True
        )
        with (
            mock.patch(
                "chs.storage.StorageProviderFactory.get_active_provider",
                side_effect=Exception("No provider"),
            ),
            mock.patch.object(manager, "_upload_boh") as mock_boh,
            mock.patch.object(manager, "_upload_workspace") as mock_workspace,
        ):
            manager.archive()
            mock_boh.assert_not_called()
            mock_workspace.assert_not_called()


if __name__ == "__main__":
    unittest.main()
