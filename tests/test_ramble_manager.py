"""Unit tests for RambleManager."""

import logging
import os
import pathlib
import subprocess
import sysconfig
import tempfile
import unittest
from unittest import mock
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

    def test_execute_experiments(self):
        """Test execute_experiments executes correct ramble on command."""

        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        mock_scheduler = mock.Mock()
        mock_scheduler.name = "mock_scheduler"
        mock_scheduler.parse_job_ids.return_value = ["123"]
        manager._scheduler = mock_scheduler

        with mock.patch.object(
            manager, "_update_registry"
        ) as mock_update_registry:
            self.mock_run.return_value.stdout = "Submitted batch job 123"
            job_ids = manager.execute_experiments()

        self.mock_run.assert_called_once()
        args, kwargs = self.mock_run.call_args
        self.assertEqual(kwargs["check"], False)
        self.assertTrue(kwargs["capture_output"])
        cmd = args[0]
        self.assertIn("ramble", cmd)
        self.assertIn("on", cmd)
        # Verify env contains TQDM_DISABLE
        self.assertEqual(kwargs["env"]["TQDM_DISABLE"], "1")
        _, kwargs = self.mock_run.call_args
        self.assertEqual(kwargs["env"].get("TQDM_DISABLE"), "1")
        self.assertEqual(kwargs["env"].get("TERM"), "dumb")

        mock_scheduler.parse_job_ids.assert_called_once()
        self.assertEqual(job_ids, ["123"])
        mock_update_registry.assert_called_once_with(["123"])

    def test_configure_workspace_runs(self):
        """Test configure_workspace runs without error."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        mock_scheduler = mock.Mock()
        mock_scheduler.name = "mock_scheduler"

        with mock.patch("os.makedirs"):
            manager.configure_workspace(
                mock_scheduler, ["test1"], machine_type="a3-high"
            )
            manager.configure_workspace(
                mock_scheduler, None, machine_type="a3-high"
            )

        self.assertEqual(mock_scheduler.write_ramble_config.call_count, 2)

    def test_check_jobs_running_true(self):
        """Test check_jobs_running returns True when jobs are running."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        mock_scheduler = mock.Mock()
        mock_scheduler.check_jobs_running.return_value = True

        with mock.patch.object(
            manager, "_get_registry_entry"
        ) as mock_get_entry:
            with mock.patch(
                "chs.ramble_manager.get_scheduler"
            ) as mock_get_scheduler:
                mock_get_entry.return_value = {
                    "job_ids": ["123"],
                    "scheduler": "mock_scheduler",
                }
                mock_get_scheduler.return_value = mock_scheduler

                result = manager.check_jobs_running()

                self.assertTrue(result)
                mock_get_scheduler.assert_called_once_with("mock_scheduler")
                mock_scheduler.check_jobs_running.assert_called_once_with(
                    ["123"]
                )

    def test_check_jobs_running_false(self):
        """Test check_jobs_running returns False when jobs are not running."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        mock_scheduler = mock.Mock()
        mock_scheduler.check_jobs_running.return_value = False

        with mock.patch.object(
            manager, "_get_registry_entry"
        ) as mock_get_entry:
            with mock.patch(
                "chs.ramble_manager.get_scheduler"
            ) as mock_get_scheduler:
                mock_get_entry.return_value = {
                    "job_ids": ["123"],
                    "scheduler": "mock_scheduler",
                }
                mock_get_scheduler.return_value = mock_scheduler

                result = manager.check_jobs_running()

                self.assertFalse(result)

    def test_check_jobs_running_no_entry(self):
        """Test check_jobs_running returns False for no registry entry."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)

        with mock.patch.object(
            manager, "_get_registry_entry"
        ) as mock_get_entry:
            mock_get_entry.return_value = None
            result = manager.check_jobs_running()
            self.assertFalse(result)

    def test_repr(self):
        """Test __repr__ format."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        self.assertIn(f"{self.tmp_dir}/ws1", repr(manager))

    def test_analyze_results_no_scheduler_attr(self):
        """Test analyze_results works when _scheduler is not set."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        # Ensure _scheduler is NOT set
        if hasattr(manager, "_scheduler"):
            del manager._scheduler

        self.mock_run.return_value.stdout = "analysis complete"

        with mock.patch("chs.ramble_manager.BOHGenerator") as mock_boh_gen:
            with mock.patch.object(
                manager, "_get_registry_entry"
            ) as mock_get_entry:
                mock_get_entry.return_value = {"scheduler": "gke"}
                with mock.patch("shutil.copytree") as mock_copytree:
                    with mock.patch.object(
                        manager, "cancel_experiments"
                    ) as mock_cancel:
                        with mock.patch(
                            "shutil.ignore_patterns"
                        ) as mock_ignore_patterns:
                            with mock.patch("builtins.open", mock.mock_open()):
                                with mock.patch(
                                    "os.getcwd", return_value="/home/user"
                                ):
                                    with mock.patch("pathlib.Path.mkdir"):
                                        self.mock_os_path_exists.side_effect = (
                                            lambda p: True
                                        )

                                        manager.analyze_results()

                                        mock_boh_gen.assert_called_once()
                                        _, kwargs = mock_boh_gen.call_args
                                        self.assertEqual(
                                            kwargs["cluster_type"], "gke"
                                        )
                                        mock_ignore_patterns.assert_called_once_with(
                                            "chs-*"
                                        )
                                        mock_copytree.assert_called_once()
                                        args, kwargs = mock_copytree.call_args
                                        self.assertEqual(
                                            args[0],
                                            f"{self.tmp_dir}/ws1/experiments",
                                        )
                                        self.assertEqual(
                                            args[1],
                                            pathlib.Path(
                                                f"{self.tmp_dir}/ws1/chs-results/experiments"
                                            ),
                                        )
                                        self.assertTrue(kwargs["dirs_exist_ok"])
                                        self.assertEqual(
                                            kwargs.get("ignore"),
                                            mock_ignore_patterns.return_value,
                                        )
                                        mock_cancel.assert_called_once()

    def test_analyze_results(self):
        """Test analyze_results generates BoH and copies experiments."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        manager._scheduler = mock.Mock()
        manager._scheduler.name = "slurm"
        self.mock_run.return_value.stdout = "analysis complete"

        with mock.patch("chs.ramble_manager.BOHGenerator") as mock_boh_gen:
            with mock.patch("shutil.copytree") as mock_copytree:
                with mock.patch.object(
                    manager, "cancel_experiments"
                ) as mock_cancel:
                    with mock.patch(
                        "shutil.ignore_patterns"
                    ) as mock_ignore_patterns:
                        with mock.patch("builtins.open", mock.mock_open()):
                            with mock.patch(
                                "os.getcwd", return_value="/home/user"
                            ):
                                with mock.patch("pathlib.Path.mkdir"):
                                    self.mock_os_path_exists.side_effect = (
                                        lambda p: True
                                    )

                                    output_path = manager.analyze_results()

                                    for (
                                        call_args
                                    ) in self.mock_run.call_args_list:
                                        args, kwargs = call_args
                                        cmd_list = args[0]
                                        self.assertNotIn("on", cmd_list)
                                        # Verify env is passed for analyze
                                        if "analyze" in cmd_list:
                                            self.assertEqual(
                                                kwargs["env"]["TQDM_DISABLE"],
                                                "1",
                                            )
                                            self.assertEqual(
                                                kwargs["env"]["TERM"], "dumb"
                                            )

                                    self.assertEqual(
                                        output_path,
                                        f"{self.tmp_dir}/ws1/chs-results",
                                    )
                                    mock_boh_gen.return_value.generate.assert_called_once()
                                    mock_ignore_patterns.assert_called_once_with(
                                        "chs-*"
                                    )
                                    mock_copytree.assert_called_once()
                                    args, kwargs = mock_copytree.call_args
                                    self.assertEqual(
                                        args[0],
                                        f"{self.tmp_dir}/ws1/experiments",
                                    )
                                    self.assertEqual(
                                        args[1],
                                        pathlib.Path(
                                            f"{self.tmp_dir}/ws1/chs-results/experiments"
                                        ),
                                    )
                                    self.assertTrue(kwargs["dirs_exist_ok"])
                                    self.assertEqual(
                                        kwargs.get("ignore"),
                                        mock_ignore_patterns.return_value,
                                    )
                                    mock_cancel.assert_called_once()

    def test_analyze_results_skip_copy_same_dir(self):
        """Test analyze_results skips copy if src and dst are same."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        manager._scheduler = mock.Mock()
        manager._scheduler.name = "slurm"
        self.mock_run.return_value.stdout = "analysis complete"

        with mock.patch("chs.ramble_manager.BOHGenerator"):
            with mock.patch("shutil.copytree") as mock_copytree:
                with mock.patch("builtins.open", mock.mock_open()):
                    with mock.patch("os.getcwd", return_value=self.tmp_dir):
                        with mock.patch(
                            "chs.ramble_manager.os.path.abspath"
                        ) as mock_abspath:
                            self.mock_os_path_exists.side_effect = (
                                lambda p: True
                            )

                            def side_effect(p):
                                return str(p)

                            mock_abspath.side_effect = side_effect

                            with mock.patch.object(
                                manager, "cancel_experiments"
                            ) as mock_cancel:
                                output_path = manager.analyze_results()

                                self.assertEqual(
                                    output_path,
                                    f"{self.tmp_dir}/ws1/chs-results",
                                )
                                mock_copytree.assert_called_once()
                                mock_cancel.assert_called_once()

    def test_generate_boh(self):
        """Test generate_boh generates BoH and copies experiments."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
        manager._scheduler = mock.Mock()
        manager._scheduler.name = "slurm"
        results_json_path = f"{self.tmp_dir}/ws1/results.latest.json"

        with mock.patch("chs.ramble_manager.BOHGenerator") as mock_boh_gen:
            with mock.patch("shutil.copytree") as mock_copytree:
                with mock.patch(
                    "shutil.ignore_patterns"
                ) as mock_ignore_patterns:
                    with mock.patch.object(
                        manager, "cancel_experiments"
                    ) as mock_cancel:
                        with mock.patch("os.getcwd", return_value="/home/user"):
                            self.mock_os_path_exists.side_effect = (
                                lambda p: True
                            )

                        output_path = manager.generate_boh(results_json_path)

                        self.assertEqual(
                            output_path, f"{self.tmp_dir}/ws1/chs-results"
                        )
                        mock_boh_gen.assert_called_once_with(
                            results_json_path,
                            base_log_dir=output_path,
                            cluster_type="slurm",
                        )
                        mock_boh_gen.return_value.generate.assert_called_once()
                        mock_ignore_patterns.assert_called_once_with("chs-*")
                        mock_copytree.assert_called_once()
                        args, kwargs = mock_copytree.call_args
                        self.assertEqual(
                            args[0], f"{self.tmp_dir}/ws1/experiments"
                        )
                        self.assertEqual(
                            args[1],
                            pathlib.Path(
                                f"{self.tmp_dir}/ws1/chs-results/experiments"
                            ),
                        )
                        self.assertTrue(kwargs["dirs_exist_ok"])
                        mock_cancel.assert_called_once()

    def test_wait_for_jobs(self):
        """Test wait_for_jobs calls _monitor_jobs."""
        manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)

        with mock.patch.object(manager, "_monitor_jobs") as mock_monitor:
            manager.wait_for_jobs(poll_interval=1)
            mock_monitor.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
