"""Unit tests for RambleManager."""

import logging
import os
import pathlib
import shutil
import subprocess
import sysconfig
import tempfile
import unittest
from unittest import mock
from chs.ramble_manager import RambleManager
import click


class TestRambleManager(unittest.TestCase):
  """Tests for RambleManager."""

  def setUp(self):
    super().setUp()
    self.tmp_dir_obj = tempfile.TemporaryDirectory()
    self.tmp_dir = self.tmp_dir_obj.name
    self.addCleanup(self.tmp_dir_obj.cleanup)
    self.mock_run = mock.Mock(spec=subprocess.run)
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

    manager._execute_shell_command(["echo", "hi"])

    self.mock_run.assert_called_once_with(
        ["echo", "hi"],
        check=True,
        text=True,
        cwd=None,
        stdout=subprocess.DEVNULL,
        stderr=None,
    )

  def test_execute_shell_command_failure(self):
    """Test _execute_shell_command raises ClickException on failure."""
    manager = RambleManager("/tmp", "ws1", self.mock_run)
    self.mock_run.side_effect = subprocess.CalledProcessError(
        1, ["bad_command"]
    )

    with self.assertRaisesRegex(
        click.ClickException, "failed with exit code 1"
    ):
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
    """Test create_workspace executes correct ramble command with setup script."""
    manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
    expected_script_path = self._get_expected_setup_script_path()

    manager.create_workspace()

    self.mock_run.assert_called_once_with(
        [
            "bash",
            "-c",
            f'source {expected_script_path} && exec "$@"',
            "bash",
            "ramble",
            "workspace",
            "create",
            "-a",
            "-d",
            f"{self.tmp_dir}/ws1",
        ],
        check=True,
        text=True,
        cwd=None,
        stdout=subprocess.DEVNULL,
        stderr=None,
    )

  def test_setup_workspace(self):
    """Test setup_workspace executes correct ramble command."""
    manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
    expected_script_path = self._get_expected_setup_script_path()

    manager.setup_workspace()

    self.mock_run.assert_called_once_with(
        [
            "bash",
            "-c",
            f'source {expected_script_path} && exec "$@"',
            "bash",
            "ramble",
            "-w",
            f"{self.tmp_dir}/ws1",
            "workspace",
            "setup",
        ],
        check=True,
        text=True,
        cwd=f"{self.tmp_dir}/ws1",
        stdout=subprocess.DEVNULL,
        stderr=None,
    )

  def test_execute_experiments(self):
    """Test execute_experiments executes correct ramble command and parses job IDs."""

    manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
    mock_scheduler = mock.Mock()
    mock_scheduler.name = "mock_scheduler"
    mock_scheduler.parse_job_ids.return_value = ["123"]
    manager._scheduler = mock_scheduler
    expected_script_path = self._get_expected_setup_script_path()

    with mock.patch.object(manager, "_update_registry") as mock_update_registry:
      self.mock_run.return_value.stdout = "Submitted batch job 123"
      job_ids = manager.execute_experiments()

    self.mock_run.assert_called_once_with(
        [
            "bash",
            "-c",
            f'source {expected_script_path} && exec "$@"',
            "bash",
            "ramble",
            "-w",
            f"{self.tmp_dir}/ws1",
            "on",
        ],
        check=True,
        text=True,
        cwd=f"{self.tmp_dir}/ws1",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

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
          "suite1", mock_scheduler, ["test1"], machine_type="a3-high"
      )
      manager.configure_workspace(
          "suite1", mock_scheduler, None, machine_type="a3-high"
      )

    self.assertEqual(mock_scheduler.write_ramble_config.call_count, 2)

  def test_check_jobs_running_true(self):
    """Test check_jobs_running returns True when jobs are running."""
    manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
    mock_scheduler = mock.Mock()
    mock_scheduler.check_jobs_running.return_value = True

    with mock.patch.object(manager, "_get_registry_entry") as mock_get_entry:
      with mock.patch("chs.ramble_manager.get_scheduler") as mock_get_scheduler:
        mock_get_entry.return_value = {
            "job_ids": ["123"],
            "scheduler": "mock_scheduler",
        }
        mock_get_scheduler.return_value = mock_scheduler

        result = manager.check_jobs_running()

        self.assertTrue(result)
        mock_get_scheduler.assert_called_once_with("mock_scheduler")
        mock_scheduler.check_jobs_running.assert_called_once_with(["123"])

  def test_check_jobs_running_false(self):
    """Test check_jobs_running returns False when jobs are not running."""
    manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
    mock_scheduler = mock.Mock()
    mock_scheduler.check_jobs_running.return_value = False

    with mock.patch.object(manager, "_get_registry_entry") as mock_get_entry:
      with mock.patch("chs.ramble_manager.get_scheduler") as mock_get_scheduler:
        mock_get_entry.return_value = {
            "job_ids": ["123"],
            "scheduler": "mock_scheduler",
        }
        mock_get_scheduler.return_value = mock_scheduler

        result = manager.check_jobs_running()

        self.assertFalse(result)

  def test_check_jobs_running_no_entry(self):
    """Test check_jobs_running returns False when no registry entry found."""
    manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)

    with mock.patch.object(manager, "_get_registry_entry") as mock_get_entry:
      mock_get_entry.return_value = None
      result = manager.check_jobs_running()
      self.assertFalse(result)

  def test_repr(self):
    """Test __repr__ format."""
    manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
    self.assertIn(f"{self.tmp_dir}/ws1", repr(manager))

  def test_analyze_results(self):
    """Test analyze_results generates BoH and copies experiments."""
    manager = RambleManager(self.tmp_dir, "ws1", self.mock_run)
    self.mock_run.return_value.stdout = "analysis complete"

    with mock.patch("chs.ramble_manager.BOHGenerator") as mock_boh_gen:
      with mock.patch("shutil.copytree") as mock_copytree:
        with mock.patch("shutil.ignore_patterns") as mock_ignore_patterns:
          with mock.patch("builtins.open", mock.mock_open()) as mock_file:
            with mock.patch("os.getcwd", return_value="/home/user"):
              with mock.patch("pathlib.Path.mkdir") as mock_mkdir:
                self.mock_os_path_exists.side_effect = lambda p: True

                output_path = manager.analyze_results()

                self.assertEqual(output_path, "/home/user/ws1")
                mock_boh_gen.return_value.generate.assert_called_once()
                mock_ignore_patterns.assert_called_once_with("chs-*")
                mock_copytree.assert_called_once()
                args, kwargs = mock_copytree.call_args
                self.assertEqual(args[0], f"{self.tmp_dir}/ws1/experiments")
                self.assertEqual(
                    args[1], pathlib.Path("/home/user/ws1/experiments")
                )
                self.assertTrue(kwargs["dirs_exist_ok"])
                self.assertEqual(
                    kwargs.get("ignore"), mock_ignore_patterns.return_value
                )


if __name__ == "__main__":
  unittest.main()
