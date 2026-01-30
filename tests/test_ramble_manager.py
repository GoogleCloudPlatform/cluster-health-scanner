"""Unit tests for RambleManager."""

import os
import subprocess
import sysconfig
import unittest
from unittest import mock
import logging
from chs.ramble_manager import RambleManager
import click


class TestRambleManager(unittest.TestCase):
  """Tests for RambleManager."""

  def setUp(self):
    super().setUp()
    self.mock_run = mock.Mock(spec=subprocess.run)
    self.mock_sysconfig = self.enterContext(
        mock.patch.object(sysconfig, "get_path", autospec=True)
    )
    self.mock_os_path_exists = self.enterContext(
        mock.patch.object(os.path, "exists", autospec=True)
    )
    self.mock_os_path_abspath = self.enterContext(
        mock.patch.object(os.path, "abspath", autospec=True)
    )

    self.mock_sysconfig.return_value = "/venv/lib/python/site-packages"
    self.mock_os_path_exists.return_value = True
    # Make abspath return the path it's given for simplicity in tests
    self.mock_os_path_abspath.side_effect = lambda x: x
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
    self.mock_os_path_exists.assert_called_once_with(
        os.path.abspath(self._get_expected_setup_script_path())
    )

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
    manager = RambleManager("/tmp", "ws1", self.mock_run)
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
            "/tmp/ws1",
        ],
        check=True,
        text=True,
        cwd=None,
        stdout=subprocess.DEVNULL,
    )

  def test_setup_workspace(self):
    """Test setup_workspace executes correct ramble command."""
    manager = RambleManager("/tmp", "ws1", self.mock_run)
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
            "/tmp/ws1",
            "workspace",
            "setup",
        ],
        check=True,
        text=True,
        cwd="/tmp/ws1",
        stdout=subprocess.DEVNULL,
    )

  def test_execute_experiments(self):
    """Test execute_experiments executes correct ramble command."""
    manager = RambleManager("/tmp", "ws1", self.mock_run)
    expected_script_path = self._get_expected_setup_script_path()

    manager.execute_experiments()

    self.mock_run.assert_called_once_with(
        [
            "bash",
            "-c",
            f'source {expected_script_path} && exec "$@"',
            "bash",
            "ramble",
            "-w",
            "/tmp/ws1",
            "on",
        ],
        check=True,
        text=True,
        cwd="/tmp/ws1",
        stdout=subprocess.DEVNULL,
    )

  def test_configure_workspace_runs(self):
    """Test configure_workspace runs without error."""
    manager = RambleManager("/tmp", "ws1", self.mock_run)
    manager.configure_workspace("suite1", "slurm", ["test1"])
    manager.configure_workspace("suite1", "slurm", None)

  def test_repr(self):
    """Test __repr__ format."""
    manager = RambleManager("/tmp", "ws1", self.mock_run)
    self.assertEqual(
        repr(manager), "RambleManager(workspace_name='ws1', path='/tmp/ws1')"
    )


if __name__ == "__main__":
  unittest.main()
