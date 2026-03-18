"""Unit tests for CHS Schedulers."""

import contextlib
import os
import subprocess
import tempfile
import unittest
from unittest import mock
from chs.schedulers.default import DefaultScheduler
from chs.schedulers.gke import GKEScheduler
from chs.schedulers.slurm import SlurmScheduler
import click


@mock.patch("chs.schedulers.slurm.logger")
class TestSlurmScheduler(unittest.TestCase):
    """Tests for SlurmScheduler."""

    def setUp(self):
        super().setUp()
        self.mock_app_data = mock.Mock()
        self.mock_app_data.get_images_dir.return_value = "/tmp/images"

    def test_name(self, mock_logger):
        scheduler = SlurmScheduler(app_data=self.mock_app_data)
        self.assertEqual(scheduler.name, "slurm")

    def test_get_nodes_under_test_nodes_arg(self, mock_logger):
        scheduler = SlurmScheduler(
            nodes="node1,node2", app_data=self.mock_app_data
        )
        self.assertEqual(scheduler.get_nodes_under_test(), ["node1", "node2"])

    def test_get_nodes_under_test_multiple_args_error(self, mock_logger):
        scheduler = SlurmScheduler(
            nodes="n1", nodelist="nl1", app_data=self.mock_app_data
        )
        with self.assertRaisesRegex(click.UsageError, "Only one of"):
            scheduler.get_nodes_under_test()

    @mock.patch("subprocess.run")
    def test_get_nodes_under_test_nodelist_arg(self, mock_run, mock_logger):
        scheduler = SlurmScheduler(
            nodelist="node-[1-2]", app_data=self.mock_app_data
        )
        mock_run.return_value.stdout = "node-1\nnode-2"

        nodes = scheduler.get_nodes_under_test()

        self.assertEqual(nodes, ["node-1", "node-2"])
        mock_run.assert_called_once()
        self.assertIn("scontrol", mock_run.call_args[0][0])

    @mock.patch("subprocess.run")
    def test_get_nodes_under_test_partition_arg(self, mock_run, mock_logger):
        """Tests get_nodes_under_test with partition argument."""
        scheduler = SlurmScheduler(partition="p1", app_data=self.mock_app_data)
        # sinfo -N natively outputs uncompressed extended hostnames
        mock_run.return_value.stdout = "node-1\nnode-2\n"

        nodes = scheduler.get_nodes_under_test()

        self.assertEqual(nodes, ["node-1", "node-2"])
        mock_run.assert_called_once()
        self.assertIn("sinfo", mock_run.call_args_list[0][0][0])

    def test_parse_job_ids(self, mock_logger):
        scheduler = SlurmScheduler(app_data=self.mock_app_data)
        output = "Submitted batch job 123\nSubmitted batch job 456"
        self.assertEqual(scheduler.parse_job_ids(output), ["123", "456"])

    @mock.patch("subprocess.run")
    def test_check_jobs_running_true(self, mock_run, mock_logger):
        scheduler = SlurmScheduler(app_data=self.mock_app_data)
        mock_run.return_value.stdout = "123"
        self.assertTrue(scheduler.check_jobs_running(["123"]))

    @mock.patch("subprocess.run")
    def test_check_jobs_running_false(self, mock_run, mock_logger):
        scheduler = SlurmScheduler(app_data=self.mock_app_data)
        mock_run.return_value.stdout = ""
        self.assertFalse(scheduler.check_jobs_running(["123"]))

    @mock.patch("os.makedirs")
    @mock.patch("os.path.exists")
    @mock.patch("os.path.expanduser")
    def test_ensure_enroot_directories(
        self, mock_expanduser, mock_exists, mock_makedirs, mock_logger
    ):
        """Test _ensure_enroot_directories creates missing paths."""
        scheduler = SlurmScheduler()
        mock_expanduser.return_value = "/home/user"
        mock_exists.side_effect = [False, False]  # Both don't exist

        scheduler._ensure_enroot_directories()

        self.assertEqual(mock_makedirs.call_count, 2)
        mock_makedirs.assert_any_call("/home/user/.enroot/run", exist_ok=True)
        mock_makedirs.assert_any_call("/home/user/.enroot/cache", exist_ok=True)

    @mock.patch("os.path.expanduser")
    def test_get_extra_env(self, mock_expanduser, mock_logger):
        """Test get_extra_env returns expected variables."""
        scheduler = SlurmScheduler()
        mock_expanduser.return_value = "/home/user"
        env = scheduler.get_extra_env()
        self.assertEqual(env["ENROOT_RUNTIME_PATH"], "/home/user/.enroot/run")
        self.assertEqual(env["ENROOT_CACHE_PATH"], "/home/user/.enroot/cache")


class TestDefaultScheduler(unittest.TestCase):
    """Tests for DefaultScheduler."""

    def test_name(self):
        scheduler = DefaultScheduler("gke", app_data=mock.Mock())
        self.assertEqual(scheduler.name, "gke")

    def test_get_nodes_under_test(self):
        scheduler = DefaultScheduler("gke", app_data=mock.Mock())
        self.assertEqual(scheduler.get_nodes_under_test(), [])

    def test_check_jobs_running(self):
        scheduler = DefaultScheduler("gke", app_data=mock.Mock())
        self.assertFalse(scheduler.check_jobs_running(["123"]))

    def test_parse_job_ids(self):
        scheduler = DefaultScheduler("gke", app_data=mock.Mock())
        self.assertEqual(scheduler.parse_job_ids("output"), [])


@mock.patch("chs.schedulers.gke.logger")
class TestGKEScheduler(unittest.TestCase):
    """Tests for GKEScheduler."""

    def setUp(self):
        super().setUp()
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = self.tmp_dir_obj.name
        self.addCleanup(self.tmp_dir_obj.cleanup)
        self.mock_app_data = mock.Mock()

    def test_init(self, mock_logger):
        """Test initialization."""
        scheduler = GKEScheduler(
            nodes="node1,node2", app_data=self.mock_app_data
        )
        self.assertEqual(scheduler.name, "gke")
        self.assertEqual(scheduler.nodes_arg, "node1,node2")
        self.assertIsNone(scheduler.nodepool_arg)

    def test_init_nodepool(self, mock_logger):
        """Test initialization with nodepool."""
        scheduler = GKEScheduler(nodepool="pool-1", app_data=self.mock_app_data)
        self.assertEqual(scheduler.name, "gke")
        self.assertIsNone(scheduler.nodes_arg)
        self.assertEqual(scheduler.nodepool_arg, "pool-1")

    def test_get_nodes_under_test(self, mock_logger):
        """Test get_nodes_under_test returns parsed nodes."""
        scheduler = GKEScheduler(
            nodes="node1, node2 ", app_data=self.mock_app_data
        )
        self.assertEqual(scheduler.get_nodes_under_test(), ["node1", "node2"])

    @mock.patch("subprocess.run")
    def test_get_nodes_under_test_empty(self, mock_run, mock_logger):
        """Test get_nodes_under_test returns empty list if no nodes."""
        scheduler = GKEScheduler(app_data=self.mock_app_data)
        mock_run.side_effect = subprocess.CalledProcessError(1, "kubectl")

        self.assertEqual(scheduler.get_nodes_under_test(), [])

        mock_logger.warning.assert_called()
        self.assertIn(
            "Failed to automatically discover nodes using kubectl",
            mock_logger.warning.call_args[0][0],
        )

    @mock.patch("subprocess.run")
    def test_get_nodes_under_test_nodepool(self, mock_run, mock_logger):
        """Test get_nodes_under_test uses nodepool to fetch nodes."""
        scheduler = GKEScheduler(nodepool="pool-1", app_data=self.mock_app_data)
        mock_run.return_value.stdout = "node-1 node-2"

        nodes = scheduler.get_nodes_under_test()

        self.assertEqual(nodes, ["node-1", "node-2"])
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertIn("kubectl", cmd)
        self.assertIn("-l", cmd)
        self.assertIn("cloud.google.com/gke-nodepool=pool-1", cmd)

    @mock.patch("subprocess.run")
    def test_get_nodes_under_test_nodes_precedence(self, mock_run, mock_logger):
        """Test nodes arg takes precedence over nodepool."""
        scheduler = GKEScheduler(
            nodes="node-1", nodepool="pool-1", app_data=self.mock_app_data
        )

        nodes = scheduler.get_nodes_under_test()

        self.assertEqual(nodes, ["node-1"])
        mock_run.assert_not_called()

    def test_write_ramble_config(self, mock_logger):
        """Test write_ramble_config generates correct YAML."""
        scheduler = GKEScheduler(nodes="node1", app_data=self.mock_app_data)
        workspace_config_dir = os.path.join(self.tmp_dir, "configs")
        os.makedirs(workspace_config_dir, exist_ok=True)

        mock_template_content = """
ramble:
  variants:
    workflow_manager: gke-suite-configs
  variables:
    nodes: {nodes}
    machine_type: {machine_type}
    tests: {tests}
"""
        with mock.patch("chs.schedulers.gke.pkg_resources.path"):
            with mock.patch(
                "builtins.open", mock.mock_open(read_data=mock_template_content)
            ):
                scheduler.write_ramble_config(
                    workspace_config_dir,
                    ["test1"],
                    "a3-high",
                )

    def test_write_ramble_config_integration(self, mock_logger):
        """Integration-like test for write_ramble_config using real template."""
        scheduler = GKEScheduler(nodes="node1", app_data=self.mock_app_data)
        workspace_config_dir = os.path.join(self.tmp_dir, "configs")
        os.makedirs(workspace_config_dir, exist_ok=True)
        template_path = os.path.join(self.tmp_dir, "gke_template.yaml")
        with open(template_path, "w") as f:
            f.write("""
variables:
  nodes: {nodes}
  machine_type: {machine_type}
  tests: {tests}
  gpu_subnet_prefix: 'rdma'
""")

        @contextlib.contextmanager
        def mock_path_func(package, resource):
            yield template_path

        with mock.patch(
            "chs.schedulers.gke.pkg_resources.path", side_effect=mock_path_func
        ):
            scheduler.write_ramble_config(
                workspace_config_dir, ["test1"], "a3-high"
            )

        output_path = os.path.join(workspace_config_dir, "ramble.yaml")
        with open(output_path, "r") as f:
            content = f.read()

        self.assertIn('nodes: ["node1"]', content)
        self.assertIn("machine_type: a3-high", content)
        self.assertIn('tests: ["test1"]', content)
        self.assertIn("gpu_subnet_prefix: 'rdma'", content)


if __name__ == "__main__":
    unittest.main()
