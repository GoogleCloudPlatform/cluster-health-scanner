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

    def test_name(self, mock_logger):
        scheduler = SlurmScheduler()
        self.assertEqual(scheduler.name, "slurm")

    def test_get_nodes_under_test_nodes_arg(self, mock_logger):
        scheduler = SlurmScheduler(nodes="node1,node2")
        self.assertEqual(scheduler.get_nodes_under_test(), ["node1", "node2"])

    def test_get_nodes_under_test_multiple_args_error(self, mock_logger):
        scheduler = SlurmScheduler(nodes="n1", nodelist="nl1")
        with self.assertRaisesRegex(click.UsageError, "Only one of"):
            scheduler.get_nodes_under_test()

    @mock.patch("subprocess.run")
    def test_get_nodes_under_test_nodelist_arg(self, mock_run, mock_logger):
        scheduler = SlurmScheduler(nodelist="node-[1-2]")
        mock_run.return_value.stdout = "node-1\nnode-2"

        nodes = scheduler.get_nodes_under_test()

        self.assertEqual(nodes, ["node-1", "node-2"])
        mock_run.assert_called_once()
        self.assertIn("scontrol", mock_run.call_args[0][0])

    @mock.patch("subprocess.run")
    def test_get_nodes_under_test_partition_arg(self, mock_run, mock_logger):
        """Tests get_nodes_under_test with partition argument."""
        scheduler = SlurmScheduler(partition="p1")
        mock_run.side_effect = [
            mock.Mock(stdout="node-[1-2]"),
            mock.Mock(stdout="node-1\nnode-2"),
        ]

        nodes = scheduler.get_nodes_under_test()

        self.assertEqual(nodes, ["node-1", "node-2"])
        self.assertEqual(mock_run.call_count, 2)
        self.assertIn("sinfo", mock_run.call_args_list[0][0][0])
        self.assertIn("scontrol", mock_run.call_args_list[1][0][0])

    def test_parse_job_ids(self, mock_logger):
        scheduler = SlurmScheduler()
        output = "Submitted batch job 123\nSubmitted batch job 456"
        self.assertEqual(scheduler.parse_job_ids(output), ["123", "456"])

    @mock.patch("subprocess.run")
    def test_check_jobs_running_true(self, mock_run, mock_logger):
        scheduler = SlurmScheduler()
        mock_run.return_value.stdout = "123"
        self.assertTrue(scheduler.check_jobs_running(["123"]))

    @mock.patch("subprocess.run")
    def test_check_jobs_running_false(self, mock_run, mock_logger):
        scheduler = SlurmScheduler()
        mock_run.return_value.stdout = ""
        self.assertFalse(scheduler.check_jobs_running(["123"]))


class TestDefaultScheduler(unittest.TestCase):
    """Tests for DefaultScheduler."""

    def test_name(self):
        scheduler = DefaultScheduler("gke")
        self.assertEqual(scheduler.name, "gke")

    def test_get_nodes_under_test(self):
        scheduler = DefaultScheduler("gke")
        self.assertEqual(scheduler.get_nodes_under_test(), [])

    def test_check_jobs_running(self):
        scheduler = DefaultScheduler("gke")
        self.assertFalse(scheduler.check_jobs_running(["123"]))

    def test_parse_job_ids(self):
        scheduler = DefaultScheduler("gke")
        self.assertEqual(scheduler.parse_job_ids("output"), [])


@mock.patch("chs.schedulers.gke.logger")
class TestGKEScheduler(unittest.TestCase):
    """Tests for GKEScheduler."""

    def setUp(self):
        super().setUp()
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = self.tmp_dir_obj.name
        self.addCleanup(self.tmp_dir_obj.cleanup)

    def test_init(self, mock_logger):
        """Test initialization."""
        scheduler = GKEScheduler(nodes="node1,node2")
        self.assertEqual(scheduler.name, "gke")
        self.assertEqual(scheduler.nodes_arg, "node1,node2")
        self.assertIsNone(scheduler.nodepool_arg)

    def test_init_nodepool(self, mock_logger):
        """Test initialization with nodepool."""
        scheduler = GKEScheduler(nodepool="pool-1")
        self.assertEqual(scheduler.name, "gke")
        self.assertIsNone(scheduler.nodes_arg)
        self.assertEqual(scheduler.nodepool_arg, "pool-1")

    def test_get_nodes_under_test(self, mock_logger):
        """Test get_nodes_under_test returns parsed nodes."""
        scheduler = GKEScheduler(nodes="node1, node2 ")
        self.assertEqual(scheduler.get_nodes_under_test(), ["node1", "node2"])

    @mock.patch("subprocess.run")
    def test_get_nodes_under_test_empty(self, mock_run, mock_logger):
        """Test get_nodes_under_test returns empty list if no nodes."""
        scheduler = GKEScheduler()
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
        scheduler = GKEScheduler(nodepool="pool-1")
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
        scheduler = GKEScheduler(nodes="node-1", nodepool="pool-1")

        nodes = scheduler.get_nodes_under_test()

        self.assertEqual(nodes, ["node-1"])
        mock_run.assert_not_called()

    def test_write_ramble_config(self, mock_logger):
        """Test write_ramble_config generates correct YAML."""
        scheduler = GKEScheduler(nodes="node1")
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
                    workspace_config_dir, "suite", ["test1"], "a3-high"
                )

    def test_write_ramble_config_integration(self, mock_logger):
        """Integration-like test for write_ramble_config using real template."""
        scheduler = GKEScheduler(nodes="node1")
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
                workspace_config_dir, "suite", ["test1"], "a3-high"
            )

        output_path = os.path.join(workspace_config_dir, "ramble.yaml")
        with open(output_path, "r") as f:
            content = f.read()

        self.assertIn("nodes: [node1]", content)
        self.assertIn("machine_type: a3-high", content)
        self.assertIn("tests: [test1]", content)
        self.assertIn("gpu_subnet_prefix: 'rdma'", content)


if __name__ == "__main__":
    unittest.main()
