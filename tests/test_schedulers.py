"""Unit tests for CHS Schedulers."""

import contextlib
import os
import subprocess
import tempfile
import unittest
import json
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

    def test_discover_network_interfaces_with_anchor(self, mock_logger):
        """Test _discover_network_interfaces uses default route as anchor."""
        scheduler = SlurmScheduler(app_data=self.mock_app_data)
        with mock.patch.object(scheduler, "_run_srun") as mock_srun:
            mock_srun.side_effect = [
                "default via 192.168.0.1 dev enp0s19 proto dhcp",
                "default via 192.168.0.1 dev enp0s19 proto dhcp\n"
                "192.168.64.0/18 via 192.168.64.1 dev enp192s20 proto dhcp\n"
                "192.168.128.0/21 via 192.168.128.1 dev gpu0rdma0 proto dhcp",
            ]
            interfaces = scheduler._discover_network_interfaces()
            self.assertEqual(interfaces, ["enp0s19", "enp192s20"])

    def test_discover_network_interfaces_no_anchor_empty(self, mock_logger):
        """Test that no interface is returned when no default route is found."""
        scheduler = SlurmScheduler(app_data=self.mock_app_data)
        with mock.patch.object(scheduler, "_run_srun") as mock_srun:
            mock_srun.side_effect = [
                "",  # No default route
                "192.168.0.0/24 dev eth0 proto kernel\n"
                "172.17.0.0/16 dev docker0",
            ]
            interfaces = scheduler._discover_network_interfaces()
            self.assertEqual(interfaces, [])
            mock_logger.info.assert_called_with(
                "Detected network interfaces: %r", []
            )


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
    default_interface_name: {default_interface_name}
    all_interfaces_json: {all_interfaces_json}
"""
        with mock.patch.object(
            GKEScheduler,
            "_discover_networks",
            return_value=(["default", "gvnic-1"], ["gpu-net-1"]),
        ):
            with mock.patch("chs.schedulers.gke.pkg_resources.as_file"):
                with mock.patch(
                    "builtins.open",
                    mock.mock_open(read_data=mock_template_content),
                ):
                    scheduler.write_ramble_config(
                        workspace_config_dir, ["test1"], "a3-high"
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
  default_interface_name: {default_interface_name}
  all_interfaces_json: {all_interfaces_json}
""")

        @contextlib.contextmanager
        def mock_as_file_func(path):
            yield template_path

        with mock.patch.object(
            GKEScheduler,
            "_discover_networks",
            return_value=(["default"], ["rdma-0"]),
        ):
            with mock.patch(
                "chs.schedulers.gke.pkg_resources.as_file",
                side_effect=mock_as_file_func,
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
        self.assertIn("default_interface_name: eth0", content)
        self.assertIn(
            'all_interfaces_json: [{"interfaceName": "eth0", '
            '"network": "default"}, {"interfaceName": "eth1", '
            '"network": "rdma-0"}]',
            content,
        )

    @mock.patch("subprocess.run")
    def test_discover_networks_success(self, mock_run, mock_logger):
        """Test _discover_networks with success and grouping."""
        scheduler = GKEScheduler()
        mock_run.return_value.stdout = json.dumps(
            {
                "items": [
                    {
                        "metadata": {"name": "rdma-1"},
                        "spec": {"type": "Device"},
                    },
                    {
                        "metadata": {"name": "rdma-0"},
                        "spec": {"type": "Device"},
                    },
                    {"metadata": {"name": "gvnic-1"}, "spec": {"type": "L3"}},
                    {"metadata": {"name": "default"}, "spec": {"type": "L3"}},
                ]
            }
        )

        networks_l3, networks_device = scheduler._discover_networks()

        self.assertEqual(networks_l3, ["default", "gvnic-1"])
        self.assertEqual(networks_device, ["rdma-0", "rdma-1"])
        mock_run.assert_called_once()
        self.assertIn("kubectl", mock_run.call_args[0][0])
        self.assertIn("networks.networking.gke.io", mock_run.call_args[0][0])
        self.assertIn("-o", mock_run.call_args[0][0])
        self.assertIn("json", mock_run.call_args[0][0])

    @mock.patch("subprocess.run")
    def test_discover_networks_failure(self, mock_run, mock_logger):
        """Test _discover_networks with failure."""
        scheduler = GKEScheduler()
        mock_run.side_effect = subprocess.CalledProcessError(1, "kubectl")

        networks = scheduler._discover_networks()

        self.assertEqual(networks, ([], []))

    @mock.patch("subprocess.run")
    def test_discover_networks_no_default(self, mock_run, mock_logger):
        """Test _discover_networks fails if default L3 network is missing."""
        scheduler = GKEScheduler()
        mock_run.return_value.stdout = json.dumps(
            {
                "items": [
                    {"metadata": {"name": "gvnic-1"}, "spec": {"type": "L3"}},
                ]
            }
        )
        networks_l3, networks_device = scheduler._discover_networks()
        self.assertEqual(networks_l3, [])
        self.assertEqual(networks_device, [])
        mock_logger.warning.assert_called_with(
            "Default L3 network not found. Discovered L3: %s", "gvnic-1"
        )

    @mock.patch("subprocess.run")
    def test_discover_networks_malformed_items(self, mock_run, mock_logger):
        """Test _discover_networks handles non-list items field."""
        scheduler = GKEScheduler()
        mock_run.return_value.stdout = json.dumps({"items": "not-a-list"})
        networks_l3, networks_device = scheduler._discover_networks()
        self.assertEqual(networks_l3, [])
        self.assertEqual(networks_device, [])
        mock_logger.debug.assert_called_with(
            "Unexpected network discovery output format: 'items' is not a list"
        )

    @mock.patch("subprocess.run")
    def test_write_ramble_config_discovery_failure(self, mock_run, mock_logger):
        """Test write_ramble_config raises NetworkDiscoveryException."""
        from chs.exceptions import NetworkDiscoveryException

        scheduler = GKEScheduler()
        with mock.patch.object(
            GKEScheduler, "_discover_networks", return_value=([], [])
        ):
            with self.assertRaises(NetworkDiscoveryException):
                scheduler.write_ramble_config("/tmp", ["test1"], "a3-high")


if __name__ == "__main__":
    unittest.main()
