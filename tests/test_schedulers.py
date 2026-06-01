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
        mock_exists.side_effect = [False, False, False]  # All 3 don't exist

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
        self.assertEqual(
            env["ENROOT_CONFIG_CREDENTIALS"], "/home/user/.enroot/.credentials"
        )

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
        self.mock_app_data.get_run.return_value = {}

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
ramble:
  variables:
    nodes: {nodes}
    machine_type: {machine_type}
    tests: {tests}
    default_interface_name: {default_interface_name}
    all_interfaces_json: |
      {all_interfaces_json}
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
        import yaml

        with open(output_path, "r") as f:
            data = yaml.safe_load(f)

        variables = data["ramble"]["variables"]
        self.assertEqual(variables["nodes"], ["node1"])
        self.assertEqual(variables["machine_type"], "a3-high")
        self.assertEqual(variables["tests"], ["test1"])
        self.assertEqual(variables["default_interface_name"], "eth0")

        import json

        all_interfaces = json.loads(variables["all_interfaces_json"].strip())
        self.assertEqual(len(all_interfaces), 2)
        self.assertEqual(all_interfaces[0]["interfaceName"], "eth0")
        self.assertEqual(all_interfaces[0]["network"], "default")

    def test_write_ramble_config_with_timeout(self, mock_logger):
        """Test write_ramble_config injects timeout and deadline."""
        scheduler = GKEScheduler(nodes="node1", app_data=self.mock_app_data)
        workspace_config_dir = os.path.join(self.tmp_dir, "configs")
        os.makedirs(workspace_config_dir, exist_ok=True)
        template_path = os.path.join(self.tmp_dir, "gke_template.yaml")
        with open(template_path, "w") as f:
            f.write("""
ramble:
  variables:
    nodes: {nodes}
    machine_type: {machine_type}
    tests: {tests}
    default_interface_name: {default_interface_name}
    all_interfaces_json: |
      {all_interfaces_json}
    timeout: {timeout}
    deadline: {deadline}
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
                    workspace_config_dir,
                    ["test1"],
                    "a3-high",
                    timeout=600,
                    deadline=1800,
                )

        output_path = os.path.join(workspace_config_dir, "ramble.yaml")
        import yaml

        with open(output_path, "r") as f:
            data = yaml.safe_load(f)

        variables = data["ramble"]["variables"]
        self.assertEqual(variables["timeout"], 600)
        self.assertEqual(variables["deadline"], 1800)

    @mock.patch("chs.schedulers.gke.KueueManager")
    def test_setup_scheduler_resources_kueue_success(
        self, mock_kueue_mgr_class, mock_logger
    ):
        """Test setup_scheduler_resources caches labels on success."""
        scheduler = GKEScheduler(nodes="node1", app_data=self.mock_app_data)
        scheduler.kueue_supported = True
        self.mock_app_data.get_run.return_value = {"kueue_supported": True}

        mock_kueue_mgr_instance = mock_kueue_mgr_class.return_value
        mock_kueue_mgr_instance.setup.return_value = (
            "kueue.x-k8s.io/queue-name: chs-local-queue"
        )

        workspace_config_dir = os.path.join(self.tmp_dir, "configs")

        tests = scheduler.setup_scheduler_resources(
            workspace_config_dir,
            [{"name": "nccl-cluster"}],
            "a3-high",
            workspace_name="ws1",
        )

        self.assertEqual(tests, [{"name": "nccl-cluster"}])
        self.assertEqual(
            scheduler._jobset_labels_yaml,
            "kueue.x-k8s.io/queue-name: chs-local-queue",
        )
        self.assertEqual(scheduler._suspend_jobset, "true")

    @mock.patch("chs.schedulers.gke.KueueManager")
    def test_setup_scheduler_resources_kueue_fallback(
        self, mock_kueue_mgr_class, mock_logger
    ):
        """Test setup_scheduler_resources falls back if Kueue setup fails."""
        from chs.exceptions import KueueException

        scheduler = GKEScheduler(nodes="node1", app_data=self.mock_app_data)
        scheduler.kueue_supported = True
        self.mock_app_data.get_run.return_value = {"kueue_supported": True}

        mock_kueue_mgr_instance = mock_kueue_mgr_class.return_value
        mock_kueue_mgr_instance.setup.side_effect = KueueException(
            "setup failed"
        )

        workspace_config_dir = os.path.join(self.tmp_dir, "configs")

        tests = scheduler.setup_scheduler_resources(
            workspace_config_dir,
            [{"name": "nccl-cluster"}, {"name": "test1"}],
            "a3-high",
            workspace_name="ws1",
        )

        self.assertEqual(tests, [{"name": "test1"}])
        self.assertFalse(scheduler.kueue_supported)
        self.assertEqual(scheduler._jobset_labels_yaml, "")
        self.assertEqual(scheduler._suspend_jobset, "false")

    def test_write_ramble_config_uses_cached_labels(self, mock_logger):
        """Test write_ramble_config writes cached properties."""
        scheduler = GKEScheduler(nodes="node1", app_data=self.mock_app_data)
        scheduler._jobset_labels_yaml = "kueue/queue: local-q"
        scheduler._suspend_jobset = "true"

        workspace_config_dir = os.path.join(self.tmp_dir, "configs")
        os.makedirs(workspace_config_dir, exist_ok=True)
        template_path = os.path.join(self.tmp_dir, "gke_template.yaml")
        with open(template_path, "w") as f:
            f.write("""
ramble:
  variables:
    nodes: []
    machine_type: ""
    tests: []
    default_interface_name: ""
    all_interfaces_json: ""
    jobset_labels_yaml: ""
    suspend_jobset: false
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
                    workspace_config_dir,
                    [{"name": "test1"}],
                    "a3-high",
                )

        output_path = os.path.join(workspace_config_dir, "ramble.yaml")
        import yaml

        with open(output_path, "r") as f:
            data = yaml.safe_load(f)

        variables = data["ramble"]["variables"]
        self.assertEqual(
            variables["jobset_labels_yaml"], "kueue/queue: local-q"
        )
        self.assertEqual(variables["suspend_jobset"], "true")

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
    def test_write_ramble_config_discovery_fallback(
        self, mock_run, mock_logger
    ):
        """Test write_ramble_config network discovery fallback."""
        mock_run.return_value.stdout = "{}"
        scheduler = GKEScheduler(nodes="node1", app_data=self.mock_app_data)
        workspace_config_dir = os.path.join(self.tmp_dir, "configs")
        os.makedirs(workspace_config_dir, exist_ok=True)
        template_path = os.path.join(self.tmp_dir, "gke_template.yaml")
        with open(template_path, "w") as f:
            f.write("""
ramble:
  variables:
    nodes: {nodes}
    machine_type: {machine_type}
    tests: {tests}
    default_interface_name: {default_interface_name}
    all_interfaces_json: |
      {all_interfaces_json}
""")

        @contextlib.contextmanager
        def mock_as_file_func(path):
            yield template_path

        with mock.patch.object(
            GKEScheduler, "_discover_networks", return_value=([], [])
        ):
            with mock.patch(
                "chs.schedulers.gke.pkg_resources.as_file",
                side_effect=mock_as_file_func,
            ):
                scheduler.write_ramble_config(
                    workspace_config_dir, ["test1"], "a3-high"
                )

        output_path = os.path.join(workspace_config_dir, "ramble.yaml")
        import yaml

        with open(output_path, "r") as f:
            data = yaml.safe_load(f)

        variables = data["ramble"]["variables"]
        self.assertEqual(variables["default_interface_name"], "eth0")

        import json

        all_interfaces = json.loads(variables["all_interfaces_json"].strip())
        self.assertEqual(len(all_interfaces), 1)
        self.assertEqual(all_interfaces[0]["interfaceName"], "eth0")
        self.assertEqual(all_interfaces[0]["network"], "default")

    @mock.patch("subprocess.run")
    def test_get_topology(self, mock_run, mock_logger):
        """Test get_topology returns mapped nodes."""
        scheduler = GKEScheduler(
            nodes="node1,node2", app_data=self.mock_app_data
        )

        mock_nodes_json = {
            "kind": "List",
            "items": [
                {
                    "metadata": {
                        "name": "node1",
                        "labels": {
                            "cloud.google.com/gce-topology-block": "block1"
                        },
                    }
                },
                {
                    "metadata": {
                        "name": "node2",
                        "labels": {
                            "cloud.google.com/gce-topology-block": "block2"
                        },
                    }
                },
            ],
        }
        mock_run.return_value.stdout = json.dumps(mock_nodes_json)

        topology = scheduler.get_topology()

        self.assertEqual(len(topology), 2)
        self.assertEqual(topology["block1"]["default-subblock"], ["node1"])
        self.assertEqual(topology["block2"]["default-subblock"], ["node2"])

    @mock.patch("subprocess.run")
    def test_get_topology_missing_label(self, mock_run, mock_logger):
        """Test get_topology falls back to default-block."""
        scheduler = GKEScheduler(
            nodes="node1,node2", app_data=self.mock_app_data
        )

        mock_nodes_json = {
            "kind": "List",
            "items": [
                {"metadata": {"name": "node1", "labels": {}}},
                {
                    "metadata": {
                        "name": "node2",
                        "labels": {
                            "cloud.google.com/gce-topology-block": "block2"
                        },
                    }
                },
            ],
        }
        mock_run.return_value.stdout = json.dumps(mock_nodes_json)

        topology = scheduler.get_topology()

        self.assertEqual(len(topology), 2)
        self.assertEqual(
            topology["default-block"]["default-subblock"], ["node1"]
        )
        self.assertEqual(topology["block2"]["default-subblock"], ["node2"])

    @mock.patch("chs.schedulers.gke.KueueManager")
    def test_cleanup_scheduler_resources(
        self, mock_kueue_mgr_class, mock_logger
    ):
        """Test cleanup_scheduler_resources invokes KueueManager."""
        scheduler = GKEScheduler(app_data=self.mock_app_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "configs")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "kueue-configs.yaml")
            with open(config_path, "w") as f:
                f.write("dummy")

            scheduler.workspace_config_dir = config_dir
            scheduler.kueue_supported = True

            scheduler.cleanup_scheduler_resources()

            mock_kueue_mgr_instance = mock_kueue_mgr_class.return_value
            mock_kueue_mgr_instance.cleanup.assert_called_once_with(config_path)

    @mock.patch("chs.schedulers.gke.KueueManager")
    def test_cleanup_scheduler_resources_file_backstop(
        self, mock_kueue_mgr_class, mock_logger
    ):
        """Test cleanup_scheduler_resources cleans up statelessly."""
        scheduler = GKEScheduler(app_data=self.mock_app_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "configs")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "kueue-configs.yaml")
            with open(config_path, "w") as f:
                f.write("dummy")

            self.mock_app_data.get_run.return_value = {
                "kueue_supported": False,
                "workspace_path": tmpdir,
            }
            scheduler.kueue_supported = False
            scheduler.workspace_config_dir = None

            scheduler.cleanup_scheduler_resources(workspace_name="ws1")

            mock_kueue_mgr_instance = mock_kueue_mgr_class.return_value
            mock_kueue_mgr_instance.cleanup.assert_called_once_with(config_path)

    @mock.patch("subprocess.run")
    def test_cancel_jobs_deletes_secrets(self, mock_run, mock_logger):
        """Test that _cancel_jobs deletes associated secrets."""
        scheduler = GKEScheduler(app_data=self.mock_app_data)
        mock_run.return_value.stdout = ""

        scheduler._cancel_jobs(["job1", "job2"])

        self.assertEqual(mock_run.call_count, 2)
        args = mock_run.call_args_list[1][0][0]
        self.assertIn("delete", args)
        self.assertIn("secret", args)
        self.assertIn("job1-ssh", args)
        self.assertIn("job2-ssh", args)

    @mock.patch("subprocess.run")
    def test_cleanup_scheduler_resources_deletes_secrets(
        self, mock_run, mock_logger
    ):
        """Test that cleanup_scheduler_resources deletes secrets."""
        scheduler = GKEScheduler(app_data=self.mock_app_data)

        run_entry = {
            "job_entries": {
                "job1": {"status": "Completed"},
                "job2": {"status": "Completed"},
            }
        }
        mock_run.return_value.stdout = ""

        scheduler.cleanup_scheduler_resources(run_entry=run_entry)

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertIn("delete", args)
        self.assertIn("secret", args)
        self.assertIn("job1-ssh", args)
        self.assertIn("job2-ssh", args)


if __name__ == "__main__":
    unittest.main()
