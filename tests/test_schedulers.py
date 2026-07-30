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
import yaml
from unittest import mock
from chs.schedulers.kubernetes import KubernetesScheduler, TpuSliceData
from chs.schedulers.slurm import SlurmScheduler
from chs.schedulers import detect_scheduler
from chs.exceptions import ChsSchedulerException
import click
from chs import exceptions as ex


@mock.patch("chs.schedulers.slurm.logger")
class TestSlurmScheduler(unittest.TestCase):
    """Tests for SlurmScheduler."""

    def setUp(self):
        super().setUp()
        self.mock_app_data = mock.Mock()
        self.mock_app_data.get_images_dir.return_value = "/tmp/images"

    def test_name(self, mock_logger):
        scheduler = SlurmScheduler()
        self.assertEqual(scheduler.name, "slurm")

    @mock.patch("subprocess.run")
    def test_get_nodes_under_test_nodes_arg(self, mock_run, mock_logger):
        scheduler = SlurmScheduler(
            nodes="node1,node2",
        )
        mock_run.return_value.stdout = "node1\nnode2"
        self.assertEqual(scheduler.get_nodes_under_test(), ["node1", "node2"])

    def test_get_nodes_under_test_multiple_args_error(self, mock_logger):
        scheduler = SlurmScheduler(
            nodes="n1",
            nodelists="nl1",
        )
        with self.assertRaisesRegex(click.UsageError, "Only one of"):
            scheduler.get_nodes_under_test()

    @mock.patch("subprocess.run")
    def test_get_nodes_under_test_nodelists_arg(self, mock_run, mock_logger):
        scheduler = SlurmScheduler(
            nodelists="node-[1-2]",
        )
        mock_run.return_value.stdout = "node-1\nnode-2"

        nodes = scheduler.get_nodes_under_test()

        self.assertEqual(nodes, ["node-1", "node-2"])
        self.assertEqual(mock_run.call_count, 2)
        self.assertIn("scontrol", mock_run.call_args_list[0][0][0])

    @mock.patch("subprocess.run")
    def test_get_nodes_under_test_partition_arg(self, mock_run, mock_logger):
        """Tests get_nodes_under_test with partitions argument."""
        scheduler = SlurmScheduler(
            partitions="p1",
        )
        # sinfo -N natively outputs uncompressed extended hostnames
        mock_run.return_value.stdout = "node-1\nnode-2\n"

        nodes = scheduler.get_nodes_under_test()

        self.assertEqual(nodes, ["node-1", "node-2"])
        self.assertEqual(mock_run.call_count, 2)
        self.assertIn("sinfo", mock_run.call_args_list[0][0][0])

    @mock.patch("builtins.open", new_callable=mock.mock_open, read_data="12345")
    def test_extract_job_id_success(self, mock_open, _mock_logger):
        scheduler = SlurmScheduler()
        job_id = scheduler.extract_job_id("/tmp/exp")
        self.assertEqual(job_id, "12345")
        mock_open.assert_called_once_with("/tmp/exp/.slurm_job", "r")

    @mock.patch(
        "builtins.open", new_callable=mock.mock_open, read_data="not_a_digit"
    )
    def test_extract_job_id_not_digit(self, mock_open, _mock_logger):
        scheduler = SlurmScheduler()
        with self.assertRaises(ChsSchedulerException):
            scheduler.extract_job_id("/tmp/exp")

    @mock.patch("builtins.open", side_effect=IOError("Permission denied"))
    def test_extract_job_id_io_error(self, mock_open, _mock_logger):
        scheduler = SlurmScheduler()
        with self.assertRaises(ChsSchedulerException):
            scheduler.extract_job_id("/tmp/exp")

    @mock.patch("subprocess.run")
    def test_has_no_failed_jobs_true(self, mock_run, mock_logger):
        scheduler = SlurmScheduler()
        mock_run.return_value.stdout = "chs-123|COMPLETED"
        self.assertTrue(scheduler.has_no_failed_jobs())

    @mock.patch("subprocess.run")
    def test_has_no_failed_jobs_false(self, mock_run, mock_logger):
        scheduler = SlurmScheduler()
        mock_run.return_value.stdout = "chs-123|FAILED"
        self.assertFalse(scheduler.has_no_failed_jobs())

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
        scheduler = SlurmScheduler()
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
        scheduler = SlurmScheduler()
        with mock.patch.object(scheduler, "_run_srun") as mock_srun:
            mock_srun.side_effect = [
                "",  # No default route
                "192.168.0.0/24 dev eth0 proto kernel\n"
                "172.17.0.0/16 dev docker0",
            ]
            interfaces = scheduler._discover_network_interfaces()
            self.assertEqual(interfaces, [])
            mock_logger.debug.assert_called_with(
                "Detected network interfaces: %r", []
            )

    @mock.patch("google.auth.default")
    def test_resolve_project_id_slurm_success(
        self, mock_auth_default, mock_logger
    ):
        scheduler = SlurmScheduler()
        mock_auth_default.return_value = (None, "my-project-id")
        project_id = scheduler._resolve_project_id("node1")
        self.assertEqual(project_id, "my-project-id")

    @mock.patch("google.auth.default")
    def test_resolve_project_id_slurm_no_project(
        self, mock_auth_default, mock_logger
    ):
        scheduler = SlurmScheduler()
        mock_auth_default.return_value = (None, None)
        with self.assertRaisesRegex(
            ex.ChsSchedulerException, "Could not determine GCP project"
        ):
            scheduler._resolve_project_id("node1")

    @mock.patch("google.auth.default")
    def test_resolve_project_id_slurm_error(
        self, mock_auth_default, mock_logger
    ):
        scheduler = SlurmScheduler()
        mock_auth_default.side_effect = Exception("auth error")
        with self.assertRaisesRegex(
            ex.ChsSchedulerException, "Failed to get GCP credentials"
        ):
            scheduler._resolve_project_id("node1")


@mock.patch("chs.schedulers.kubernetes.logger")
class TestKubernetesScheduler(unittest.TestCase):
    """Tests for KubernetesScheduler."""

    def setUp(self):
        super().setUp()
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = self.tmp_dir_obj.name
        self.addCleanup(self.tmp_dir_obj.cleanup)
        self.mock_app_data = mock.Mock()
        self.mock_app_data.get_run.return_value = {}

    def test_init(self, mock_logger):
        """Test initialization."""
        scheduler = KubernetesScheduler(
            nodes="node1,node2",
        )
        self.assertEqual(scheduler.name, "kubernetes")
        self.assertEqual(scheduler.nodes_arg, "node1,node2")
        self.assertIsNone(scheduler.nodepools_arg)

    def test_init_nodepool(self, mock_logger):
        """Test initialization with nodepools."""
        scheduler = KubernetesScheduler(
            nodepools="pool-1",
        )
        self.assertEqual(scheduler.name, "kubernetes")
        self.assertIsNone(scheduler.nodes_arg)
        self.assertEqual(scheduler.nodepools_arg, "pool-1")

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_busy_nodes",
        return_value=set(),
    )
    @mock.patch("chs.schedulers.kubernetes.KubernetesScheduler._get_tpu_slices")
    def test_get_nodes_under_test(
        self, mock_tpu_slices, mock_busy, mock_logger
    ):
        """Test get_nodes_under_test returns parsed nodes."""
        mock_tpu_slices.return_value = TpuSliceData(
            pools={}, node_to_pool={}, pool_to_topology={}
        )
        scheduler = KubernetesScheduler(
            nodes="node1, node2 ",
        )
        self.assertCountEqual(
            scheduler.get_nodes_under_test(), ["node1", "node2"]
        )

    @mock.patch("subprocess.run")
    def test_resolve_raw_nodes_empty(self, mock_run, mock_logger):
        """Test resolve_raw_nodes returns empty list if no nodes."""
        scheduler = KubernetesScheduler()
        mock_run.side_effect = subprocess.CalledProcessError(1, "kubectl")

        self.assertEqual(scheduler.resolve_raw_nodes(), [])

    @mock.patch("subprocess.run")
    def test_resolve_raw_nodes_nodepool(self, mock_run, mock_logger):
        """Test resolve_raw_nodes uses nodepools to fetch nodes."""
        scheduler = KubernetesScheduler(
            nodepools="pool-1",
        )
        mock_run.return_value.stdout = "node-1 node-2"

        nodes = scheduler.resolve_raw_nodes()

        self.assertEqual(nodes, ["node-1", "node-2"])
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertIn("kubectl", cmd)
        self.assertIn("-l", cmd)
        self.assertIn("cloud.google.com/gke-nodepool in (pool-1)", cmd)

    @mock.patch("subprocess.run")
    def test_resolve_raw_nodes_precedence(self, mock_run, mock_logger):
        """Test nodes arg takes precedence over nodepools."""
        scheduler = KubernetesScheduler(
            nodes="node-1",
            nodepools="pool-1",
        )
        nodes = scheduler.resolve_raw_nodes()
        self.assertEqual(nodes, ["node-1"])
        mock_run.assert_not_called()

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler.get_nodes_under_test",
        return_value=["node1"],
    )
    def test_write_ramble_config(self, mock_filter, mock_logger):
        """Test write_ramble_config generates correct YAML."""
        scheduler = KubernetesScheduler(
            nodes="node1",
        )
        workspace_config_dir = os.path.join(self.tmp_dir, "configs")
        os.makedirs(workspace_config_dir, exist_ok=True)

        mock_template_content = """
ramble:
  variants:
    workflow_manager: kubernetes-jobset
  variables:
    nodes: {nodes}
    machine_type: {machine_type}
    tests: {tests}
    default_interface_name: {default_interface_name}
    all_interfaces_json: {all_interfaces_json}
"""
        with mock.patch.object(
            KubernetesScheduler,
            "_discover_networks",
            return_value=(["default", "gvnic-1"], ["gpu-net-1"]),
        ):
            with mock.patch("chs.schedulers.kubernetes.pkg_resources.as_file"):
                with mock.patch(
                    "builtins.open",
                    mock.mock_open(read_data=mock_template_content),
                ):
                    scheduler.write_ramble_config(
                        workspace_config_dir, ["test1"], "a3-megagpu-8g"
                    )

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler.get_nodes_under_test",
        return_value=["node1"],
    )
    def test_write_ramble_config_integration(self, mock_filter, mock_logger):
        """Integration-like test for write_ramble_config using real template."""
        scheduler = KubernetesScheduler(
            nodes="node1",
        )
        workspace_config_dir = os.path.join(self.tmp_dir, "configs")
        os.makedirs(workspace_config_dir, exist_ok=True)
        template_path = os.path.join(self.tmp_dir, "kubernetes_template.yaml")
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
            KubernetesScheduler,
            "_discover_networks",
            return_value=(["default"], ["rdma-0"]),
        ):
            with mock.patch(
                "chs.schedulers.kubernetes.pkg_resources.as_file",
                side_effect=mock_as_file_func,
            ):
                scheduler.write_ramble_config(
                    workspace_config_dir, ["test1"], "a3-megagpu-8g"
                )

        output_path = os.path.join(workspace_config_dir, "ramble.yaml")
        import yaml

        with open(output_path, "r") as f:
            data = yaml.safe_load(f)

        variables = data["ramble"]["variables"]
        self.assertEqual(variables["nodes"], ["node1"])
        self.assertEqual(variables["machine_type"], "a3-megagpu-8g")
        self.assertEqual(variables["tests"], ["test1"])
        self.assertEqual(variables["default_interface_name"], "eth0")

        import json

        all_interfaces = json.loads(variables["all_interfaces_json"].strip())
        self.assertEqual(len(all_interfaces), 2)
        self.assertEqual(all_interfaces[0]["interfaceName"], "eth0")
        self.assertEqual(all_interfaces[0]["network"], "default")

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler.get_nodes_under_test",
        return_value=["node1", "node2"],
    )
    def test_write_ramble_config_with_deadline(self, mock_filter, mock_logger):
        """Test write_ramble_config injects timeout and deadline."""
        scheduler = KubernetesScheduler(
            nodes="node1",
        )
        workspace_config_dir = os.path.join(self.tmp_dir, "configs")
        os.makedirs(workspace_config_dir, exist_ok=True)
        template_path = os.path.join(self.tmp_dir, "kubernetes_template.yaml")
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

    deadline: {deadline}
""")

        @contextlib.contextmanager
        def mock_as_file_func(path):
            yield template_path

        with mock.patch.object(
            KubernetesScheduler,
            "_discover_networks",
            return_value=(["default"], ["rdma-0"]),
        ):
            with mock.patch(
                "chs.schedulers.kubernetes.pkg_resources.as_file",
                side_effect=mock_as_file_func,
            ):
                scheduler.write_ramble_config(
                    workspace_config_dir,
                    ["test1"],
                    "a3-megagpu-8g",
                    deadline=1800,
                )

        output_path = os.path.join(workspace_config_dir, "ramble.yaml")
        import yaml

        with open(output_path, "r") as f:
            data = yaml.safe_load(f)

        variables = data["ramble"]["variables"]
        self.assertEqual(variables["deadline"], 1800)

    @mock.patch("subprocess.run")
    @mock.patch("chs.schedulers.kubernetes.KueueManager")
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler.get_nodes_under_test",
        return_value=["node1"],
    )
    def test_setup_scheduler_resources_kueue_success(
        self, mock_validate, mock_kueue_mgr_class, mock_run, mock_logger
    ):
        """Test setup_scheduler_resources caches labels on success."""
        scheduler = KubernetesScheduler(
            nodes="node1",
        )
        scheduler.kueue_supported = True
        self.mock_app_data.get_run.return_value = {"kueue_supported": True}

        mock_kueue_mgr_instance = mock_kueue_mgr_class.return_value
        mock_kueue_mgr_instance.generate_configs.return_value = (
            "kueue.x-k8s.io/queue-name: chs-local-queue"
        )

        workspace_config_dir = os.path.join(self.tmp_dir, "configs")

        tests = scheduler.setup_scheduler_resources(
            workspace_config_dir,
            [{"name": "nccl-cluster"}],
            "a3-megagpu-8g",
            workspace_name="ws1",
        )

        self.assertEqual(tests, [{"name": "nccl-cluster"}])
        self.assertEqual(
            scheduler._jobset_labels_yaml,
            "kueue.x-k8s.io/queue-name: chs-local-queue",
        )
        self.assertEqual(scheduler._suspend_jobset, "true")

    @mock.patch("subprocess.run")
    @mock.patch("chs.schedulers.kubernetes.KueueManager")
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler.get_nodes_under_test",
        return_value=["node1"],
    )
    def test_setup_scheduler_resources_kueue_fallback(
        self, mock_validate, mock_kueue_mgr_class, mock_run, mock_logger
    ):
        """Test setup_scheduler_resources falls back if Kueue setup fails."""
        from chs.exceptions import KueueException

        scheduler = KubernetesScheduler(
            nodes="node1",
        )
        scheduler.kueue_supported = True
        self.mock_app_data.get_run.return_value = {"kueue_supported": True}

        mock_kueue_mgr_instance = mock_kueue_mgr_class.return_value
        mock_kueue_mgr_instance.generate_configs.side_effect = KueueException(
            "setup failed"
        )

        workspace_config_dir = os.path.join(self.tmp_dir, "configs")

        tests = scheduler.setup_scheduler_resources(
            workspace_config_dir,
            [{"name": "nccl-cluster"}, {"name": "test1"}],
            "a3-megagpu-8g",
            workspace_name="ws1",
        )

        self.assertEqual(tests, [{"name": "test1"}])
        self.assertFalse(scheduler.kueue_supported)
        self.assertEqual(scheduler._jobset_labels_yaml, "")
        self.assertEqual(scheduler._suspend_jobset, "false")

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler.get_nodes_under_test",
        return_value=["node1", "node2"],
    )
    def test_write_ramble_config_uses_cached_labels(
        self, mock_filter, mock_logger
    ):
        """Test write_ramble_config writes cached properties."""
        scheduler = KubernetesScheduler(
            nodes="node1",
        )
        scheduler._jobset_labels_yaml = "kueue/queue: local-q"
        scheduler._suspend_jobset = "true"

        workspace_config_dir = os.path.join(self.tmp_dir, "configs")
        os.makedirs(workspace_config_dir, exist_ok=True)
        template_path = os.path.join(self.tmp_dir, "kubernetes_template.yaml")
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
            KubernetesScheduler,
            "_discover_networks",
            return_value=(["default"], ["rdma-0"]),
        ):
            with mock.patch(
                "chs.schedulers.kubernetes.pkg_resources.as_file",
                side_effect=mock_as_file_func,
            ):
                scheduler.write_ramble_config(
                    workspace_config_dir,
                    [{"name": "test1"}],
                    "a3-megagpu-8g",
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
        scheduler = KubernetesScheduler()
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
        scheduler = KubernetesScheduler()
        mock_run.side_effect = subprocess.CalledProcessError(1, "kubectl")

        networks = scheduler._discover_networks()

        self.assertEqual(networks, ([], []))

    @mock.patch("subprocess.run")
    def test_discover_networks_no_default(self, mock_run, mock_logger):
        """Test _discover_networks fails if default L3 network is missing."""
        scheduler = KubernetesScheduler()
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
        scheduler = KubernetesScheduler()
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
        scheduler = KubernetesScheduler(
            nodes="node1",
        )
        workspace_config_dir = os.path.join(self.tmp_dir, "configs")
        os.makedirs(workspace_config_dir, exist_ok=True)
        template_path = os.path.join(self.tmp_dir, "kubernetes_template.yaml")
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
            KubernetesScheduler, "_discover_networks", return_value=([], [])
        ):
            with mock.patch(
                "chs.schedulers.kubernetes.pkg_resources.as_file",
                side_effect=mock_as_file_func,
            ):
                scheduler.write_ramble_config(
                    workspace_config_dir, ["test1"], "a3-megagpu-8g"
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
        scheduler = KubernetesScheduler(
            nodes="node1,node2",
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
        scheduler = KubernetesScheduler(
            nodes="node1,node2",
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

    @mock.patch("subprocess.run")
    @mock.patch("chs.schedulers.kubernetes.KueueManager")
    def test_cleanup_scheduler_resources(
        self, mock_kueue_mgr_class, mock_run, mock_logger
    ):
        """Test cleanup_scheduler_resources invokes KueueManager."""
        scheduler = KubernetesScheduler()
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
            mock_kueue_mgr_instance.cleanup.assert_called_once_with()

    @mock.patch("subprocess.run")
    @mock.patch("chs.schedulers.kubernetes.KueueManager")
    def test_cleanup_scheduler_resources_file_backstop(
        self, mock_kueue_mgr_class, mock_run, mock_logger
    ):
        """Test cleanup_scheduler_resources cleans up statelessly."""
        scheduler = KubernetesScheduler()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "configs")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "kueue-configs.yaml")
            with open(config_path, "w") as f:
                f.write("dummy")

            scheduler.kueue_supported = False
            scheduler.workspace_config_dir = None

            scheduler.cleanup_scheduler_resources()

            mock_kueue_mgr_instance = mock_kueue_mgr_class.return_value
            mock_kueue_mgr_instance.cleanup.assert_called_once_with()

    @mock.patch("subprocess.run")
    def test_cancel_jobs_deletes_secrets(self, mock_run, mock_logger):
        """Test that _cancel_jobs deletes associated secrets."""
        scheduler = KubernetesScheduler()
        mock_run.return_value.stdout = "chs-job1-ssh chs-job2-ssh other-secret"

        scheduler.cancel_jobs(["job1", "job2"])

        args = mock_run.call_args_list[0][0][0]
        self.assertIn("delete", args)
        self.assertIn("jobset", args)

        args_secrets = mock_run.call_args_list[2][0][0]
        self.assertIn("delete", args_secrets)
        self.assertIn("secrets", args_secrets)
        self.assertIn("chs-job1-ssh", args_secrets)
        self.assertNotIn("other-secret", args_secrets)

    @mock.patch("subprocess.run")
    def test_cleanup_scheduler_resources_deletes_secrets(
        self, mock_run, mock_logger
    ):
        """Test that cleanup_scheduler_resources deletes secrets."""
        scheduler = KubernetesScheduler()

        mock_run.return_value.stdout = "chs-job1-ssh"
        scheduler.cleanup_scheduler_resources()

        args = mock_run.call_args_list[6][0][0]
        self.assertIn("delete", args)
        self.assertIn("secrets", args)
        self.assertIn("chs-job1-ssh", args)

    @mock.patch("subprocess.run")
    def test_resolve_project_id_kubernetes_success(self, mock_run, mock_logger):
        scheduler = KubernetesScheduler()
        mock_run.return_value.stdout = "gce://my-project-id/us-central1-a/node1"
        project_id = scheduler._resolve_project_id("node1")
        self.assertEqual(project_id, "my-project-id")
        mock_run.assert_called_once_with(
            [
                "kubectl",
                "get",
                "node",
                "node1",
                "-o",
                "jsonpath={.spec.providerID}",
            ],
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        )

    @mock.patch("subprocess.run")
    def test_resolve_project_id_kubernetes_invalid_format(
        self, mock_run, mock_logger
    ):
        scheduler = KubernetesScheduler()
        invalid_formats = [
            "invalid-format",
            "gce://my-project-id",
            "gce://my-project-id/us-central1-a",
            "gce://my-project-id/us-central1-a/node1/extra",
            "gce://my-project-id//node1",
            "gke://my-project-id/us-central1-a/node1",
        ]
        for fmt in invalid_formats:
            mock_run.return_value.stdout = fmt
            with self.assertRaisesRegex(
                ex.ChsSchedulerException, "Unexpected providerID format"
            ):
                scheduler._resolve_project_id("node1")

    @mock.patch("subprocess.run")
    def test_resolve_project_id_kubernetes_error(self, mock_run, mock_logger):
        scheduler = KubernetesScheduler()
        mock_run.side_effect = subprocess.CalledProcessError(1, "kubectl")
        with self.assertRaisesRegex(
            ex.ChsSchedulerException, "Failed to resolve project ID"
        ):
            scheduler._resolve_project_id("node1")

    @mock.patch("chs.schedulers.base.query_gce_instances_info")
    def test_detect_nodes_hardware_success(self, mock_query_info, mock_logger):
        scheduler = KubernetesScheduler()
        with mock.patch.object(
            scheduler, "_resolve_project_id", return_value="my-project"
        ):
            mock_query_info.return_value = {
                "node1": {
                    "machine_type": "a3-megagpu-8g",
                    "instance_id": "12345",
                },
                "node2": {
                    "machine_type": "a3-megagpu-8g",
                    "instance_id": "67890",
                },
            }
            machine_types = scheduler.detect_nodes_hardware(["node1", "node2"])
            self.assertEqual(machine_types, ["a3-megagpu-8g"])
            self.assertEqual(
                scheduler.node_to_instance_id,
                {"node1": "12345", "node2": "67890"},
            )

    @mock.patch("chs.schedulers.base.query_gce_instances_info")
    def test_detect_nodes_hardware_heterogeneous(
        self, mock_query_info, mock_logger
    ):
        scheduler = KubernetesScheduler()
        with mock.patch.object(
            scheduler, "_resolve_project_id", return_value="my-project"
        ):
            mock_query_info.return_value = {
                "node1": {
                    "machine_type": "a3-megagpu-8g",
                    "instance_id": "12345",
                },
                "node2": {
                    "machine_type": "a3-ultragpu-8g",
                    "instance_id": "67890",
                },
            }
            machine_types = scheduler.detect_nodes_hardware(["node1", "node2"])
            self.assertEqual(machine_types, ["a3-megagpu-8g", "a3-ultragpu-8g"])

    @mock.patch("chs.schedulers.base.query_gce_instances_info")
    def test_detect_nodes_hardware_unknown(self, mock_query_info, mock_logger):
        scheduler = KubernetesScheduler()
        with mock.patch.object(
            scheduler, "_resolve_project_id", return_value="my-project"
        ):
            mock_query_info.return_value = {
                "node1": {
                    "machine_type": "a3-megagpu-8g",
                    "instance_id": "12345",
                },
            }
            machine_types = scheduler.detect_nodes_hardware(["node1", "node2"])
            self.assertEqual(machine_types, ["a3-megagpu-8g", "unknown"])
            self.assertEqual(scheduler.node_to_instance_id, {"node1": "12345"})

    @mock.patch(
        "builtins.open",
        new_callable=mock.mock_open,
        read_data=(
            "apiVersion: jobset.x-k8s.io/v1alpha2\n"
            "kind: JobSet\n"
            "metadata:\n"
            "  name: my-jobset"
        ),
    )
    def test_extract_job_id_success(self, mock_open, _mock_logger):
        scheduler = KubernetesScheduler()
        job_id = scheduler.extract_job_id("/tmp/exp")
        self.assertEqual(job_id, "my-jobset")
        mock_open.assert_called_once_with("/tmp/exp/jobset.yaml", "r")

    @mock.patch(
        "builtins.open",
        new_callable=mock.mock_open,
        read_data=(
            "apiVersion: v1\n"
            "kind: Service\n"
            "metadata:\n"
            "  name: my-service"
        ),
    )
    def test_extract_job_id_no_jobset(self, mock_open, _mock_logger):
        scheduler = KubernetesScheduler()
        with self.assertRaises(ChsSchedulerException):
            scheduler.extract_job_id("/tmp/exp")

    @mock.patch(
        "builtins.open",
        new_callable=mock.mock_open,
        read_data="invalid yaml: : :",
    )
    def test_extract_job_id_yaml_error(self, mock_open, _mock_logger):
        scheduler = KubernetesScheduler()
        with mock.patch(
            "yaml.safe_load_all", side_effect=yaml.YAMLError("parse error")
        ):
            with self.assertRaises(ChsSchedulerException):
                scheduler.extract_job_id("/tmp/exp")

    @mock.patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_extract_job_id_io_error(self, mock_open, _mock_logger):
        scheduler = KubernetesScheduler()
        with self.assertRaises(ChsSchedulerException):
            scheduler.extract_job_id("/tmp/exp")

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler.get_nodes_under_test",
        return_value=["node1", "node2"],
    )
    def test_get_nodes_under_test_validates_args(
        self, mock_filter, _mock_logger
    ):
        scheduler = KubernetesScheduler(nodes="node1,node2")
        nodes = scheduler.get_nodes_under_test()
        self.assertEqual(nodes, ["node1", "node2"])

    @mock.patch("chs.schedulers.kubernetes.KubernetesScheduler._run_kubectl")
    def test_cleanup_dynamic_slices(self, mock_kubectl, mock_logger):
        mock_kubectl.return_value = "slices.accelerator.gke.io"
        scheduler = KubernetesScheduler()
        scheduler._cleanup_dynamic_slices()
        mock_kubectl.assert_has_calls(
            [
                mock.call(
                    [
                        "get",
                        "crd",
                        "slices.accelerator.gke.io",
                        "--ignore-not-found",
                    ]
                ),
                mock.call(
                    [
                        "delete",
                        "slices.accelerator.gke.io",
                        "-A",
                        "-l",
                        "created-by=chs",
                        "--ignore-not-found",
                    ]
                ),
            ]
        )

    @mock.patch("chs.schedulers.kubernetes.KubernetesScheduler._run_kubectl")
    def test_cleanup_dynamic_slices_crd_not_found(
        self,
        mock_kubectl,
        mock_logger,
    ):
        mock_kubectl.return_value = ""
        scheduler = KubernetesScheduler()
        scheduler._cleanup_dynamic_slices()
        mock_kubectl.assert_called_once_with(
            [
                "get",
                "crd",
                "slices.accelerator.gke.io",
                "--ignore-not-found",
            ]
        )

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._delete_namespace"
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler."
        "_cleanup_dynamic_slices"
    )
    @mock.patch("chs.schedulers.kubernetes.KubernetesScheduler._run_kubectl")
    def test_cancel_jobs_by_prefix_does_not_delete_namespace(
        self,
        mock_run_kubectl,
        mock_cleanup_slices,
        mock_delete_ns,
        mock_logger,
    ):
        mock_run_kubectl.return_value = (
            "jobset/chs-test-job1\njobset/chs-test-job2"
        )
        scheduler = KubernetesScheduler()
        with mock.patch.object(scheduler, "cancel_jobs") as mock_cancel_jobs:
            scheduler.cancel_jobs_by_prefix("chs-test")
            mock_cancel_jobs.assert_called_once_with(
                ["chs-test-job1", "chs-test-job2"]
            )
            mock_delete_ns.assert_not_called()

    @mock.patch("chs.schedulers.kubernetes.KubernetesScheduler._run_kubectl")
    @mock.patch("time.time", side_effect=[0, 1])
    def test_wait_for_slices_active_success(
        self, mock_time, mock_kubectl, mock_logger
    ):
        scheduler = KubernetesScheduler()
        mock_kubectl.return_value = (
            '{"items": [{"status": {"conditions": '
            '[{"type": "Ready", "reason": "ACTIVE"}]}}]}'
        )
        scheduler._wait_for_slices_active()
        mock_kubectl.assert_called_once()

    @mock.patch("chs.schedulers.kubernetes.KubernetesScheduler._run_kubectl")
    @mock.patch("time.time", side_effect=[0, 1])
    def test_wait_for_slices_active_failed(
        self, mock_time, mock_kubectl, mock_logger
    ):
        scheduler = KubernetesScheduler()
        mock_kubectl.return_value = (
            '{"items": [{"status": {"conditions": '
            '[{"type": "Ready", "reason": "FAILED"}]}}]}'
        )
        with self.assertRaises(ChsSchedulerException):
            scheduler._wait_for_slices_active()


class TestSchedulerDetection(unittest.TestCase):
    """Tests for detect_scheduler function."""

    @mock.patch("shutil.which")
    def test_detect_slurm_success(self, mock_which):
        """Test that Slurm is detected if sinfo/squeue exist."""
        mock_which.side_effect = lambda cmd: cmd in ["sinfo", "squeue"]

        scheduler = detect_scheduler()
        self.assertEqual(scheduler, "slurm")

    @mock.patch("shutil.which")
    @mock.patch("os.path.exists")
    def test_detect_kubernetes_in_cluster(self, mock_exists, mock_which):
        """Test K8s detected in-cluster if token exists (and Slurm not
        found).
        """
        # Slurm not found
        mock_which.return_value = False

        # In-cluster token exists
        mock_exists.side_effect = (
            lambda p: p == "/var/run/secrets/kubernetes.io/serviceaccount/token"
        )

        scheduler = detect_scheduler()
        self.assertEqual(scheduler, "kubernetes")

    @mock.patch("subprocess.run")
    @mock.patch("shutil.which")
    @mock.patch("os.path.exists")
    @mock.patch.dict("os.environ", {}, clear=True)
    def test_detect_kubernetes_out_of_cluster(
        self, mock_exists, mock_which, mock_run
    ):
        """Test K8s detected out-of-cluster if kubeconfig exists."""
        # Mock subprocess.run to return a context
        mock_run.return_value.stdout = "test-context"
        # Slurm not found, kubectl found
        mock_which.side_effect = lambda cmd: cmd == "kubectl"

        # kubeconfig exists
        mock_exists.side_effect = lambda p: p == os.path.expanduser(
            "~/.kube/config"
        )

        scheduler = detect_scheduler()
        self.assertEqual(scheduler, "kubernetes")

    @mock.patch("shutil.which")
    @mock.patch("os.path.exists")
    @mock.patch.dict("os.environ", {}, clear=True)
    def test_detect_kubernetes_missing_kubectl(self, mock_exists, mock_which):
        """Test K8s detection fails fast if kubectl is missing."""
        mock_exists.side_effect = lambda p: p == os.path.expanduser(
            "~/.kube/config"
        )
        mock_which.return_value = False

        with self.assertRaisesRegex(
            ChsSchedulerException, "Missing required executable: kubectl"
        ):
            detect_scheduler()

    @mock.patch("subprocess.run")
    @mock.patch("shutil.which")
    @mock.patch("os.path.exists")
    @mock.patch.dict("os.environ", {}, clear=True)
    def test_detect_kubernetes_missing_context(
        self, mock_exists, mock_which, mock_run
    ):
        """Test K8s detection fails if no active context is set."""
        mock_exists.side_effect = lambda p: p == os.path.expanduser(
            "~/.kube/config"
        )
        mock_which.side_effect = lambda cmd: cmd == "kubectl"
        mock_run.return_value.stdout = ""

        with self.assertRaisesRegex(
            ChsSchedulerException, "no active kubectl context is set"
        ):
            detect_scheduler()

    @mock.patch("subprocess.run")
    @mock.patch("shutil.which")
    @mock.patch("os.path.exists")
    @mock.patch.dict("os.environ", {"KUBECONFIG": "/some/path"})
    def test_detect_kubernetes_out_of_cluster_env(
        self, mock_exists, mock_which, mock_run
    ):
        """Test K8s detected out-of-cluster if KUBECONFIG env is set and
        file exists.
        """
        # Mock subprocess.run to return a context
        mock_run.return_value.stdout = "test-context"
        # Slurm not found, kubectl found
        mock_which.side_effect = lambda cmd: cmd == "kubectl"
        mock_exists.side_effect = lambda p: p == "/some/path"

        scheduler = detect_scheduler()
        self.assertEqual(scheduler, "kubernetes")

    @mock.patch("subprocess.run")
    @mock.patch("shutil.which")
    @mock.patch("os.path.exists")
    @mock.patch.dict("os.environ", {"KUBECONFIG": "/path1:/path2"})
    def test_detect_kubernetes_out_of_cluster_env_multi(
        self, mock_exists, mock_which, mock_run
    ):
        """Test K8s detected if KUBECONFIG has multiple paths and one exists."""
        # Mock subprocess.run to return a context
        mock_run.return_value.stdout = "test-context"
        # Slurm not found, kubectl found
        mock_which.side_effect = lambda cmd: cmd == "kubectl"
        # /path2 exists, /path1 does not
        mock_exists.side_effect = lambda p: p == "/path2"

        scheduler = detect_scheduler()
        self.assertEqual(scheduler, "kubernetes")

    @mock.patch("shutil.which")
    @mock.patch("os.path.exists")
    @mock.patch.dict("os.environ", {"KUBECONFIG": "/path1:/path2"})
    def test_detect_kubernetes_out_of_cluster_env_multi_fail(
        self, mock_exists, mock_which
    ):
        """Test K8s not detected if KUBECONFIG has paths but none exist."""
        # Slurm not found, kubectl found
        mock_which.side_effect = lambda cmd: cmd == "kubectl"
        mock_exists.return_value = False

        with self.assertRaises(ChsSchedulerException):
            detect_scheduler()

    @mock.patch("shutil.which")
    @mock.patch("os.path.exists")
    def test_detect_scheduler_fail(self, mock_exists, mock_which):
        """Test that ChsSchedulerException is raised if all checks fail."""
        mock_which.return_value = False
        mock_exists.return_value = False

        with self.assertRaises(ChsSchedulerException):
            detect_scheduler()


if __name__ == "__main__":
    unittest.main()
