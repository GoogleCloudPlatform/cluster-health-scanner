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

"""Unit tests for KueueManager."""

import os
import subprocess
import tempfile
import unittest
from unittest import mock

import yaml

from chs.constants import SUPPORTED_DYNAMIC_SLICE_TOPOLOGY
from chs.exceptions import KueueException
from chs.utils.kueue import KueueManager


class TestKueueManager(unittest.TestCase):
    """Tests for KueueManager."""

    def setUp(self):
        super().setUp()
        self.mock_scheduler = mock.Mock()
        self.mock_scheduler.kueue_supported = False
        self.mock_scheduler.workspace_config_dir = None
        self.kueue_mgr = KueueManager(self.mock_scheduler)
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = self.tmp_dir_obj.name
        self.addCleanup(self.tmp_dir_obj.cleanup)

    def test_generate_configs_unsupported_machine_type_raises(self):
        """Test generate_configs with unsupported machine type raises
        KueueException."""
        with self.assertRaisesRegex(
            KueueException, "not recognized in MACHINE_TYPE_RESOURCE_MAP"
        ):
            self.kueue_mgr.generate_configs(
                self.tmp_dir, "unsupported-mt", ["node1"]
            )

    def test_generate_configs_success_renders(self):
        """Test generate_configs renders template and returns correct label."""
        self.mock_scheduler._run_kubectl.return_value = "v1beta2"
        label = self.kueue_mgr.generate_configs(
            self.tmp_dir, "a3-megagpu-8g", ["node1", "node2"]
        )
        self.assertEqual(label, "kueue.x-k8s.io/queue-name: chs-local-queue")

        config_path = os.path.join(self.tmp_dir, "kueue-configs.yaml")
        self.assertTrue(os.path.exists(config_path))
        with open(config_path, "r") as f:
            docs = list(yaml.safe_load_all(f))
            self.assertTrue(
                any(
                    d.get("apiVersion") == "kueue.x-k8s.io/v1beta2"
                    for d in docs
                )
            )
            cq = next(d for d in docs if d["kind"] == "ClusterQueue")
            flavor_res = cq["spec"]["resourceGroups"][0]["flavors"][0][
                "resources"
            ][0]
            self.assertEqual(flavor_res["name"], "nvidia.com/gpu")
            self.assertEqual(flavor_res["nominalQuota"], "16")

    def test_generate_configs_tpu_success_renders(self):
        """Test generate_configs for TPU renders correct flavor and
        resource name."""
        self.mock_scheduler._run_kubectl.return_value = "v1beta1"
        label = self.kueue_mgr.generate_configs(
            self.tmp_dir, "tpu7x-standard-4t", ["node1", "node2"]
        )
        self.assertEqual(label, "kueue.x-k8s.io/queue-name: chs-local-queue")

        config_path = os.path.join(self.tmp_dir, "kueue-configs.yaml")
        self.assertTrue(os.path.exists(config_path))
        with open(config_path, "r") as f:
            docs = list(yaml.safe_load_all(f))
            self.assertTrue(
                any(
                    d.get("apiVersion") == "kueue.x-k8s.io/v1beta1"
                    for d in docs
                )
            )
            cq = next(d for d in docs if d["kind"] == "ClusterQueue")
            flavor_res = cq["spec"]["resourceGroups"][0]["flavors"][0][
                "resources"
            ][0]
            self.assertEqual(flavor_res["name"], "google.com/tpu")
            self.assertEqual(flavor_res["nominalQuota"], "8")

    def _render_all(
        self,
        machine_type: str,
        dynamic_slicing: bool = False,
    ) -> list[dict]:
        """Renders configs and returns all parsed YAML documents."""
        self.mock_scheduler._run_kubectl.return_value = "v1beta1"
        self.kueue_mgr.generate_configs(
            self.tmp_dir,
            machine_type,
            ["node1", "node2"],
            dynamic_slicing=dynamic_slicing,
        )
        config_path = os.path.join(self.tmp_dir, "kueue-configs.yaml")
        with open(config_path, "r") as f:
            return [d for d in yaml.safe_load_all(f) if d]

    def test_generate_configs_flat_topology_omits_slice_levels(self):
        """Non-dynamic clusters get a hostname-only topology.

        TAS requires every node in the flavor to carry a label for every
        level, so GPU and statically sliced TPU nodes must not be given the
        slice controller's block or sub-block levels.
        """
        docs = self._render_all("a3-megagpu-8g", dynamic_slicing=False)
        topologies = [d for d in docs if d["kind"] == "Topology"]
        flavors = [d for d in docs if d["kind"] == "ResourceFlavor"]

        self.assertEqual(len(topologies), 1)
        self.assertEqual(
            topologies[0]["metadata"]["name"], "chs-node-topology-static"
        )
        self.assertEqual(
            [level["nodeLabel"] for level in topologies[0]["spec"]["levels"]],
            ["kubernetes.io/hostname"],
        )
        self.assertEqual(len(flavors), 1)
        self.assertEqual(
            flavors[0]["metadata"]["name"],
            "chs-flavor-static",
        )
        self.assertEqual(
            flavors[0]["spec"]["topologyName"],
            "chs-node-topology-static",
        )

    def test_generate_configs_dynamic_topology_adds_slice_levels(self):
        """Dynamic slicing creates dynamic and static topologies and flavors."""
        docs = self._render_all("tpu7x-standard-4t", dynamic_slicing=True)
        topologies = {
            d["metadata"]["name"]: d for d in docs if d["kind"] == "Topology"
        }
        flavors = {
            d["metadata"]["name"]: d
            for d in docs
            if d["kind"] == "ResourceFlavor"
        }
        cluster_queue = next(d for d in docs if d["kind"] == "ClusterQueue")

        self.assertIn("chs-node-topology-dynamic", topologies)
        self.assertIn("chs-node-topology-static", topologies)

        self.assertEqual(
            [
                level["nodeLabel"]
                for level in topologies["chs-node-topology-dynamic"]["spec"][
                    "levels"
                ]
            ],
            [
                "cloud.google.com/gce-topology-block",
                f"cloud.google.com/gke-tpu-partition-{SUPPORTED_DYNAMIC_SLICE_TOPOLOGY}-id",
                "kubernetes.io/hostname",
            ],
        )
        self.assertEqual(
            [
                level["nodeLabel"]
                for level in topologies["chs-node-topology-static"]["spec"][
                    "levels"
                ]
            ],
            ["kubernetes.io/hostname"],
        )

        self.assertIn("chs-flavor-dynamic", flavors)
        self.assertIn("chs-flavor-static", flavors)
        self.assertEqual(
            flavors["chs-flavor-dynamic"]["spec"]["topologyName"],
            "chs-node-topology-dynamic",
        )
        self.assertEqual(
            flavors["chs-flavor-static"]["spec"]["topologyName"],
            "chs-node-topology-static",
        )

        cq_flavor_names = [
            f["name"]
            for f in cluster_queue["spec"]["resourceGroups"][0]["flavors"]
        ]
        self.assertEqual(
            cq_flavor_names, ["chs-flavor-dynamic", "chs-flavor-static"]
        )

    def test_generate_configs_never_renders_admission_check(self):
        """CHS drives slice formation itself, so Kueue must not gate on it."""
        for dynamic_slicing in (True, False):
            with self.subTest(dynamic_slicing=dynamic_slicing):
                docs = self._render_all(
                    "tpu7x-standard-4t", dynamic_slicing=dynamic_slicing
                )
                kinds = [d["kind"] for d in docs]
                self.assertNotIn("AdmissionCheck", kinds)
                cq = next(d for d in docs if d["kind"] == "ClusterQueue")
                self.assertNotIn("admissionChecks", cq["spec"])

    def test_apply_configs_success(self):
        """Test apply_configs cleans up and applies configurations."""
        self.mock_scheduler._run_kubectl.side_effect = [
            "",  # cleanup localqueue
            "",  # cleanup clusterqueue
            "",  # cleanup resourceflavor
            "",  # cleanup topology
            "",  # apply
            "",  # wait
        ]
        self.kueue_mgr.apply_configs(self.tmp_dir)
        self.assertEqual(self.mock_scheduler._run_kubectl.call_count, 6)

    def test_apply_and_verify_permission_error_raises(self):
        """Test applying config with kubectl error raises KueueException."""
        self.mock_scheduler._run_kubectl.side_effect = (
            subprocess.CalledProcessError(
                1, "kubectl apply", stderr="permission denied"
            )
        )
        with self.assertRaisesRegex(
            KueueException, "Failed to create ResourceFlavors"
        ):
            self.kueue_mgr.apply_and_verify("/tmp/nonexistent.yaml")

    def test_apply_and_verify_success(self):
        """Test apply_and_verify successfully runs apply and wait commands."""
        self.mock_scheduler._run_kubectl.return_value = ""
        self.kueue_mgr.apply_and_verify("/tmp/kueue-configs.yaml")

        self.mock_scheduler._run_kubectl.assert_any_call(
            ["apply", "-f", "/tmp/kueue-configs.yaml"],
            timeout=None,
        )

        self.mock_scheduler._run_kubectl.assert_any_call(
            [
                "wait",
                "--for=condition=Active",
                "clusterqueue/chs-cluster-queue",
                "--timeout=300s",
            ],
            timeout=350,
        )

    def test_apply_and_verify_timeout_raises(self):
        """Test timeout during verification wait loop raises KueueException."""
        self.mock_scheduler._run_kubectl.side_effect = [
            "",
            subprocess.CalledProcessError(
                1, "kubectl wait", stderr="timed out"
            ),
        ]

        with self.assertRaisesRegex(
            KueueException, "Timeout or error waiting for Kueue ClusterQueue"
        ):
            self.kueue_mgr.apply_and_verify("/tmp/nonexistent.yaml")

        self.mock_scheduler._run_kubectl.assert_any_call(
            [
                "wait",
                "--for=condition=Active",
                "clusterqueue/chs-cluster-queue",
                "--timeout=300s",
            ],
            timeout=350,
        )

    def test_cleanup_success(self):
        """Test cleanup deletes resources via kubectl."""
        self.mock_scheduler._run_kubectl.return_value = ""
        self.kueue_mgr.cleanup()
        self.assertEqual(self.mock_scheduler._run_kubectl.call_count, 4)
        self.mock_scheduler._run_kubectl.assert_any_call(
            [
                "delete",
                "localqueue",
                "chs-local-queue",
                "-n",
                "chs",
                "--ignore-not-found",
            ],
            timeout=None,
        )
        self.mock_scheduler._run_kubectl.assert_any_call(
            [
                "delete",
                "clusterqueue",
                "chs-cluster-queue",
                "--ignore-not-found",
            ],
            timeout=None,
        )
        self.mock_scheduler._run_kubectl.assert_any_call(
            [
                "delete",
                "resourceflavor",
                "chs-flavor-dynamic",
                "chs-flavor-static",
                "--ignore-not-found",
            ],
            timeout=None,
        )
        self.mock_scheduler._run_kubectl.assert_any_call(
            [
                "delete",
                "topology",
                "chs-node-topology-dynamic",
                "chs-node-topology-static",
                "--ignore-not-found",
            ],
            timeout=None,
        )

    def test_cleanup_deletes_topology_after_resource_flavor(self):
        """Topologies must outlive the ResourceFlavor that references them."""
        self.mock_scheduler._run_kubectl.return_value = ""
        self.kueue_mgr.cleanup()

        resources = [
            call.args[0][1]
            for call in self.mock_scheduler._run_kubectl.call_args_list
        ]
        self.assertLess(
            resources.index("resourceflavor"), resources.index("topology")
        )

    def test_cleanup_failure_raises(self):
        """Test cleanup error raises KueueException."""
        self.mock_scheduler._run_kubectl.side_effect = (
            subprocess.CalledProcessError(
                1, "kubectl delete", stderr="conflict"
            )
        )
        with self.assertRaisesRegex(
            KueueException, "Failed to delete Kueue resources"
        ):
            self.kueue_mgr.cleanup()

    def test_get_kueue_api_version_v1beta2_preferred(self):
        """Test _get_kueue_api_version returns v1beta2 when both
        are supported."""
        self.mock_scheduler._run_kubectl.return_value = "v1beta1 v1beta2"
        self.assertEqual(self.kueue_mgr._get_kueue_api_version(), "v1beta2")

    def test_get_kueue_api_version_v1beta2(self):
        """Test _get_kueue_api_version returns v1beta2 when only
        v1beta2 is supported."""
        self.mock_scheduler._run_kubectl.return_value = "v1beta2"
        self.assertEqual(self.kueue_mgr._get_kueue_api_version(), "v1beta2")

    def test_get_kueue_api_version_v1beta1_only(self):
        """Test _get_kueue_api_version returns v1beta1 when only
        v1beta1 is supported."""
        self.mock_scheduler._run_kubectl.return_value = "v1beta1"
        self.assertEqual(self.kueue_mgr._get_kueue_api_version(), "v1beta1")

    def test_get_kueue_api_version_fallback(self):
        """Test _get_kueue_api_version falls back to v1beta1 on error."""
        self.mock_scheduler._run_kubectl.side_effect = (
            subprocess.CalledProcessError(1, "cmd")
        )
        self.assertEqual(self.kueue_mgr._get_kueue_api_version(), "v1beta1")


if __name__ == "__main__":
    unittest.main()
