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
import tempfile
import unittest
from unittest import mock

import yaml

from chs.constants import (
    CHS_NAMESPACE,
    DELETE_WAIT_TIMEOUT_SEC,
    SUPPORTED_DYNAMIC_SLICE_TOPOLOGY,
)
from chs.exceptions import KueueException
from chs.utils.k8s_client import KubeError
from chs.utils.kueue import (
    FLAVOR_NAMES,
    KUEUE_API_GROUP,
    TOPOLOGY_NAMES,
    KueueManager,
)
from tests.k8s_fakes import Call, install_fake_client, patch_k8s_client

_CLUSTER_QUEUE_CRD = "clusterqueues.kueue.x-k8s.io"
_DELETIONS = (
    ("localqueues", ["chs-local-queue"], CHS_NAMESPACE),
    ("clusterqueues", ["chs-cluster-queue"], None),
    ("resourceflavors", FLAVOR_NAMES, None),
    ("topologies", TOPOLOGY_NAMES, None),
)
_SERVER_ERROR = KubeError(
    "Error from server (InternalError): etcdserver: request timed out",
    status=500,
)


def _cleanup_calls(version: str = "v1beta1") -> list[Call]:
    return [
        Call("get_crd", {"name": _CLUSTER_QUEUE_CRD}),
        *[
            Call(
                "delete_custom_objects",
                {
                    "group": KUEUE_API_GROUP,
                    "version": version,
                    "plural": plural,
                    "object_names": object_names,
                    "namespace": namespace,
                    "wait_timeout_seconds": DELETE_WAIT_TIMEOUT_SEC,
                },
            )
            for plural, object_names, namespace in _DELETIONS
        ],
    ]


class TestKueueManager(unittest.TestCase):
    """Tests for KueueManager."""

    def setUp(self):
        super().setUp()
        self.kube = install_fake_client(self)
        self.mock_scheduler = mock.Mock()
        self.mock_scheduler.kueue_supported = False
        self.mock_scheduler.workspace_config_dir = None
        self.kueue_mgr = KueueManager(self.mock_scheduler)
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = self.tmp_dir_obj.name
        self.addCleanup(self.tmp_dir_obj.cleanup)
        self.config_path = os.path.join(self.tmp_dir, "kueue-configs.yaml")

    def _add_kueue_objects(self):
        for plural, object_names, namespace in _DELETIONS:
            for name in object_names:
                self.kube.add_object(
                    KUEUE_API_GROUP,
                    plural,
                    {"metadata": {"name": name}},
                    namespace=namespace,
                )

    def _stored_kueue_names(self) -> list[str]:
        remaining = []
        for plural, object_names, namespace in _DELETIONS:
            stored = set(
                self.kube.stored_names(KUEUE_API_GROUP, plural, namespace)
            )
            remaining.extend(name for name in object_names if name in stored)
        return remaining

    def test_generate_configs_unsupported_machine_type_raises(self):
        """Test generate_configs with unsupported machine type raises
        KueueException."""
        with self.assertRaisesRegex(
            KueueException, "not recognized in MACHINE_TYPE_RESOURCE_MAP"
        ):
            self.kueue_mgr.generate_configs(
                self.tmp_dir, "unsupported-mt", ["node1"]
            )
        self.assertEqual(self.kube.calls, [])

    def test_generate_configs_success_renders(self):
        """Test generate_configs renders template and returns correct label."""
        self.kube.add_crd(_CLUSTER_QUEUE_CRD, versions=("v1beta2",))
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
        self.kube.add_crd(_CLUSTER_QUEUE_CRD, versions=("v1beta1",))
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
        self.kube.add_crd(_CLUSTER_QUEUE_CRD, versions=("v1beta1",))
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
        """Non-dynamic clusters get a hostname-only topology."""
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
        """Test apply_configs cleans up and applies configs."""
        self.kueue_mgr.generate_configs(
            self.tmp_dir, "a3-megagpu-8g", ["node1", "node2"]
        )
        self.kube.add_object(
            KUEUE_API_GROUP,
            "clusterqueues",
            {
                "metadata": {"name": "chs-cluster-queue"},
                "status": {
                    "conditions": [{"type": "Active", "status": "True"}]
                },
            },
        )
        with (
            mock.patch.object(self.kueue_mgr, "cleanup") as mock_cleanup,
            mock.patch.object(
                self.kueue_mgr, "apply_and_verify"
            ) as mock_apply_and_verify,
        ):
            self.kueue_mgr.apply_configs(self.tmp_dir)
        mock_cleanup.assert_called_once_with()
        mock_apply_and_verify.assert_called_once_with(self.config_path)

    def test_apply_and_verify_permission_error_raises(self):
        """Test a Kubernetes API error on apply raises KueueException."""
        self.kueue_mgr.generate_configs(
            self.tmp_dir, "a3-megagpu-8g", ["node1", "node2"]
        )
        err = KubeError(
            "Error from server (Forbidden): clusterqueues.kueue.x-k8s.io is "
            'forbidden: User "chs" cannot create resource "clusterqueues"',
            status=403,
        )
        self.kube.fail_next("apply_custom_object", err)
        with self.assertRaises(KueueException) as ctx:
            self.kueue_mgr.apply_and_verify(self.config_path)
        self.assertEqual(
            str(ctx.exception),
            "Failed to create ResourceFlavors, ClusterQueues, and/or "
            f"LocalQueues. Details: {err}",
        )
        self.assertIs(ctx.exception.__cause__, err)
        self.assertEqual(self.kube.calls_to("get_custom_object"), [])

    def test_apply_and_verify_success(self):
        """Test apply_and_verify applies all docs and waits for Active."""
        self.kueue_mgr.generate_configs(
            self.tmp_dir, "a3-megagpu-8g", ["node1", "node2"]
        )
        self.kube.add_object(
            KUEUE_API_GROUP,
            "clusterqueues",
            {
                "metadata": {"name": "chs-cluster-queue"},
                "status": {
                    "conditions": [{"type": "Active", "status": "True"}]
                },
            },
        )
        self.kube.calls.clear()

        self.kueue_mgr.apply_and_verify(self.config_path)

        self.assertEqual(len(self.kube.calls_to("apply_custom_object")), 4)
        self.assertEqual(
            self.kube.calls_to("get_custom_object"),
            [
                Call(
                    "get_custom_object",
                    {
                        "group": KUEUE_API_GROUP,
                        "version": "v1beta1",
                        "plural": "clusterqueues",
                        "name": "chs-cluster-queue",
                    },
                )
            ],
        )
        self.assertEqual(
            self.kube.stored_names(KUEUE_API_GROUP, "localqueues", "chs"),
            ["chs-local-queue"],
        )
        self.assertEqual(
            self.kube.stored_names(KUEUE_API_GROUP, "resourceflavors"),
            ["chs-flavor-static"],
        )
        self.assertEqual(
            self.kube.stored_names(KUEUE_API_GROUP, "topologies"),
            ["chs-node-topology-static"],
        )

    def test_apply_and_verify_uses_cluster_queue_version_and_requires_it(self):
        """ClusterQueue API version comes from the ClusterQueue doc itself."""
        docs = [
            {
                "apiVersion": f"{KUEUE_API_GROUP}/v1beta1",
                "kind": "ResourceFlavor",
                "metadata": {"name": "chs-flavor-static"},
            },
            {
                "apiVersion": f"{KUEUE_API_GROUP}/v1beta2",
                "kind": "ClusterQueue",
                "metadata": {"name": "chs-cluster-queue"},
            },
        ]
        with open(self.config_path, "w") as f:
            yaml.safe_dump_all(docs, f)
        self.kube.add_object(
            KUEUE_API_GROUP,
            "clusterqueues",
            {
                "metadata": {"name": "chs-cluster-queue"},
                "status": {
                    "conditions": [{"type": "Active", "status": "True"}]
                },
            },
        )

        self.kueue_mgr.apply_and_verify(self.config_path)

        self.assertEqual(
            self.kube.calls_to("get_custom_object"),
            [
                Call(
                    "get_custom_object",
                    {
                        "group": KUEUE_API_GROUP,
                        "version": "v1beta2",
                        "plural": "clusterqueues",
                        "name": "chs-cluster-queue",
                    },
                )
            ],
        )

        with open(self.config_path, "w") as f:
            yaml.safe_dump_all([docs[0]], f)
        with self.assertRaisesRegex(
            KueueException, "No ClusterQueue document found"
        ):
            self.kueue_mgr.apply_and_verify(self.config_path)

    def test_apply_and_verify_polls_then_times_out(self):
        """Test verification wait loop polls and raises on timeout."""
        self.kueue_mgr.generate_configs(
            self.tmp_dir, "a3-megagpu-8g", ["node1", "node2"]
        )
        self.kube.add_object(
            KUEUE_API_GROUP,
            "clusterqueues",
            {"metadata": {"name": "chs-cluster-queue"}, "status": None},
        )
        self.kube.calls.clear()

        with (
            mock.patch(
                "chs.utils.kueue.time.monotonic", side_effect=[0, 10, 301]
            ),
            mock.patch("chs.utils.kueue.time.sleep") as mock_sleep,
        ):
            with self.assertRaises(KueueException) as ctx:
                self.kueue_mgr.apply_and_verify(self.config_path)

        mock_sleep.assert_called_once_with(0.5)
        cause = ctx.exception.__cause__
        self.assertIsInstance(cause, KubeError)
        self.assertEqual(
            str(ctx.exception),
            "Timeout or error waiting for Kueue ClusterQueue "
            f"'chs-cluster-queue' to become active. Details: {cause}",
        )

    def test_cleanup_success(self):
        """Test cleanup deletes all resources in dependency order."""
        self._add_kueue_objects()
        self.kueue_mgr.cleanup()
        self.assertEqual(self.kube.calls, _cleanup_calls("v1beta1"))
        self.assertEqual(self._stored_kueue_names(), [])

    def test_cleanup_ignores_missing_objects(self):
        """Test cleanup succeeds when nothing is left."""
        self.kueue_mgr.cleanup()
        self.assertEqual(self.kube.calls, _cleanup_calls("v1beta1"))

    def test_cleanup_deletes_topology_after_resource_flavor(self):
        """Topologies must outlive the ResourceFlavor that references them."""
        self.kueue_mgr.cleanup()
        delete_calls = self.kube.calls_to("delete_custom_objects")
        plurals = [call.kwargs["plural"] for call in delete_calls]
        self.assertEqual(
            plurals,
            ["localqueues", "clusterqueues", "resourceflavors", "topologies"],
        )

    def test_cleanup_failure_raises(self):
        """Test cleanup error raises KueueException."""
        err = KubeError(
            "Error from server (Conflict): Operation cannot be fulfilled",
            status=409,
        )
        self.kube.fail_always("delete_custom_objects", err)
        with self.assertRaises(KueueException) as ctx:
            self.kueue_mgr.cleanup()
        self.assertEqual(
            str(ctx.exception),
            f"Failed to delete Kueue resources. Details: {err}",
        )
        self.assertIs(ctx.exception.__cause__, err)
        self.assertEqual(
            self.kube.calls,
            [
                Call("get_crd", {"name": _CLUSTER_QUEUE_CRD}),
                Call(
                    "delete_custom_objects",
                    {
                        "group": KUEUE_API_GROUP,
                        "version": "v1beta1",
                        "plural": "localqueues",
                        "object_names": ["chs-local-queue"],
                        "namespace": CHS_NAMESPACE,
                        "wait_timeout_seconds": DELETE_WAIT_TIMEOUT_SEC,
                    },
                ),
            ],
        )

    def test_cleanup_without_kubeconfig_raises(self):
        """Test a client that cannot be created raises KueueException."""
        err = KubeError("Invalid kube-config file. No configuration.")
        with patch_k8s_client(error=err):
            with self.assertRaises(KueueException) as ctx:
                self.kueue_mgr.cleanup()
        self.assertEqual(
            str(ctx.exception),
            f"Failed to delete Kueue resources. Details: {err}",
        )
        self.assertIs(ctx.exception.__cause__, err)

    def test_cleanup_timeout_raises(self):
        """A stuck finalizer must surface as KueueException, not hang."""
        self.kube.fail_next(
            "delete_custom_objects",
            None,
            KubeError("Timed out waiting for deletion of chs-cluster-queue"),
        )
        with self.assertRaisesRegex(
            KueueException, "Failed to delete Kueue resources"
        ):
            self.kueue_mgr.cleanup()
        self.assertEqual(self.kube.calls, _cleanup_calls()[:3])

    def test_apply_configs_cleanup_failure_propagates(self):
        """apply_configs fails fast if pre-apply cleanup times out or fails."""
        self.kueue_mgr.generate_configs(
            self.tmp_dir, "a3-megagpu-8g", ["node1", "node2"]
        )
        self.kube.fail_next(
            "delete_custom_objects",
            KubeError("Timed out waiting for deletion of chs-local-queue"),
        )
        with self.assertRaisesRegex(
            KueueException, "Failed to delete Kueue resources"
        ):
            self.kueue_mgr.apply_configs(self.tmp_dir)
        self.assertEqual(self.kube.calls_to("apply_custom_object"), [])

    def test_apply_and_verify_apply_timeout_raises(self):
        """An apply timeout is reported as KueueException."""
        self.kueue_mgr.generate_configs(
            self.tmp_dir, "a3-megagpu-8g", ["node1", "node2"]
        )
        self.kube.fail_next("apply_custom_object", KubeError("read timed out"))
        with self.assertRaisesRegex(
            KueueException, "Failed to create ResourceFlavors"
        ):
            self.kueue_mgr.apply_and_verify(self.config_path)

    def test_get_kueue_api_version_v1beta2_preferred(self):
        """Test _get_kueue_api_version returns v1beta2 when both
        are supported."""
        self.kube.add_crd(_CLUSTER_QUEUE_CRD, versions=("v1beta1", "v1beta2"))
        self.assertEqual(self.kueue_mgr._get_kueue_api_version(), "v1beta2")
        self.assertEqual(
            self.kube.calls,
            [Call("get_crd", {"name": _CLUSTER_QUEUE_CRD})],
        )

    def test_get_kueue_api_version_v1beta2(self):
        """Test _get_kueue_api_version returns v1beta2 when only
        v1beta2 is supported."""
        self.kube.add_crd(_CLUSTER_QUEUE_CRD, versions=("v1beta2",))
        self.assertEqual(self.kueue_mgr._get_kueue_api_version(), "v1beta2")

    def test_get_kueue_api_version_v1beta1_only(self):
        """Test _get_kueue_api_version returns v1beta1 when only
        v1beta1 is supported."""
        with self.subTest("v1beta1 served"):
            self.kube.add_crd(_CLUSTER_QUEUE_CRD, versions=("v1beta1",))
            self.assertEqual(self.kueue_mgr._get_kueue_api_version(), "v1beta1")
        with self.subTest("no versions listed"):
            self.kube.add_object(
                "apiextensions.k8s.io",
                "customresourcedefinitions",
                {"metadata": {"name": _CLUSTER_QUEUE_CRD}},
            )
            self.assertEqual(self.kueue_mgr._get_kueue_api_version(), "v1beta1")

    def test_get_kueue_api_version_fallback(self):
        """Test _get_kueue_api_version falls back to v1beta1 on error."""
        with self.subTest("CRD not installed"):
            self.assertEqual(self.kueue_mgr._get_kueue_api_version(), "v1beta1")
        with self.subTest("API error"):
            self.kube.add_crd(_CLUSTER_QUEUE_CRD, versions=("v1beta2",))
            self.kube.fail_next("get_crd", _SERVER_ERROR)
            self.assertEqual(self.kueue_mgr._get_kueue_api_version(), "v1beta1")
        with self.subTest("no kubeconfig"):
            with patch_k8s_client(error=KubeError("no configuration")):
                self.assertEqual(
                    self.kueue_mgr._get_kueue_api_version(), "v1beta1"
                )


if __name__ == "__main__":
    unittest.main()
