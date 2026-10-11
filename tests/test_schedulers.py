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
import datetime
import json
import os
import tempfile
from typing import Any
import unittest
import yaml
from unittest import mock
from chs.constants import (
    CHS_NAMESPACE,
    DELETE_WAIT_TIMEOUT_SEC,
    NAMESPACE_DELETE_TIMEOUT_SEC,
    SUPPORTED_DYNAMIC_SLICE_TOPOLOGY,
)
from chs.schedulers.kubernetes import (
    KubernetesScheduler,
    TpuSliceData,
)
from chs.schedulers.slurm import SlurmScheduler
from chs.schedulers import _get_kubectl_context, detect_scheduler
from chs.exceptions import ChsSchedulerException
from chs.utils import k8s_client
import click
from chs import exceptions as ex
from tests.k8s_fakes import (
    Call,
    install_fake_client,
)

_LIST_CHS_SLICES = Call(
    "list_custom_objects",
    {
        "group": "accelerator.gke.io",
        "version": "v1beta1",
        "plural": "slices",
        "label_selector": "created-by=chs",
    },
)

_DELETE_CHS_NAMESPACE = Call(
    "delete_namespace",
    {
        "name": CHS_NAMESPACE,
        "wait_timeout_seconds": NAMESPACE_DELETE_TIMEOUT_SEC,
    },
)


def _chs_slice(pool_name: str, reason: str) -> dict:
    """Returns a GKE Slice object created by CHS with the given Ready reason."""
    return {
        "metadata": {
            "name": f"chs-slice-{pool_name}",
            "labels": {"created-by": "chs", "chs-nodepool": pool_name},
        },
        "status": {
            "conditions": [{"type": "Ready", "reason": reason}],
        },
    }


def _provider_node(name: str, provider_id: str) -> dict:
    """Returns a Node object with the given spec.providerID."""
    return {
        "metadata": {"name": name},
        "spec": {"providerID": provider_id},
    }


def _job_object(
    name: str,
    nodes: list[str] | None = None,
    suspended: bool = True,
    creation_timestamp: str = "2026-10-07T00:00:00Z",
) -> dict[str, Any]:
    """Returns a Job object with the given name and optional pinned nodes."""
    metadata = {
        "name": name,
        "creationTimestamp": creation_timestamp,
        "namespace": CHS_NAMESPACE,
    }
    spec: dict[str, Any] = {"suspend": suspended}
    if nodes is not None:
        required_terms = [
            {
                "matchExpressions": [
                    {
                        "key": "kubernetes.io/hostname",
                        "operator": "In",
                        "values": nodes,
                    }
                ]
            }
        ]
        affinity = {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": required_terms
                }
            }
        }
        spec["template"] = {"spec": {"affinity": affinity}}
    return {
        "metadata": metadata,
        "spec": spec,
    }


def _job_manifest(name: str, node: str) -> dict[str, Any]:
    """Returns a rendered jobset.yaml Job pinned to one node."""
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        **_job_object(name, [node]),
    }


def _pending_pod(
    name: str,
    job_name: str | None = None,
    creation_timestamp: str = "2026-10-07T00:00:00Z",
    scheduled: bool = False,
    unschedulable: bool = True,
) -> dict[str, Any]:
    """Returns a Pending Pod object with the given scheduling condition."""
    labels = {}
    if job_name:
        labels["batch.kubernetes.io/job-name"] = job_name
    conditions = []
    if not scheduled and unschedulable:
        conditions.append(
            {
                "type": "PodScheduled",
                "status": "False",
                "reason": "Unschedulable",
            }
        )
    elif scheduled:
        conditions.append(
            {
                "type": "PodScheduled",
                "status": "True",
            }
        )
    return {
        "metadata": {
            "name": name,
            "namespace": CHS_NAMESPACE,
            "labels": labels,
            "creationTimestamp": creation_timestamp,
        },
        "status": {
            "phase": "Pending",
            "conditions": conditions,
        },
    }


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

    def test_allow_interdomain_pairs(self, mock_logger):
        scheduler = SlurmScheduler()
        self.assertFalse(scheduler.allow_interdomain_pairs("a4x-highgpu-4g"))
        self.assertFalse(
            scheduler.allow_interdomain_pairs("a4x-maxgpu-4g-metal")
        )
        self.assertTrue(scheduler.allow_interdomain_pairs("a3-megagpu-8g"))
        self.assertTrue(scheduler.allow_interdomain_pairs("a3-ultragpu-8g"))
        self.assertTrue(scheduler.allow_interdomain_pairs())


@mock.patch("chs.schedulers.kubernetes.logger")
class TestKubernetesScheduler(unittest.TestCase):
    """Tests for KubernetesScheduler."""

    def setUp(self):
        super().setUp()
        self.kube = install_fake_client(self)
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = self.tmp_dir_obj.name
        self.addCleanup(self.tmp_dir_obj.cleanup)
        self.mock_app_data = mock.Mock()
        self.mock_app_data.get_run.return_value = {}

    def test_allow_interdomain_pairs(self, mock_logger):
        scheduler = KubernetesScheduler()
        self.assertTrue(scheduler.allow_interdomain_pairs())
        self.assertTrue(
            scheduler.allow_interdomain_pairs("a4x-maxgpu-4g-metal")
        )

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
        "chs.schedulers.kubernetes.KubernetesScheduler._get_not_ready_nodes",
        return_value=set(),
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_busy_nodes",
        return_value=set(),
    )
    @mock.patch("chs.schedulers.kubernetes.KubernetesScheduler._get_tpu_slices")
    def test_get_nodes_under_test(
        self,
        mock_tpu_slices,
        mock_busy,
        mock_not_ready,
        mock_logger,
    ):
        """Test get_nodes_under_test returns parsed nodes."""
        for name in ("node1", "node2"):
            self.kube.add_object("", "nodes", {"metadata": {"name": name}})
        mock_tpu_slices.return_value = TpuSliceData(
            pools={}, node_to_pool={}, pool_to_topology={}
        )
        scheduler = KubernetesScheduler(
            nodes="node1, node2 ",
        )
        self.assertCountEqual(
            scheduler.get_nodes_under_test(), ["node1", "node2"]
        )

    def test_is_node_busy_true(self, mock_logger):
        """Test _is_node_busy returns True for nodes with TPU/GPU requests."""
        scheduler = KubernetesScheduler()
        self.kube.add_object(
            "",
            "pods",
            {
                "metadata": {"name": "tpu-pod"},
                "spec": {
                    "nodeName": "node1",
                    "containers": [
                        {
                            "name": "worker",
                            "resources": {"requests": {"google.com/tpu": "4"}},
                        }
                    ],
                },
                "status": {"phase": "Running"},
            },
        )
        self.assertTrue(scheduler._is_node_busy("node1"))
        self.assertEqual(
            self.kube.calls,
            [
                Call(
                    "list_pods",
                    {
                        "field_selector": (
                            "spec.nodeName=node1,"
                            "status.phase!=Succeeded,status.phase!=Failed"
                        ),
                    },
                )
            ],
        )

    def test_is_node_busy_false(self, mock_logger):
        """Test _is_node_busy returns False if no accelerator pods exist."""
        scheduler = KubernetesScheduler()
        self.kube.add_object(
            "",
            "pods",
            {
                "metadata": {"name": "cpu-pod"},
                "spec": {
                    "nodeName": "node1",
                    "containers": [
                        {
                            "name": "worker",
                            "resources": {"requests": {"cpu": "4"}},
                        }
                    ],
                },
                "status": {"phase": "Running"},
            },
        )
        self.assertFalse(scheduler._is_node_busy("node1"))

    def test_is_node_busy_zero_and_limits(self, mock_logger):
        """Test _is_node_busy ignores 0/"0" accelerator counts and checks
        both requests and limits."""
        scheduler = KubernetesScheduler()
        self.kube.add_object(
            "",
            "pods",
            {
                "metadata": {"name": "zero-accel-pod"},
                "spec": {
                    "nodeName": "node1",
                    "containers": [
                        {
                            "name": "worker",
                            "resources": {
                                "requests": {
                                    "nvidia.com/gpu": "0",
                                    "google.com/tpu": 0,
                                    "cpu": "4",
                                },
                                "limits": {
                                    "nvidia.com/gpu": "0",
                                    "google.com/tpu": 0,
                                    "memory": "16Gi",
                                },
                            },
                        }
                    ],
                },
                "status": {"phase": "Running"},
            },
        )
        self.assertFalse(scheduler._is_node_busy("node1"))

        cases = [
            (
                "gpu in requests",
                "containers",
                {"requests": {"nvidia.com/gpu": "1"}},
            ),
            (
                "gpu in limits",
                "containers",
                {"limits": {"nvidia.com/gpu": "1"}},
            ),
            (
                "tpu in requests",
                "containers",
                {"requests": {"google.com/tpu": "1"}},
            ),
            (
                "tpu in limits",
                "containers",
                {"limits": {"google.com/tpu": "1"}},
            ),
            (
                "tpu in initContainers",
                "initContainers",
                {"requests": {"google.com/tpu": "1"}},
            ),
        ]
        for label, field_name, resources in cases:
            with self.subTest(label):
                self.kube.objects.clear()
                self.kube.add_object(
                    "",
                    "pods",
                    {
                        "metadata": {"name": "accel-pod"},
                        "spec": {
                            "nodeName": "node1",
                            field_name: [
                                {"name": "worker", "resources": resources}
                            ],
                        },
                        "status": {"phase": "Running"},
                    },
                )
                self.assertTrue(scheduler._is_node_busy("node1"))

    @mock.patch("time.sleep", return_value=None)
    def test_is_node_busy_error_retries(self, mock_sleep, mock_logger):
        """Test _is_node_busy raises exception after API retries fail."""
        scheduler = KubernetesScheduler()
        self.kube.fail_always(
            "list_pods", k8s_client.KubeError("boom", status=500)
        )
        with self.assertRaises(ChsSchedulerException):
            scheduler._is_node_busy("node1")
        self.assertEqual(len(self.kube.calls_to("list_pods")), 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @mock.patch("chs.schedulers.kubernetes.KubernetesScheduler._is_node_busy")
    def test_get_busy_nodes(self, mock_is_node_busy, mock_logger):
        """Test _get_busy_nodes batches calls and returns busy nodes."""
        scheduler = KubernetesScheduler()
        mock_is_node_busy.side_effect = lambda n: n == "node2"
        target_nodes = ["node1", "node2", "node3"]

        busy_nodes = scheduler._get_busy_nodes(target_nodes)
        self.assertEqual(busy_nodes, {"node2"})
        self.assertEqual(mock_is_node_busy.call_count, 3)

    def test_get_not_ready_nodes(self, mock_logger):
        """Test _get_not_ready_nodes identifies NotReady and Cordoned nodes."""
        scheduler = KubernetesScheduler()

        for node in [
            {
                "metadata": {"name": "ready-node"},
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
            },
            {
                "metadata": {"name": "not-ready-node"},
                "status": {
                    "conditions": [{"type": "Ready", "status": "False"}]
                },
            },
            {
                "metadata": {"name": "unknown-node"},
                "status": {
                    "conditions": [{"type": "Ready", "status": "Unknown"}]
                },
            },
            {
                "metadata": {"name": "cordoned-node"},
                "spec": {"unschedulable": True},
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
            },
            {
                "metadata": {"name": "empty-conditions-node"},
                "status": {"conditions": []},
            },
            {
                "metadata": {"name": "null-status-node"},
                "status": None,
                "spec": None,
            },
        ]:
            self.kube.add_object("", "nodes", node)

        not_ready = scheduler._get_not_ready_nodes()

        self.assertEqual(
            not_ready,
            {
                "not-ready-node",
                "unknown-node",
                "cordoned-node",
                "empty-conditions-node",
                "null-status-node",
            },
        )

    def test_get_not_ready_nodes_with_nodepools(self, mock_logger):
        """Test _get_not_ready_nodes sends nodepools filter to API."""
        scheduler = KubernetesScheduler(nodepools="pool1,pool2")
        for name, pool in (("node1", "pool1"), ("node2", "pool2")):
            self.kube.add_object(
                "",
                "nodes",
                {
                    "metadata": {
                        "name": name,
                        "labels": {"cloud.google.com/gke-nodepool": pool},
                    }
                },
            )

        scheduler._get_not_ready_nodes()

        self.assertEqual(
            self.kube.calls,
            [
                Call(
                    "list_nodes",
                    {
                        "label_selector": (
                            "cloud.google.com/gke-nodepool in (pool1,pool2)"
                        )
                    },
                )
            ],
        )

    def test_get_not_ready_nodes_list_kind_null_items(self, mock_logger):
        """Test empty node list in _get_not_ready_nodes."""
        scheduler = KubernetesScheduler()

        not_ready = scheduler._get_not_ready_nodes()

        self.assertEqual(not_ready, set())

    def test_fetch_raw_node_resources(self, mock_logger):
        """Test _fetch_raw_node_resources lists non-system nodes."""
        scheduler = KubernetesScheduler()
        stored = self.kube.add_object(
            "", "nodes", {"metadata": {"name": "node1"}}
        )

        resources = scheduler._fetch_raw_node_resources()

        self.assertEqual(resources, [stored])
        self.assertEqual(
            self.kube.calls,
            [
                Call(
                    "list_nodes",
                    {"label_selector": "cloud.google.com/gke-nodepool!=system"},
                )
            ],
        )

    def test_fetch_raw_node_resources_missing_nodepool(self, mock_logger):
        """Test _fetch_raw_node_resources raises error for missing nodepool."""
        scheduler = KubernetesScheduler(nodepools="pool-a,pool-b")
        self.kube.add_object(
            "",
            "nodes",
            {
                "metadata": {
                    "name": "node1",
                    "labels": {"cloud.google.com/gke-nodepool": "pool-a"},
                }
            },
        )

        with self.assertRaisesRegex(ex.ChsValidationError, "pool-b"):
            scheduler._fetch_raw_node_resources()

    def test_classify_unhealthy_nodes(self, mock_logger):
        """Test _classify_unhealthy_nodes filters unschedulable nodes."""
        scheduler = KubernetesScheduler()
        node_resources = [
            {
                "metadata": {"name": "ready-node"},
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
            },
            {
                "metadata": {"name": "not-ready-node"},
                "status": {
                    "conditions": [{"type": "Ready", "status": "False"}]
                },
            },
            {
                "metadata": {"name": "cordoned-node"},
                "spec": {"unschedulable": True},
            },
        ]

        unhealthy = scheduler._classify_unhealthy_nodes(node_resources)

        self.assertEqual(unhealthy, {"not-ready-node", "cordoned-node"})

    def test_get_not_ready_nodes_invalid_json(self, mock_logger):
        """Test _get_not_ready_nodes raises exception on API failure."""
        scheduler = KubernetesScheduler()
        self.kube.fail_next("list_nodes", k8s_client.KubeError("API failure"))

        with self.assertRaises(ChsSchedulerException):
            scheduler._get_not_ready_nodes()

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_not_ready_nodes",
        return_value={"node2"},
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_busy_nodes",
        return_value=set(),
    )
    @mock.patch("chs.schedulers.kubernetes.KubernetesScheduler._get_tpu_slices")
    def test_get_nodes_under_test_tpu_slice_not_ready(
        self,
        mock_tpu_slices,
        mock_busy,
        mock_not_ready,
        mock_logger,
    ):
        """Test TPU slice is dropped if any node in slice is NotReady."""
        self.kube.add_object("", "nodes", {"metadata": {"name": "node1"}})
        mock_tpu_slices.return_value = TpuSliceData(
            pools={"pool1": ["node1", "node2"]},
            node_to_pool={"node1": "pool1", "node2": "pool1"},
            pool_to_topology={"pool1": "2x2"},
        )
        scheduler = KubernetesScheduler(nodes="node1")

        valid_nodes = scheduler.get_nodes_under_test()

        self.assertEqual(valid_nodes, [])
        mock_logger.warning.assert_called_once()

    def test_resolve_raw_nodes_empty(self, mock_logger):
        """Test resolve_raw_nodes returns empty list if node discovery fails."""
        scheduler = KubernetesScheduler()
        self.kube.fail_next("list_nodes", k8s_client.KubeError("boom"))

        self.assertEqual(scheduler.resolve_raw_nodes(), [])

    def test_resolve_raw_nodes_nodepool(self, mock_logger):
        """Test resolve_raw_nodes uses nodepools to fetch nodes."""
        scheduler = KubernetesScheduler(
            nodepools="pool-1",
        )
        for name in ("node-1", "node-2"):
            self.kube.add_object(
                "",
                "nodes",
                {
                    "metadata": {
                        "name": name,
                        "labels": {"cloud.google.com/gke-nodepool": "pool-1"},
                    }
                },
            )

        nodes = scheduler.resolve_raw_nodes()

        self.assertEqual(nodes, ["node-1", "node-2"])
        self.assertEqual(
            self.kube.calls,
            [
                Call(
                    "list_nodes",
                    {
                        "label_selector": (
                            "cloud.google.com/gke-nodepool in (pool-1)"
                        )
                    },
                )
            ],
        )

    def test_resolve_raw_nodes_precedence(self, mock_logger):
        """Test nodes arg overrides nodepools, and executes validation."""
        scheduler = KubernetesScheduler(
            nodes="node-1",
            nodepools="pool-1",
        )
        self.kube.add_object("", "nodes", {"metadata": {"name": "node-1"}})
        nodes = scheduler.resolve_raw_nodes()
        self.assertEqual(nodes, ["node-1"])
        self.assertEqual(
            self.kube.calls,
            [Call("list_nodes", {})],
        )

    def test_resolve_raw_nodes_invalid(self, mock_logger):
        """Test resolve_raw_nodes errors if provided nodes don't exist."""
        scheduler = KubernetesScheduler(
            nodes="invalid-node",
        )

        with self.assertRaises(ex.ChsSchedulerException) as cm:
            scheduler.resolve_raw_nodes()

        self.assertIn(
            "One or more targeted nodes not found in the cluster.",
            str(cm.exception),
        )
        self.assertIn("invalid-node", str(cm.exception))

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

        with open(output_path, "r") as f:
            data = yaml.safe_load(f)

        variables = data["ramble"]["variables"]
        self.assertEqual(variables["nodes"], ["node1"])
        self.assertEqual(variables["machine_type"], "a3-megagpu-8g")
        self.assertEqual(variables["tests"], ["test1"])
        self.assertEqual(variables["default_interface_name"], "eth0")

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

        with open(output_path, "r") as f:
            data = yaml.safe_load(f)

        variables = data["ramble"]["variables"]
        self.assertEqual(variables["deadline"], 1800)

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler.get_nodes_under_test",
        return_value=["node1"],
    )
    def test_setup_scheduler_resources_success(
        self, mock_validate, mock_logger
    ):
        """Test setup_scheduler_resources creates dir and returns tests."""
        scheduler = KubernetesScheduler(
            nodes="node1",
        )
        workspace_config_dir = os.path.join(self.tmp_dir, "configs")

        tests = scheduler.setup_scheduler_resources(
            workspace_config_dir,
            [{"name": "nccl-cluster"}],
            "a3-megagpu-8g",
            workspace_name="ws1",
        )

        self.assertEqual(tests, [{"name": "nccl-cluster"}])
        self.assertTrue(os.path.isdir(workspace_config_dir))

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler.get_nodes_under_test",
        return_value=["node1", "node2"],
    )
    def test_write_ramble_config_template_defaults(
        self, mock_filter, mock_logger
    ):
        """Test write_ramble_config writes defaults with suspend_jobset."""
        scheduler = KubernetesScheduler(
            nodes="node1",
        )

        workspace_config_dir = os.path.join(self.tmp_dir, "configs")
        os.makedirs(workspace_config_dir, exist_ok=True)

        with mock.patch.object(
            KubernetesScheduler,
            "_discover_networks",
            return_value=(["default"], ["rdma-0"]),
        ):
            scheduler.write_ramble_config(
                workspace_config_dir,
                [{"name": "test1"}],
                "a3-megagpu-8g",
            )

        output_path = os.path.join(workspace_config_dir, "ramble.yaml")

        with open(output_path, "r") as f:
            data = yaml.safe_load(f)

        variables = data["ramble"]["variables"]
        self.assertEqual(
            variables["jobset_labels_yaml"], "app.kubernetes.io/managed-by: chs"
        )
        self.assertEqual(variables["suspend_jobset"], "true")

    def test_discover_networks_success(self, mock_logger):
        """Test _discover_networks with success and grouping."""
        scheduler = KubernetesScheduler()
        for network in [
            {"metadata": {"name": "rdma-1"}, "spec": {"type": "Device"}},
            {"metadata": {"name": "rdma-0"}, "spec": {"type": "Device"}},
            {"metadata": {"name": "gvnic-1"}, "spec": {"type": "L3"}},
            {"metadata": {"name": "default"}, "spec": {"type": "L3"}},
        ]:
            self.kube.add_object("networking.gke.io", "networks", network)

        networks_l3, networks_device = scheduler._discover_networks()

        self.assertEqual(networks_l3, ["default", "gvnic-1"])
        self.assertEqual(networks_device, ["rdma-0", "rdma-1"])
        self.assertEqual(
            self.kube.calls,
            [
                Call(
                    "list_custom_objects",
                    {
                        "group": "networking.gke.io",
                        "version": "v1",
                        "plural": "networks",
                    },
                )
            ],
        )

    def test_discover_networks_failure(self, mock_logger):
        """Test _discover_networks with failure."""
        scheduler = KubernetesScheduler()
        self.kube.fail_next("list_custom_objects", k8s_client.KubeError("boom"))

        networks = scheduler._discover_networks()

        self.assertEqual(networks, ([], []))

    def test_discover_networks_no_default(self, mock_logger):
        """Test _discover_networks fails if default L3 network is missing."""
        scheduler = KubernetesScheduler()
        self.kube.add_object(
            "networking.gke.io",
            "networks",
            {"metadata": {"name": "gvnic-1"}, "spec": {"type": "L3"}},
        )
        networks_l3, networks_device = scheduler._discover_networks()
        self.assertEqual(networks_l3, [])
        self.assertEqual(networks_device, [])
        mock_logger.warning.assert_called_with(
            "The default Layer 3 network configuration "
            "was not found. Discovered "
            "networks: %s. Ensure network "
            "configuration is correct.",
            "gvnic-1",
        )

    def test_discover_networks_malformed_items(self, mock_logger):
        """Test _discover_networks handles malformed network items."""
        scheduler = KubernetesScheduler()
        with self.subTest("item missing spec.type is ignored"):
            self.kube.add_object(
                "networking.gke.io",
                "networks",
                {"metadata": {"name": "broken"}},
            )
            self.kube.add_object(
                "networking.gke.io",
                "networks",
                {"metadata": {"name": "default"}, "spec": {"type": "L3"}},
            )
            self.assertEqual(scheduler._discover_networks(), (["default"], []))
        with self.subTest("non-dict item returns empty lists"):
            with mock.patch.object(
                k8s_client, "list_custom_objects", return_value=["not-a-dict"]
            ):
                self.assertEqual(scheduler._discover_networks(), ([], []))
                mock_logger.warning.assert_called()

    def test_write_ramble_config_discovery_fallback(self, mock_logger):
        """Test write_ramble_config network discovery fallback."""
        self.kube.add_object(
            "",
            "nodes",
            {
                "metadata": {"name": "node1"},
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
            },
        )
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

        with open(output_path, "r") as f:
            data = yaml.safe_load(f)

        variables = data["ramble"]["variables"]
        self.assertEqual(variables["default_interface_name"], "eth0")

        all_interfaces = json.loads(variables["all_interfaces_json"].strip())
        self.assertEqual(len(all_interfaces), 1)
        self.assertEqual(all_interfaces[0]["interfaceName"], "eth0")
        self.assertEqual(all_interfaces[0]["network"], "default")

    def test_get_topology(self, mock_logger):
        """Test get_topology returns mapped nodes."""
        scheduler = KubernetesScheduler(
            nodes="node1,node2",
        )
        for name, block in (("node1", "block1"), ("node2", "block2")):
            self.kube.add_object(
                "",
                "nodes",
                {
                    "metadata": {
                        "name": name,
                        "labels": {
                            "cloud.google.com/gce-topology-block": block
                        },
                    },
                    "status": {
                        "conditions": [{"type": "Ready", "status": "True"}]
                    },
                },
            )

        topology = scheduler.get_topology()

        self.assertEqual(len(topology), 2)
        self.assertEqual(topology["block1"]["default-subblock"], ["node1"])
        self.assertEqual(topology["block2"]["default-subblock"], ["node2"])

    def test_get_topology_missing_label(self, mock_logger):
        """Test get_topology falls back to default-block."""
        scheduler = KubernetesScheduler(
            nodes="node1,node2",
        )
        self.kube.add_object(
            "",
            "nodes",
            {
                "metadata": {"name": "node1", "labels": {}},
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
            },
        )
        self.kube.add_object(
            "",
            "nodes",
            {
                "metadata": {
                    "name": "node2",
                    "labels": {"cloud.google.com/gce-topology-block": "block2"},
                },
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
            },
        )

        topology = scheduler.get_topology()

        self.assertEqual(len(topology), 2)
        self.assertEqual(
            topology["default-block"]["default-subblock"], ["node1"]
        )
        self.assertEqual(topology["block2"]["default-subblock"], ["node2"])

    def test_cleanup_scheduler_resources(self, mock_logger):
        """Test cleanup_scheduler_resources cleans up slices and namespace."""
        scheduler = KubernetesScheduler()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "configs")
            os.makedirs(config_dir, exist_ok=True)
            scheduler.workspace_config_dir = config_dir

            scheduler.cleanup_scheduler_resources()

            # No slices exist, so delete_custom_objects receives []; the
            # namespace delete takes the secrets with it.
            self.assertEqual(
                self.kube.calls,
                [
                    _LIST_CHS_SLICES,
                    Call(
                        "delete_custom_objects",
                        {
                            "group": "accelerator.gke.io",
                            "version": "v1beta1",
                            "plural": "slices",
                            "object_names": [],
                            "wait_timeout_seconds": 300,
                        },
                    ),
                    _DELETE_CHS_NAMESPACE,
                ],
            )

    def test_cleanup_scheduler_resources_file_backstop(self, mock_logger):
        """Test cleanup_scheduler_resources cleans up statelessly."""
        scheduler = KubernetesScheduler()
        scheduler.workspace_config_dir = None

        scheduler.cleanup_scheduler_resources()

        self.assertIn(_DELETE_CHS_NAMESPACE, self.kube.calls)

    def test_cancel_jobs_deletes_secrets(self, mock_logger):
        """Test cancel_jobs deletes only cancelled jobs' -ssh secrets."""
        for name in ("chs-job1", "chs-job2", "chs-job3"):
            self.kube.add_object(
                "batch",
                "jobs",
                {"metadata": {"name": name}},
                namespace=CHS_NAMESPACE,
            )
        for name in (
            "chs-job1-ssh",
            "chs-job2-ssh",
            "chs-job3-ssh",
            "other-secret",
        ):
            self.kube.add_object(
                "",
                "secrets",
                {"metadata": {"name": name}},
                namespace=CHS_NAMESPACE,
            )
        scheduler = KubernetesScheduler()

        scheduler.cancel_jobs(["chs-job1", "chs-job2"])

        self.assertEqual(
            self.kube.calls,
            [
                Call(
                    "delete_jobs",
                    {
                        "object_names": ["chs-job1", "chs-job2"],
                        "namespace": CHS_NAMESPACE,
                        "wait_timeout_seconds": DELETE_WAIT_TIMEOUT_SEC,
                    },
                ),
                Call("list_secrets", {"namespace": CHS_NAMESPACE}),
                Call(
                    "delete_secret",
                    {"name": "chs-job1-ssh", "namespace": CHS_NAMESPACE},
                ),
                Call(
                    "delete_secret",
                    {"name": "chs-job2-ssh", "namespace": CHS_NAMESPACE},
                ),
            ],
        )
        self.assertEqual(self.kube.stored_names("batch", "jobs"), ["chs-job3"])
        stored_secrets = self.kube.stored_names("", "secrets")
        self.assertIn("chs-job3-ssh", stored_secrets)
        self.assertIn("other-secret", stored_secrets)
        self.assertNotIn("chs-job1-ssh", stored_secrets)
        self.assertNotIn("chs-job2-ssh", stored_secrets)

    def test_cancel_jobs_without_kubeconfig_is_logged(self, mock_logger):
        error = k8s_client.KubeError("no kubeconfig")
        install_fake_client(self, error=error)
        scheduler = KubernetesScheduler()
        # Cancellation is best effort: the error is logged, not raised.
        scheduler.cancel_jobs(["job1"])
        mock_logger.error.assert_any_call(
            "Encountered an error while attempting to "
            "delete active Jobs. Error: %s",
            error,
        )

    def test_cleanup_scheduler_resources_deletes_secrets(self, mock_logger):
        """Test that cleanup_scheduler_resources deletes secrets when the
        namespace, which would delete them too, cannot be deleted."""
        forbidden = k8s_client.KubeError("forbidden", status=403)
        self.kube.fail_next("delete_namespace", forbidden)
        self.kube.add_object(
            "", "namespaces", {"metadata": {"name": CHS_NAMESPACE}}
        )
        self.kube.add_object(
            "",
            "secrets",
            {"metadata": {"name": "chs-job1-ssh"}},
            namespace=CHS_NAMESPACE,
        )
        scheduler = KubernetesScheduler()

        scheduler.cleanup_scheduler_resources()

        self.assertIn(
            _DELETE_CHS_NAMESPACE,
            self.kube.calls,
        )
        mock_logger.warning.assert_any_call(
            "Failed to delete Kubernetes namespace. This may "
            "leave orphaned resources. Error: %s",
            forbidden,
        )
        self.assertEqual(
            self.kube.calls_to("delete_secret"),
            [
                Call(
                    "delete_secret",
                    {"name": "chs-job1-ssh", "namespace": CHS_NAMESPACE},
                )
            ],
        )
        self.assertEqual(self.kube.stored_names("", "secrets"), [])

    def test_resolve_project_id_kubernetes_success(self, mock_logger):
        self.kube.add_object(
            "",
            "nodes",
            _provider_node("node1", "gce://my-project-id/us-central1-a/node1"),
        )
        scheduler = KubernetesScheduler()
        project_id = scheduler._resolve_project_id("node1")
        self.assertEqual(project_id, "my-project-id")
        self.assertEqual(self.kube.calls, [Call("get_node", {"name": "node1"})])

    def test_resolve_project_id_kubernetes_uses_cached_node(self, mock_logger):
        # A node object cached by node discovery already has the providerID.
        scheduler = KubernetesScheduler()
        scheduler._resolved_node_resources = [
            _provider_node("node1", "gce://cached-project/us-central1-a/node1")
        ]
        self.assertEqual(
            scheduler._resolve_project_id("node1"), "cached-project"
        )
        self.assertEqual(self.kube.calls, [])

    def test_resolve_project_id_kubernetes_invalid_format(self, mock_logger):
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
            with self.subTest(provider_id=fmt):
                self.kube.add_object("", "nodes", _provider_node("node1", fmt))
                with self.assertRaisesRegex(
                    ex.ChsSchedulerException, "Unexpected providerID format"
                ):
                    scheduler._resolve_project_id("node1")
        with self.subTest(provider_id=None):
            self.kube.add_object("", "nodes", {"metadata": {"name": "node2"}})
            with self.assertRaisesRegex(
                ex.ChsSchedulerException, "Unexpected providerID format"
            ):
                scheduler._resolve_project_id("node2")

    def test_resolve_project_id_kubernetes_error(self, mock_logger):
        with self.subTest("node not found"):
            scheduler = KubernetesScheduler()
            with self.assertRaisesRegex(
                ex.ChsSchedulerException,
                "Failed to resolve project ID from node 'node1': .*"
                'nodes "node1" not found',
            ) as cm:
                scheduler._resolve_project_id("node1")
            self.assertIsInstance(cm.exception.__cause__, k8s_client.KubeError)
            self.assertEqual(cm.exception.__cause__.status, 404)
        with self.subTest("API error"):
            error = k8s_client.KubeError("internal error", status=500)
            self.kube.fail_next("get_node", error)
            with self.assertRaisesRegex(
                ex.ChsSchedulerException, "Failed to resolve project ID"
            ) as cm:
                scheduler._resolve_project_id("node1")
            self.assertIs(cm.exception.__cause__, error)
        with self.subTest("no kubeconfig"):
            error = k8s_client.KubeError("no kubeconfig")
            self.kube.fail_next("get_node", error)
            with self.assertRaisesRegex(
                ex.ChsSchedulerException, "Failed to resolve project ID"
            ) as cm:
                scheduler._resolve_project_id("node1")
            self.assertIs(cm.exception.__cause__, error)

    @mock.patch("chs.schedulers.base.query_gce_instances_info")
    def test_detect_nodes_hardware_success(self, mock_query_info, mock_logger):
        scheduler = KubernetesScheduler()
        # Node objects without GKE's instance-type label and instance_id
        # annotation send every node to the GCE API.
        scheduler._resolved_node_resources = [
            {"metadata": {"name": "node1"}},
            {"metadata": {"name": "node2"}},
        ]
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
        # Node objects without GKE's instance-type label and instance_id
        # annotation send every node to the GCE API.
        scheduler._resolved_node_resources = [
            {"metadata": {"name": "node1"}},
            {"metadata": {"name": "node2"}},
        ]
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
        # Node objects without GKE's instance-type label and instance_id
        # annotation send every node to the GCE API.
        scheduler._resolved_node_resources = [
            {"metadata": {"name": "node1"}},
            {"metadata": {"name": "node2"}},
        ]
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

    @mock.patch("chs.schedulers.base.query_gce_instances_info")
    def test_detect_nodes_hardware_fallback_to_labels(
        self, mock_query_info, mock_logger
    ):
        scheduler = KubernetesScheduler()
        self.kube.add_object("", "nodes", {"metadata": {"name": "node1"}})
        self.kube.add_object(
            "",
            "nodes",
            {
                "metadata": {
                    "name": "node2",
                    "labels": {
                        "node.kubernetes.io/instance-type": (
                            "a5x-highgpu-4g-metal"
                        ),
                    },
                }
            },
        )
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
            self.assertEqual(
                machine_types, ["a3-megagpu-8g", "a5x-highgpu-4g-metal"]
            )
            self.assertEqual(
                scheduler.node_to_instance_id,
                {"node1": "12345", "node2": "node2"},
            )
        self.assertEqual(
            self.kube.calls,
            [
                Call(
                    "list_nodes",
                    {"label_selector": "cloud.google.com/gke-nodepool!=system"},
                )
            ],
        )

    @mock.patch(
        "builtins.open",
        new_callable=mock.mock_open,
        read_data=(
            "apiVersion: batch/v1\n"
            "kind: Job\n"
            "metadata:\n"
            "  name: my-job"
        ),
    )
    def test_extract_job_id_success(self, mock_open, _mock_logger):
        scheduler = KubernetesScheduler()
        job_id = scheduler.extract_job_id("/tmp/exp")
        self.assertEqual(job_id, "my-job")
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
    def test_extract_job_id_no_job(self, mock_open, _mock_logger):
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

    def test_cleanup_dynamic_slices(self, mock_logger):
        self.kube.add_object(
            "accelerator.gke.io", "slices", _chs_slice("pool1", "Ready")
        )
        self.kube.add_object(
            "accelerator.gke.io",
            "slices",
            {"metadata": {"name": "user-slice"}},
        )
        scheduler = KubernetesScheduler()
        scheduler._cleanup_dynamic_slices()
        self.assertEqual(
            self.kube.calls,
            [
                _LIST_CHS_SLICES,
                Call(
                    "delete_custom_objects",
                    {
                        "group": "accelerator.gke.io",
                        "version": "v1beta1",
                        "plural": "slices",
                        "object_names": ["chs-slice-pool1"],
                        "wait_timeout_seconds": 300,
                    },
                ),
            ],
        )
        # Only the slices CHS created are deleted.
        self.assertEqual(
            self.kube.stored_names("accelerator.gke.io", "slices"),
            ["user-slice"],
        )
        mock_logger.warning.assert_not_called()

    def test_cleanup_dynamic_slices_crd_not_found(self, mock_logger):
        # Without the Slice CRD there is nothing to delete, and a list
        # failing with 404 is silently ignored.
        self.kube.unregister_type("accelerator.gke.io", "slices")
        scheduler = KubernetesScheduler()
        scheduler._cleanup_dynamic_slices()
        self.assertEqual(self.kube.calls, [_LIST_CHS_SLICES])
        mock_logger.warning.assert_not_called()

    def test_cleanup_dynamic_slices_failure_is_logged(self, mock_logger):
        error = k8s_client.KubeError("timed out deleting slices")
        self.kube.fail_next("delete_custom_objects", error)
        scheduler = KubernetesScheduler()
        # Cleanup is best effort: the error is logged, not raised.
        scheduler._cleanup_dynamic_slices()
        mock_logger.warning.assert_called_once_with(
            "Failed to delete dynamic TPU slices after execution. "
            "Manual cleanup may be required. Error: %s",
            error,
        )

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._delete_namespace"
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler."
        "_cleanup_dynamic_slices"
    )
    def test_cancel_jobs_by_prefix_does_not_delete_namespace(
        self,
        mock_cleanup_slices,
        mock_delete_ns,
        mock_logger,
    ):
        for name in ("chs-test-job1", "chs-test-job2", "other-job"):
            self.kube.add_object(
                "batch",
                "jobs",
                {"metadata": {"name": name}},
                namespace=CHS_NAMESPACE,
            )
        scheduler = KubernetesScheduler()
        with mock.patch.object(scheduler, "cancel_jobs") as mock_cancel_jobs:
            scheduler.cancel_jobs_by_prefix("chs-test")
            mock_cancel_jobs.assert_called_once_with(
                ["chs-test-job1", "chs-test-job2"]
            )
            mock_delete_ns.assert_not_called()
            mock_cleanup_slices.assert_not_called()
        self.assertEqual(
            self.kube.calls,
            [Call("list_jobs", {"namespace": CHS_NAMESPACE})],
        )

    def test_get_active_job_names_timeout_raises(self, mock_logger):
        """A timeout must not look like "no jobs" to the foreign-run check."""
        self.kube.fail_next("list_jobs", k8s_client.KubeError("read timed out"))
        with self.assertRaisesRegex(
            ex.ChsSchedulerException, "Failed to list active Jobs"
        ):
            KubernetesScheduler().get_active_job_names()

    def test_get_active_job_names_api_error_returns_empty(self, mock_logger):
        self.kube.fail_next(
            "list_jobs",
            k8s_client.KubeError("forbidden", status=403),
        )
        self.assertEqual(KubernetesScheduler().get_active_job_names(), [])

    def test_get_active_job_names_excludes_finished_jobs(self, mock_logger):
        for name, status in (
            ("chs-running", {"active": 1}),
            ("chs-pending", {}),
            (
                "chs-complete",
                {"conditions": [{"type": "Complete", "status": "True"}]},
            ),
            (
                "chs-failed",
                {"conditions": [{"type": "Failed", "status": "True"}]},
            ),
        ):
            self.kube.add_object(
                "batch",
                "jobs",
                {"metadata": {"name": name}, "status": status},
                namespace=CHS_NAMESPACE,
            )
        self.assertEqual(
            KubernetesScheduler().get_active_job_names(),
            ["chs-running", "chs-pending"],
        )

    def test_get_job_statuses_from_job_conditions(self, mock_logger):
        """Job conditions and counts map to CHS job statuses."""
        cases = {
            "done": {"conditions": [{"type": "Complete", "status": "True"}]},
            "timed-out": {
                "failed": 1,
                "conditions": [
                    {
                        "type": "Failed",
                        "status": "True",
                        "reason": "DeadlineExceeded",
                    }
                ],
            },
            "failed": {
                "failed": 1,
                "conditions": [
                    {
                        "type": "Failed",
                        "status": "True",
                        "reason": "BackoffLimitExceeded",
                    }
                ],
            },
            "pod-failed-with-active-siblings": {
                "active": 1,
                "failed": 1,
            },
            "finishing": {
                "active": 1,
                "conditions": [
                    {"type": "SuccessCriteriaMet", "status": "True"}
                ],
            },
            "running": {"active": 2},
            "pending": {},
        }
        for name, status in cases.items():
            self.kube.add_object(
                "batch",
                "jobs",
                {"metadata": {"name": name}, "status": status},
                namespace=CHS_NAMESPACE,
            )

        statuses = KubernetesScheduler().get_job_statuses([*cases, "missing"])

        self.assertEqual(
            statuses,
            {
                "done": "COMPLETED",
                "timed-out": "TIMEOUT",
                "failed": "UNKNOWN",
                "pod-failed-with-active-siblings": "RUNNING",
                "finishing": "RUNNING",
                "running": "RUNNING",
                "pending": "PENDING",
                "missing": "UNKNOWN",
            },
        )
        self.assertEqual(
            self.kube.calls, [Call("list_jobs", {"namespace": CHS_NAMESPACE})]
        )

    def test_ensure_namespace_applies_worker_service(self, mock_logger):
        KubernetesScheduler()._ensure_namespace_exists()

        service = self.kube.objects[("", "services", CHS_NAMESPACE, "chs")]
        self.assertEqual(
            service["spec"],
            {
                "clusterIP": "None",
                "publishNotReadyAddresses": True,
                "selector": {"chs.gke.io/worker": "true"},
            },
        )
        KubernetesScheduler()._ensure_namespace_exists()
        self.assertEqual(len(self.kube.calls_to("apply_service")), 2)

    def test_fetch_job_log_selects_pods_by_job_name(self, mock_logger):
        for index in ("0", "1"):
            self.kube.add_object(
                "",
                "pods",
                {
                    "metadata": {
                        "name": f"job1-{index}",
                        "labels": {
                            "batch.kubernetes.io/job-name": "job1",
                            "batch.kubernetes.io/job-completion-index": index,
                        },
                    },
                    "spec": {"containers": [{"name": "c"}]},
                },
                namespace=CHS_NAMESPACE,
            )
        scheduler = KubernetesScheduler()

        self.assertEqual(
            [p["metadata"]["name"] for p in scheduler._get_pods("job1")],
            ["job1-0", "job1-1"],
        )
        self.assertEqual(
            [
                p["metadata"]["name"]
                for p in scheduler._get_pods("job1", head_pod_only=True)
            ],
            ["job1-0"],
        )
        self.assertEqual(
            self.kube.calls_to("list_pods")[1].kwargs["label_selector"],
            "batch.kubernetes.io/job-name=job1,"
            "batch.kubernetes.io/job-completion-index=0",
        )

    @mock.patch("time.sleep")
    @mock.patch("time.time", side_effect=[0, 1])
    def test_wait_for_slices_active_success(
        self, mock_time, mock_sleep, mock_logger
    ):
        scheduler = KubernetesScheduler()
        self.kube.add_object(
            "accelerator.gke.io", "slices", _chs_slice("mypool", "ACTIVE")
        )
        success, failed = scheduler._wait_for_slices_active(["mypool"])
        self.assertEqual(self.kube.calls, [_LIST_CHS_SLICES])
        self.assertEqual(success, ["mypool"])
        self.assertEqual(failed, [])
        mock_sleep.assert_not_called()

    @mock.patch("time.sleep")
    @mock.patch("time.time", side_effect=[0, 1])
    def test_wait_for_slices_active_failed(
        self, mock_time, mock_sleep, mock_logger
    ):
        scheduler = KubernetesScheduler()
        self.kube.add_object(
            "accelerator.gke.io", "slices", _chs_slice("mypool", "FAILED")
        )
        success, failed = scheduler._wait_for_slices_active(["mypool"])
        self.assertEqual(self.kube.calls, [_LIST_CHS_SLICES])
        self.assertEqual(success, [])
        self.assertEqual(failed, ["mypool"])
        mock_sleep.assert_not_called()

    @mock.patch("time.sleep")
    @mock.patch("time.time", side_effect=[0, 10, 310])
    def test_wait_for_slices_active_timeout(
        self, mock_time, mock_sleep, mock_logger
    ):
        scheduler = KubernetesScheduler()
        self.kube.add_object(
            "accelerator.gke.io", "slices", _chs_slice("mypool", "CREATING")
        )
        success, failed = scheduler._wait_for_slices_active(
            ["mypool"], timeout_sec=300
        )
        self.assertEqual(success, [])
        self.assertEqual(failed, ["mypool"])
        self.assertEqual(self.kube.calls, [_LIST_CHS_SLICES])
        mock_sleep.assert_called_once_with(10)

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._ensure_namespace_exists"
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._provision_dynamic_slices"
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler.get_nodes_under_test",
        return_value=["node1"],
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._group_nodes_by_slice",
        return_value={"mypool": ["node1"]},
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._cleanup_dynamic_slices"
    )
    def test_apply_scheduler_resources_failed_nodepools(
        self,
        mock_cleanup,
        mock_group,
        mock_get_nodes,
        mock_provision,
        mock_ensure_ns,
        mock_logger,
    ):
        scheduler = KubernetesScheduler()
        scheduler._resolved_nodes = ["node1"]

        scheduler._tpu_slice_data = TpuSliceData(
            pools={"mypool": ["node1"]},
            node_to_pool={"node1": "mypool"},
            pool_to_topology={"mypool": "2x2"},
            dynamic_slice_partitions={"mypool": "mypool-partition"},
            pool_to_accelerator={"mypool": "tpu-v4-podslice"},
        )
        mock_provision.return_value = ([], ["mypool"])
        with self.assertRaises(ChsSchedulerException):
            scheduler.apply_scheduler_resources()

        self.assertEqual(scheduler._resolved_nodes, [])
        self.assertNotIn("mypool", scheduler._tpu_slice_data.pools)
        self.assertNotIn("mypool", scheduler._tpu_slice_data.pool_to_topology)
        self.assertNotIn("node1", scheduler._tpu_slice_data.node_to_pool)

    def test_get_already_fulfilled_partitions_success(self, mock_logger):
        scheduler = KubernetesScheduler()
        for slice_obj in [
            {
                "metadata": {"name": "slice1"},
                "spec": {"partitionIds": ["part-1"]},
            },
            {
                "metadata": {
                    "name": "slice2",
                    "deletionTimestamp": "2023-10-01T00:00:00Z",
                },
                "spec": {"partitionIds": ["part-2"]},
            },
            {
                "metadata": {"name": "slice3"},
                "spec": {"partitionIds": ["part-3", "part-4"]},
            },
        ]:
            self.kube.add_object("accelerator.gke.io", "slices", slice_obj)
        fulfilled = scheduler._get_already_fulfilled_partitions()
        self.assertCountEqual(fulfilled, ["part-1", "part-3", "part-4"])
        self.assertEqual(
            self.kube.calls,
            [
                Call(
                    "list_custom_objects",
                    {
                        "group": "accelerator.gke.io",
                        "version": "v1beta1",
                        "plural": "slices",
                    },
                )
            ],
        )

    def test_get_already_fulfilled_partitions_error(self, mock_logger):
        scheduler = KubernetesScheduler()
        self.kube.add_object(
            "accelerator.gke.io",
            "slices",
            {
                "metadata": {"name": "slice1"},
                "spec": {"partitionIds": ["part-1"]},
            },
        )
        with self.subTest("request fails"):
            self.kube.fail_next(
                "list_custom_objects", k8s_client.KubeError("error")
            )
            fulfilled = scheduler._get_already_fulfilled_partitions()
            self.assertEqual(fulfilled, set())
        with self.subTest("Slice CRD not installed"):
            self.kube.unregister_type("accelerator.gke.io", "slices")
            fulfilled = scheduler._get_already_fulfilled_partitions()
            self.assertEqual(fulfilled, set())

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_already_fulfilled_partitions"
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._wait_for_slices_active"
    )
    def test_provision_dynamic_slices_all_fulfilled(
        self, mock_wait, mock_get_fulfilled, mock_logger
    ):
        scheduler = KubernetesScheduler()
        target_pools = {"pool1": ["node1"]}

        scheduler.workspace_config_dir = "/tmp"
        scheduler._tpu_slice_data = TpuSliceData(
            pools=target_pools,
            node_to_pool={"node1": "pool1"},
            pool_to_topology={"pool1": "2x2"},
            dynamic_slice_partitions={"pool1": "part1"},
            pool_to_accelerator={"pool1": "tpu-v4-podslice"},
        )
        mock_get_fulfilled.return_value = {"part1"}
        success, failed = scheduler._provision_dynamic_slices(target_pools)
        self.assertEqual(success, ["pool1"])
        self.assertEqual(failed, [])
        # Nothing is applied.
        self.assertEqual(self.kube.calls, [])
        mock_wait.assert_not_called()

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_already_fulfilled_partitions"
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._wait_for_slices_active"
    )
    def test_provision_dynamic_slices_none_fulfilled(
        self, mock_wait, mock_get_fulfilled, mock_logger
    ):
        scheduler = KubernetesScheduler()
        target_pools = {"pool1": ["node1"]}

        scheduler.workspace_config_dir = self.tmp_dir
        scheduler._tpu_slice_data = TpuSliceData(
            pools=target_pools,
            node_to_pool={"node1": "pool1"},
            pool_to_topology={"pool1": "2x2"},
            dynamic_slice_partitions={"pool1": "part1"},
            pool_to_accelerator={"pool1": "tpu-v4-podslice"},
        )
        mock_get_fulfilled.return_value = set()
        mock_wait.return_value = (["pool1"], [])
        success, failed = scheduler._provision_dynamic_slices(target_pools)
        self.assertEqual(success, ["pool1"])
        self.assertEqual(failed, [])
        applied = self.kube.objects[
            ("accelerator.gke.io", "slices", None, "chs-slice-pool1")
        ]
        self.assertEqual(len(self.kube.calls), 1)
        self.assertEqual(self.kube.calls[0].method, "apply_custom_object")
        self.assertEqual(self.kube.calls[0].kwargs["plural"], "slices")
        self.assertEqual(
            applied["metadata"]["labels"],
            {"created-by": "chs", "chs-nodepool": "pool1"},
        )
        self.assertEqual(applied["spec"]["partitionIds"], ["part1"])
        mock_wait.assert_called_once_with(["pool1"], 300)

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._fetch_raw_node_resources"
    )
    def test__get_tpu_slices_provision_only_healthy(
        self, mock_fetch, mock_logger
    ):
        """Test topology parsing for healthy PROVISION_ONLY nodepools."""
        mock_fetch.return_value = [
            {
                "metadata": {
                    "name": "node1",
                    "labels": {
                        "cloud.google.com/gke-tpu-accelerator": "tpu-v4",
                        "cloud.google.com/gke-nodepool": "pool1",
                        (
                            "cloud.google.com/" "gke-accelerator-topology-mode"
                        ): "PROVISION_ONLY",
                        (
                            f"cloud.google.com/gke-tpu-partition-"
                            f"{SUPPORTED_DYNAMIC_SLICE_TOPOLOGY}-state"
                        ): ("HEALTHY"),
                        (
                            f"cloud.google.com/gke-tpu-partition-"
                            f"{SUPPORTED_DYNAMIC_SLICE_TOPOLOGY}-id"
                        ): ("partition1"),
                    },
                }
            }
        ]

        scheduler = KubernetesScheduler()
        res = scheduler._get_tpu_slices()

        self.assertEqual(res.pools["pool1"], ["node1"])
        self.assertEqual(res.node_to_pool["node1"], "pool1")
        self.assertEqual(
            res.pool_to_topology["pool1"], SUPPORTED_DYNAMIC_SLICE_TOPOLOGY
        )
        self.assertEqual(res.dynamic_slice_partitions["pool1"], "partition1")
        self.assertEqual(res.pool_to_accelerator["pool1"], "tpu-v4")
        mock_logger.warning.assert_not_called()

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._fetch_raw_node_resources"
    )
    def test__get_tpu_slices_provision_only_unhealthy(
        self, mock_fetch, mock_logger
    ):
        """Test topology parsing skips unhealthy PROVISION_ONLY nodepools."""
        mock_fetch.return_value = [
            {
                "metadata": {
                    "name": "node1",
                    "labels": {
                        "cloud.google.com/gke-tpu-accelerator": "tpu-v4",
                        "cloud.google.com/gke-nodepool": "pool1",
                        (
                            "cloud.google.com/" "gke-accelerator-topology-mode"
                        ): "PROVISION_ONLY",
                        (
                            f"cloud.google.com/gke-tpu-partition-"
                            f"{SUPPORTED_DYNAMIC_SLICE_TOPOLOGY}-state"
                        ): ("UNHEALTHY"),
                    },
                }
            }
        ]

        scheduler = KubernetesScheduler()
        res = scheduler._get_tpu_slices()

        self.assertNotIn("pool1", res.pools)
        mock_logger.warning.assert_any_call(
            (
                "Skipping nodepool '%s' because the %s "
                "topology is currently unhealthy."
            ),
            "pool1",
            SUPPORTED_DYNAMIC_SLICE_TOPOLOGY,
        )

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._fetch_raw_node_resources"
    )
    def test__get_tpu_slices_provision_only_unsupported(
        self, mock_fetch, mock_logger
    ):
        """Test topology parsing skips unsupported PROVISION_ONLY nodepools."""
        mock_fetch.return_value = [
            {
                "metadata": {
                    "name": "node1",
                    "labels": {
                        "cloud.google.com/gke-tpu-accelerator": "tpu-v4",
                        "cloud.google.com/gke-nodepool": "pool1",
                        (
                            "cloud.google.com/" "gke-accelerator-topology-mode"
                        ): "PROVISION_ONLY",
                        # Missing the topology state label completely
                    },
                }
            }
        ]

        scheduler = KubernetesScheduler()
        res = scheduler._get_tpu_slices()

        self.assertNotIn("pool1", res.pools)
        mock_logger.warning.assert_any_call(
            (
                "Skipping nodepool '%s' because the %s "
                "topology is unsupported."
            ),
            "pool1",
            SUPPORTED_DYNAMIC_SLICE_TOPOLOGY,
        )

    def test_poll_jobs_starts_planned_jobs_from_one_list(self, mock_logger):
        """Test poll_jobs starts free Jobs and returns their old status."""
        for name, node in (("job1", "node1"), ("job2", "node2")):
            self.kube.add_object(
                "batch",
                "jobs",
                _job_object(name, [node], suspended=True),
                namespace=CHS_NAMESPACE,
            )
        self.kube.add_object(
            "batch",
            "jobs",
            _job_object("job3", ["node1"], suspended=True),
            namespace=CHS_NAMESPACE,
        )
        scheduler = KubernetesScheduler()

        statuses = scheduler.poll_jobs(["job1", "job2", "job3"])

        self.assertEqual(
            statuses, {"job1": "PENDING", "job2": "PENDING", "job3": "PENDING"}
        )
        self.assertEqual(len(self.kube.calls_to("list_jobs")), 1)
        patches = self.kube.calls_to("patch_job")
        self.assertEqual(
            sorted(call.kwargs["name"] for call in patches), ["job1", "job2"]
        )
        for call in patches:
            self.assertEqual(call.kwargs["body"], {"spec": {"suspend": False}})
            self.assertEqual(call.kwargs["namespace"], CHS_NAMESPACE)

    def test_poll_jobs_starts_jobs_in_queue_order(self, mock_logger):
        """Test the queue, not creation time, decides who starts first."""
        self.kube.add_object(
            "batch",
            "jobs",
            _job_object(
                "job-a", ["node1"], creation_timestamp="2026-10-07T00:00:00Z"
            ),
            namespace=CHS_NAMESPACE,
        )
        self.kube.add_object(
            "batch",
            "jobs",
            _job_object(
                "job-b", ["node1"], creation_timestamp="2026-10-07T00:00:05Z"
            ),
            namespace=CHS_NAMESPACE,
        )
        scheduler = KubernetesScheduler()

        scheduler.poll_jobs(["job-b", "job-a"])

        patches = self.kube.calls_to("patch_job")
        self.assertEqual([call.kwargs["name"] for call in patches], ["job-b"])

    def test_poll_jobs_continues_after_failed_start(self, mock_logger):
        """Test a failed start is logged and the other Jobs still start."""
        for name, node in (("job1", "node1"), ("job2", "node2")):
            self.kube.add_object(
                "batch",
                "jobs",
                _job_object(name, [node], suspended=True),
                namespace=CHS_NAMESPACE,
            )
        self.kube.fail_next(
            "patch_job",
            k8s_client.KubeError("Conflict", status=409),
        )
        scheduler = KubernetesScheduler()

        scheduler.poll_jobs(["job1", "job2"])

        self.assertEqual(len(self.kube.calls_to("patch_job")), 2)
        mock_logger.warning.assert_called_once()

    def test_poll_jobs_raises_if_listing_fails(self, mock_logger):
        """Test poll_jobs raises instead of reporting every job UNKNOWN."""
        self.kube.fail_next(
            "list_jobs",
            k8s_client.KubeError("Forbidden", status=403),
        )
        scheduler = KubernetesScheduler()

        with self.assertRaisesRegex(ChsSchedulerException, "Forbidden"):
            scheduler.poll_jobs(["job1"])

        self.assertEqual(self.kube.calls_to("patch_job"), [])

    def test_poll_jobs_empty_job_ids(self, mock_logger):
        """Test poll_jobs returns immediately when job_ids is empty."""
        scheduler = KubernetesScheduler()
        self.assertEqual(scheduler.poll_jobs([]), {})
        self.assertEqual(self.kube.calls, [])

    def _write_experiment(self, name: str, *objects: dict) -> str:
        experiment_dir = os.path.join(self.tmp_dir, "experiments", name)
        os.makedirs(experiment_dir)
        with open(os.path.join(experiment_dir, "jobset.yaml"), "w") as f:
            yaml.safe_dump_all(objects, f)
        return experiment_dir

    def test_submit_experiments_starts_first_wave_unsuspended(
        self, mock_logger
    ):
        """Test free Jobs are created running and first, the rest wait."""
        secret = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": "job1-ssh", "namespace": CHS_NAMESPACE},
            "stringData": {"ssh-privatekey": "key"},
        }
        compute_domain = {
            "apiVersion": "resource.nvidia.com/v1beta1",
            "kind": "ComputeDomain",
            "metadata": {"name": "job4-cd", "namespace": CHS_NAMESPACE},
            "spec": {"numNodes": 1},
        }
        experiment_dirs = [
            self._write_experiment(
                "exp1", secret, _job_manifest("job1", "node1")
            ),
            self._write_experiment("exp2", _job_manifest("job2", "node2")),
            self._write_experiment("exp3", _job_manifest("job3", "node1")),
            self._write_experiment(
                "exp4", compute_domain, _job_manifest("job4", "node3")
            ),
        ]
        scheduler = KubernetesScheduler()

        with mock.patch("chs.schedulers.kubernetes._API_MAX_CONCURRENCY", 1):
            job_to_dir = scheduler.submit_experiments(experiment_dirs)

        self.assertEqual(
            list(job_to_dir.items()),
            list(zip(["job1", "job2", "job3", "job4"], experiment_dirs)),
        )
        applied = [
            call.kwargs["body"]["metadata"]["name"]
            for call in self.kube.calls
            if call.method
            in ("apply_secret", "apply_job", "apply_custom_object")
        ]
        self.assertEqual(
            applied, ["job1-ssh", "job1", "job2", "job4-cd", "job4", "job3"]
        )
        suspended = {
            name: self.kube.objects[("batch", "jobs", CHS_NAMESPACE, name)][
                "spec"
            ]["suspend"]
            for name in ("job1", "job2", "job3", "job4")
        }
        self.assertEqual(
            suspended,
            {"job1": False, "job2": False, "job3": True, "job4": False},
        )
        self.assertIn(
            ("", "secrets", CHS_NAMESPACE, "job1-ssh"), self.kube.objects
        )
        self.assertIn(
            ("resource.nvidia.com", "computedomains", CHS_NAMESPACE, "job4-cd"),
            self.kube.objects,
        )

    def test_submit_experiments_waits_for_running_jobs(self, mock_logger):
        """Test Jobs already running in the namespace keep their nodes."""
        self.kube.add_object(
            "batch",
            "jobs",
            _job_object("running", ["node1"], suspended=False),
            namespace=CHS_NAMESPACE,
        )
        experiment_dir = self._write_experiment(
            "exp1", _job_manifest("job1", "node1")
        )
        scheduler = KubernetesScheduler()

        scheduler.submit_experiments([experiment_dir])

        stored = self.kube.objects[("batch", "jobs", CHS_NAMESPACE, "job1")]
        self.assertTrue(stored["spec"]["suspend"])

    def test_submit_experiments_rejects_unsupported_kind(self, mock_logger):
        """Test an unknown kind fails before anything is created."""
        config_map = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "cm", "namespace": CHS_NAMESPACE},
        }
        experiment_dirs = [
            self._write_experiment("exp1", _job_manifest("job1", "node1")),
            self._write_experiment(
                "exp2", config_map, _job_manifest("job2", "node2")
            ),
        ]
        scheduler = KubernetesScheduler()

        with self.assertRaisesRegex(ChsSchedulerException, "ConfigMap"):
            scheduler.submit_experiments(experiment_dirs)

        self.assertEqual(self.kube.calls, [])

    def test_submit_experiments_names_failed_experiment(self, mock_logger):
        """Test a failed create raises with the experiment's name."""
        experiment_dirs = [
            self._write_experiment("exp1", _job_manifest("job1", "node1")),
            self._write_experiment("exp2", _job_manifest("job2", "node2")),
        ]
        self.kube.fail_next(
            "apply_job",
            k8s_client.KubeError("admission webhook denied", status=400),
        )
        scheduler = KubernetesScheduler()

        with (
            mock.patch("chs.schedulers.kubernetes._API_MAX_CONCURRENCY", 1),
            self.assertRaisesRegex(
                ChsSchedulerException, "exp1.*admission webhook denied"
            ),
        ):
            scheduler.submit_experiments(experiment_dirs)

    @mock.patch("time.sleep")
    def test_submit_experiments_retries_transient_errors(
        self, mock_sleep, mock_logger
    ):
        """Test throttled or dropped creates are retried with backoff."""
        experiment_dir = self._write_experiment(
            "exp1", _job_manifest("job1", "node1")
        )
        self.kube.fail_next(
            "apply_job",
            k8s_client.KubeError("Too Many Requests", status=429),
            k8s_client.KubeError("Kubernetes API request failed: reset"),
        )
        scheduler = KubernetesScheduler()

        scheduler.submit_experiments([experiment_dir])

        self.assertEqual(len(self.kube.calls_to("apply_job")), 3)
        self.assertEqual(
            [call.args[0] for call in mock_sleep.call_args_list], [1, 2]
        )
        self.assertIn(
            ("batch", "jobs", CHS_NAMESPACE, "job1"),
            self.kube.objects,
        )

    @mock.patch("time.sleep")
    def test_submit_experiments_gives_up_after_repeated_errors(
        self, mock_sleep, mock_logger
    ):
        """Test a create that keeps failing raises after five tries."""
        experiment_dir = self._write_experiment(
            "exp1", _job_manifest("job1", "node1")
        )
        self.kube.fail_always(
            "apply_job",
            k8s_client.KubeError("Service Unavailable", status=503),
        )
        scheduler = KubernetesScheduler()

        with self.assertRaisesRegex(
            ChsSchedulerException, "exp1.*Service Unavailable"
        ):
            scheduler.submit_experiments([experiment_dir])

        self.assertEqual(len(self.kube.calls_to("apply_job")), 5)
        self.assertEqual(
            [call.args[0] for call in mock_sleep.call_args_list], [1, 2, 4, 8]
        )

    def test_submit_experiments_empty(self, mock_logger):
        """Test submitting nothing makes no API calls."""
        scheduler = KubernetesScheduler()
        self.assertEqual(scheduler.submit_experiments([]), {})
        self.assertEqual(self.kube.calls, [])

    def test_get_unschedulable_jobs(self, mock_logger):
        """Test get_unschedulable_jobs returns only matching timed-out jobs."""
        now = datetime.datetime.now(datetime.timezone.utc)
        old_timestamp = (now - datetime.timedelta(seconds=400)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        young_timestamp = (now - datetime.timedelta(seconds=50)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        self.kube.add_object(
            "",
            "pods",
            _pending_pod(
                "job1-pod",
                job_name="job1",
                creation_timestamp=old_timestamp,
                scheduled=False,
                unschedulable=True,
            ),
            namespace=CHS_NAMESPACE,
        )
        self.kube.add_object(
            "",
            "pods",
            _pending_pod(
                "job2-pod",
                job_name="job2",
                creation_timestamp=young_timestamp,
                scheduled=False,
                unschedulable=True,
            ),
            namespace=CHS_NAMESPACE,
        )
        self.kube.add_object(
            "",
            "pods",
            _pending_pod(
                "job3-pod",
                job_name="job3",
                creation_timestamp=old_timestamp,
                scheduled=True,
                unschedulable=False,
            ),
            namespace=CHS_NAMESPACE,
        )
        self.kube.add_object(
            "",
            "pods",
            _pending_pod(
                "job4-pod",
                job_name="job4",
                creation_timestamp=old_timestamp,
                scheduled=False,
                unschedulable=True,
            ),
            namespace=CHS_NAMESPACE,
        )

        scheduler = KubernetesScheduler()
        unschedulable = scheduler.get_unschedulable_jobs(
            ["job1", "job2", "job3"]
        )
        self.assertEqual(unschedulable, ["job1"])

    def test_get_unschedulable_jobs_returns_empty_on_kube_error(
        self, mock_logger
    ):
        """Test get_unschedulable_jobs returns empty list on KubeError."""
        self.kube.fail_next(
            "list_pods",
            k8s_client.KubeError("Forbidden", status=403),
        )
        scheduler = KubernetesScheduler()
        self.assertEqual(scheduler.get_unschedulable_jobs(["job1"]), [])
        mock_logger.warning.assert_called_once()

    def test_get_unschedulable_jobs_empty_job_ids(self, mock_logger):
        """Test get_unschedulable_jobs returns empty list for empty job_ids."""
        scheduler = KubernetesScheduler()
        self.assertEqual(scheduler.get_unschedulable_jobs([]), [])
        self.assertEqual(self.kube.calls, [])


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

    @mock.patch.object(
        k8s_client, "current_context", return_value="test-context"
    )
    @mock.patch("shutil.which")
    @mock.patch("os.path.exists")
    @mock.patch.dict("os.environ", {}, clear=True)
    def test_detect_kubernetes_out_of_cluster(
        self, mock_exists, mock_which, mock_current_context
    ):
        """Test K8s detected out-of-cluster if kubeconfig exists."""
        # k8s_client.current_context() (formerly `kubectl config
        # current-context`) is mocked to return a context.
        # Slurm not found, kubectl found
        mock_which.side_effect = lambda cmd: cmd == "kubectl"

        # kubeconfig exists
        mock_exists.side_effect = lambda p: p == os.path.expanduser(
            "~/.kube/config"
        )

        scheduler = detect_scheduler()
        self.assertEqual(scheduler, "kubernetes")
        mock_current_context.assert_called_once_with()

    @mock.patch.object(k8s_client, "current_context")
    @mock.patch("shutil.which")
    @mock.patch("os.path.exists")
    @mock.patch.dict("os.environ", {}, clear=True)
    def test_detect_kubernetes_missing_kubectl(
        self, mock_exists, mock_which, mock_current_context
    ):
        """Test K8s detection fails fast if kubectl is missing."""
        mock_exists.side_effect = lambda p: p == os.path.expanduser(
            "~/.kube/config"
        )
        mock_which.return_value = False

        with self.assertRaisesRegex(
            ChsSchedulerException, "Missing required executable: kubectl"
        ):
            detect_scheduler()
        # kubectl is checked before the kubeconfig is read.
        mock_current_context.assert_not_called()

    @mock.patch.object(k8s_client, "current_context", return_value=None)
    @mock.patch("shutil.which")
    @mock.patch("os.path.exists")
    @mock.patch.dict("os.environ", {}, clear=True)
    def test_detect_kubernetes_missing_context(
        self, mock_exists, mock_which, mock_current_context
    ):
        """Test K8s detection fails if no active context is set."""
        mock_exists.side_effect = lambda p: p == os.path.expanduser(
            "~/.kube/config"
        )
        mock_which.side_effect = lambda cmd: cmd == "kubectl"

        with self.assertRaisesRegex(
            ChsSchedulerException, "no active kubectl context is set"
        ):
            detect_scheduler()
        mock_current_context.assert_called_once_with()

    @mock.patch.object(
        k8s_client, "current_context", return_value="test-context"
    )
    @mock.patch("shutil.which")
    @mock.patch("os.path.exists")
    @mock.patch.dict("os.environ", {"KUBECONFIG": "/some/path"})
    def test_detect_kubernetes_out_of_cluster_env(
        self, mock_exists, mock_which, mock_current_context
    ):
        """Test K8s detected out-of-cluster if KUBECONFIG env is set and
        file exists.
        """
        # Slurm not found, kubectl found
        mock_which.side_effect = lambda cmd: cmd == "kubectl"
        mock_exists.side_effect = lambda p: p == "/some/path"

        scheduler = detect_scheduler()
        self.assertEqual(scheduler, "kubernetes")
        mock_current_context.assert_called_once_with()

    @mock.patch.object(
        k8s_client, "current_context", return_value="test-context"
    )
    @mock.patch("shutil.which")
    @mock.patch("os.path.exists")
    @mock.patch.dict("os.environ", {"KUBECONFIG": "/path1:/path2"})
    def test_detect_kubernetes_out_of_cluster_env_multi(
        self, mock_exists, mock_which, mock_current_context
    ):
        """Test K8s detected if KUBECONFIG has multiple paths and one exists."""
        # Slurm not found, kubectl found
        mock_which.side_effect = lambda cmd: cmd == "kubectl"
        # /path2 exists, /path1 does not
        mock_exists.side_effect = lambda p: p == "/path2"

        scheduler = detect_scheduler()
        self.assertEqual(scheduler, "kubernetes")
        mock_current_context.assert_called_once_with()

    @mock.patch.object(
        k8s_client, "current_context", return_value="test-context"
    )
    @mock.patch("shutil.which", return_value=None)
    def test_get_kubectl_context_requires_kubectl(
        self, mock_which, mock_current_context
    ):
        """Test no context is reported if kubectl, which Ramble runs, is
        missing."""
        self.assertIsNone(_get_kubectl_context())
        mock_which.assert_called_once_with("kubectl")
        mock_current_context.assert_not_called()

    @mock.patch("subprocess.run")
    @mock.patch("shutil.which", return_value="/usr/bin/kubectl")
    def test_get_kubectl_context_reads_kubeconfig(self, mock_which, mock_run):
        """Test the context is read via k8s_client without kubectl."""
        kube = install_fake_client(self)
        kube.context = "gke_proj_zone_cluster"
        self.assertEqual(_get_kubectl_context(), "gke_proj_zone_cluster")

        kube.context = None
        self.assertIsNone(_get_kubectl_context())
        mock_run.assert_not_called()

    @mock.patch("shutil.which")
    @mock.patch("os.path.exists")
    @mock.patch.dict("os.environ", {}, clear=True)
    def test_detect_kubernetes_current_context_kube_error(
        self, mock_exists, mock_which
    ):
        """Test KubeError from current_context returns None and raises
        no-active-context ChsSchedulerException."""
        mock_exists.side_effect = lambda p: p == os.path.expanduser(
            "~/.kube/config"
        )
        mock_which.side_effect = lambda cmd: cmd == "kubectl"
        kube = install_fake_client(self)
        kube.fail_always(
            "current_context",
            k8s_client.KubeError(
                "Kubernetes configuration could not be loaded: "
                "Expected key current-context in kube-config"
            ),
        )

        self.assertIsNone(_get_kubectl_context())
        with self.assertRaisesRegex(
            ChsSchedulerException, "no active kubectl context is set"
        ):
            detect_scheduler()

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
