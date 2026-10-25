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
import re
import tempfile
from typing import Any
import unittest
import yaml
from unittest import mock
from chs.constants import (
    CHS_NAMESPACE,
    DELETE_WAIT_TIMEOUT_SEC,
    DYNAMIC_SINGLE_HOST_TOPOLOGY,
    NAMESPACE_DELETE_TIMEOUT_SEC,
    SUPPORTED_DYNAMIC_SLICE_TOPOLOGY,
)
from chs.schedulers.base import TpuSliceSummary
from chs.schedulers.kubernetes import (
    _APPLY_ATTEMPTS,
    KubernetesScheduler,
    SliceSpec,
    TpuSliceData,
    _gke_version,
    _partition_ids,
    _record_pool_node,
    _record_subslice_node,
    _slice_name,
    _with_retries,
)
from chs.schedulers.slurm import SlurmScheduler
from chs.schedulers import _get_kubectl_context, detect_scheduler
from google.auth.exceptions import GoogleAuthError
from chs.exceptions import ChsSchedulerException
from chs.utils import k8s_client
import click
from chs import exceptions as ex
from tests.k8s_fakes import (
    Call,
    install_fake_client,
)

_MANAGED_BY = "app.kubernetes.io/managed-by"
_INSTANCE = "app.kubernetes.io/instance"
_WORKER_LABELS = {_MANAGED_BY: "chs", "app.kubernetes.io/component": "worker"}

_LIST_CHS_SLICES = Call(
    "list_custom_objects",
    {
        "group": "accelerator.gke.io",
        "version": "v1beta1",
        "plural": "slices",
        "label_selector": f"{_MANAGED_BY}=chs",
    },
)

_LIST_RUN_A_SLICES = Call(
    "list_custom_objects",
    {
        "group": "accelerator.gke.io",
        "version": "v1beta1",
        "plural": "slices",
        "label_selector": f"{_MANAGED_BY}=chs,{_INSTANCE}=run-a",
    },
)

_LIST_CHS_COMPUTE_DOMAINS = Call(
    "list_custom_objects",
    {
        "group": "resource.nvidia.com",
        "version": "v1beta1",
        "plural": "computedomains",
        "namespace": CHS_NAMESPACE,
    },
)

_DELETE_CHS_NAMESPACE = Call(
    "delete_namespace",
    {
        "name": CHS_NAMESPACE,
        "wait_timeout_seconds": NAMESPACE_DELETE_TIMEOUT_SEC,
    },
)


def _chs_slice(pool_name: str, reason: str, run_id: str = "run-a") -> dict:
    """Returns a GKE Slice object created by CHS with the given Ready reason."""
    return {
        "metadata": {
            "name": f"chs-slice-{pool_name}",
            "labels": {_MANAGED_BY: "chs", _INSTANCE: run_id},
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
    """Returns a rendered job.yaml Job pinned to one node."""
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        **_job_object(name, [node]),
    }


def _accelerator_pod(
    name: str, node: str, labels: dict[str, str] | None = None
) -> dict[str, Any]:
    """Returns a running Pod on the node that uses GPUs."""
    return {
        "metadata": {"name": name, "labels": labels or {}},
        "spec": {
            "nodeName": node,
            "containers": [
                {
                    "name": "main",
                    "resources": {"limits": {"nvidia.com/gpu": "8"}},
                }
            ],
        },
        "status": {"phase": "Running"},
    }


def _finished_job(name: str, node: str, condition: str, reason: str) -> dict:
    """Returns a started Job with a True terminal condition."""
    job = _job_object(name, [node], suspended=False)
    job["status"] = {
        "conditions": [{"type": condition, "status": "True", "reason": reason}]
    }
    return job


def _waiting_pod(name: str, job_name: str, node: str) -> dict[str, Any]:
    """Returns a Job's pod pinned to the node, Unschedulable for 30 s."""
    created = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        seconds=30
    )
    pod = _pending_pod(
        name,
        job_name=job_name,
        creation_timestamp=created.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    pod["spec"] = _job_object(job_name, [node])["spec"]["template"]["spec"]
    return pod


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

    @mock.patch("subprocess.run")
    def test_get_nodes_under_test_accepts_require_whole_slices_attribute(
        self, mock_run, mock_logger
    ):
        """Slurm must accept require_whole_slices without error."""
        scheduler = SlurmScheduler(nodes="node1,node2")
        scheduler.require_whole_slices = True
        mock_run.return_value.stdout = "node1\nnode2"
        self.assertEqual(
            scheduler.get_nodes_under_test(),
            ["node1", "node2"],
        )

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
        # The nodes are resolved, filtered to idle ones and checked against
        # the jobs of other CHS runs.
        self.assertEqual(mock_run.call_count, 4)
        self.assertIn("scontrol", mock_run.call_args_list[0][0][0])
        self.assertIn("squeue", mock_run.call_args_list[-1][0][0])

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
        self.assertEqual(mock_run.call_count, 4)
        self.assertIn("sinfo", mock_run.call_args_list[0][0][0])
        self.assertIn("squeue", mock_run.call_args_list[-1][0][0])

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

    @mock.patch("chs.utils.gcp.get_default_credentials")
    def test_resolve_project_id_slurm_success(
        self, mock_auth_default, mock_logger
    ):
        scheduler = SlurmScheduler()
        mock_auth_default.return_value = (None, "my-project-id")
        project_id = scheduler._resolve_project_id("node1")
        self.assertEqual(project_id, "my-project-id")

    @mock.patch("chs.utils.gcp.get_default_credentials")
    def test_resolve_project_id_slurm_no_project(
        self, mock_auth_default, mock_logger
    ):
        scheduler = SlurmScheduler()
        mock_auth_default.return_value = (None, None)
        with self.assertRaisesRegex(
            ex.ChsSchedulerException, "Could not determine GCP project"
        ):
            scheduler._resolve_project_id("node1")

    @mock.patch("chs.utils.gcp.get_default_credentials")
    def test_resolve_project_id_slurm_error(
        self, mock_auth_default, mock_logger
    ):
        scheduler = SlurmScheduler()
        mock_auth_default.side_effect = GoogleAuthError("auth error")
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


def _tpu_node(
    name: str, pool: str, partition_ids: dict[str, str] | None = None
) -> dict[str, Any]:
    """Returns a TPU Node object of the pool in the partitions by shape."""
    labels = {
        "cloud.google.com/gke-nodepool": pool,
        "cloud.google.com/gke-tpu-accelerator": "tpu7x",
    }
    for topology, partition_id in (partition_ids or {}).items():
        labels[f"cloud.google.com/gke-tpu-partition-{topology}-id"] = (
            partition_id
        )
    return {
        "metadata": {"name": name, "labels": labels},
        "status": {"allocatable": {"google.com/tpu": "4"}},
    }


def _versioned_tpu_node(
    name: str, kubelet_version: str | None, provision_only: bool = True
) -> dict[str, Any]:
    """Returns a TPU Node object whose kubelet runs the given version."""
    node = _tpu_node(name, "dynpool" if provision_only else "staticpool")
    if provision_only:
        node["metadata"]["labels"][
            "cloud.google.com/gke-accelerator-topology-mode"
        ] = "PROVISION_ONLY"
    if kubelet_version is not None:
        node["status"]["nodeInfo"] = {"kubeletVersion": kubelet_version}
    return node


def _foreign_slice(
    name: str, partition_id: str, reason: str | None
) -> dict[str, Any]:
    """Returns a Slice not created by CHS, with the given Ready reason."""
    tpu_slice = {
        "metadata": {"name": name},
        "spec": {"partitionIds": [partition_id]},
    }
    if reason is not None:
        tpu_slice["status"] = {
            "conditions": [{"type": "Ready", "reason": reason}]
        }
    return tpu_slice


def _dynamic_tpu_node(
    name: str, single_host_state: str | None, pool_state: str = "UNHEALTHY"
) -> dict[str, Any]:
    """Returns a provision-only TPU Node with the given partition states."""
    node = _tpu_node(
        name,
        "dynpool",
        {
            DYNAMIC_SINGLE_HOST_TOPOLOGY: f"sub-{name}",
            SUPPORTED_DYNAMIC_SLICE_TOPOLOGY: "cube-1",
        },
    )
    labels = node["metadata"]["labels"]
    labels["cloud.google.com/gke-accelerator-topology-mode"] = "PROVISION_ONLY"
    labels[
        f"cloud.google.com/gke-tpu-partition-"
        f"{SUPPORTED_DYNAMIC_SLICE_TOPOLOGY}-state"
    ] = pool_state
    if single_host_state is not None:
        labels[
            f"cloud.google.com/gke-tpu-partition-"
            f"{DYNAMIC_SINGLE_HOST_TOPOLOGY}-state"
        ] = single_host_state
    return node


def _in_tpu_slice(
    node: dict[str, Any], topology: str = DYNAMIC_SINGLE_HOST_TOPOLOGY
) -> dict[str, Any]:
    """Returns the node with the topology label GKE stamps in a Slice."""
    node["metadata"]["labels"]["cloud.google.com/gke-tpu-topology"] = topology
    return node


def _labels(node: dict[str, Any]) -> dict[str, str]:
    """Returns the labels of the Node object."""
    return node["metadata"]["labels"]


def _single_host_slice(partition_id: str) -> SliceSpec:
    """Returns the SliceSpec of a single-host slice on the partition."""
    return SliceSpec(
        accelerator="tpu7x",
        topology=DYNAMIC_SINGLE_HOST_TOPOLOGY,
        partition_id=partition_id,
    )


def _dynamic_pool_data(
    subslice_partitions: dict[str, str | None], pool: str = "dynpool"
) -> TpuSliceData:
    """Returns the TpuSliceData of a dynamic pool of the given nodes.

    Nodes mapped to None have no HEALTHY single-host partition.
    """
    nodes = list(subslice_partitions)
    return TpuSliceData(
        pools={pool: nodes},
        node_to_pool={node: pool for node in nodes},
        pool_to_topology={pool: SUPPORTED_DYNAMIC_SLICE_TOPOLOGY},
        dynamic_slice_partitions={pool: f"{pool}-cube"},
        pool_to_accelerator={pool: "tpu7x"},
        all_tpu_nodes=set(nodes),
        provision_only_nodes=set(nodes),
        single_host_slices={
            node: _single_host_slice(partition_id)
            for node, partition_id in subslice_partitions.items()
            if partition_id
        },
    )


@mock.patch("chs.schedulers.kubernetes.logger")
class TestKubernetesScheduler(unittest.TestCase):
    """Tests for KubernetesScheduler."""

    def setUp(self):
        super().setUp()
        self.kube = install_fake_client(self)
        self.mock_adc = self.enterContext(
            mock.patch(
                "chs.utils.gcp.get_default_credentials",
                return_value=(None, None),
            )
        )
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
                "status": {
                    "allocatable": {"nvidia.com/gpu": "8"},
                    "conditions": [{"type": "Ready", "status": "True"}],
                },
            },
            {
                "metadata": {"name": "not-ready-node"},
                "status": {
                    "allocatable": {"nvidia.com/gpu": "8"},
                    "conditions": [{"type": "Ready", "status": "False"}],
                },
            },
            {
                "metadata": {"name": "unknown-node"},
                "status": {
                    "allocatable": {"nvidia.com/gpu": "8"},
                    "conditions": [{"type": "Ready", "status": "Unknown"}],
                },
            },
            {
                "metadata": {"name": "cordoned-node"},
                "spec": {"unschedulable": True},
                "status": {
                    "allocatable": {"nvidia.com/gpu": "8"},
                    "conditions": [{"type": "Ready", "status": "True"}],
                },
            },
            {
                "metadata": {"name": "empty-conditions-node"},
                "status": {
                    "allocatable": {"nvidia.com/gpu": "8"},
                    "conditions": [],
                },
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
                    },
                    "status": {"allocatable": {"nvidia.com/gpu": "8"}},
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
            "",
            "nodes",
            {
                "metadata": {"name": "node1"},
                "status": {"allocatable": {"nvidia.com/gpu": "8"}},
            },
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
                },
                "status": {"allocatable": {"nvidia.com/gpu": "8"}},
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
        """Test multi-host tests fall back to single-host on unready slice."""
        self.kube.add_object(
            "",
            "nodes",
            {
                "metadata": {
                    "name": "node1",
                    "labels": {
                        "cloud.google.com/gke-nodepool": "pool1",
                        "cloud.google.com/gke-tpu-accelerator": (
                            "tpu-v4-podslice"
                        ),
                    },
                },
                "status": {"allocatable": {"google.com/tpu": "4"}},
            },
        )
        self.kube.add_object(
            "",
            "nodes",
            {
                "metadata": {
                    "name": "node2",
                    "labels": {
                        "cloud.google.com/gke-nodepool": "pool1",
                        "cloud.google.com/gke-tpu-accelerator": (
                            "tpu-v4-podslice"
                        ),
                    },
                },
                "status": {"allocatable": {"google.com/tpu": "4"}},
            },
        )
        mock_tpu_slices.return_value = TpuSliceData(
            pools={"pool1": ["node1", "node2"]},
            node_to_pool={"node1": "pool1", "node2": "pool1"},
            pool_to_topology={"pool1": "2x2"},
        )
        scheduler = KubernetesScheduler(nodepools="pool1")
        scheduler.require_whole_slices = True

        valid_nodes = scheduler.get_nodes_under_test()

        self.assertEqual(valid_nodes, [])
        self.assertEqual(scheduler.partially_busy_slices, {"pool1": ["node2"]})
        mock_logger.warning.assert_called_once()
        self.assertIn(
            "Cannot run multi-host test(s)",
            mock_logger.warning.call_args[0][0],
        )

        single_host_scheduler = KubernetesScheduler(nodepools="pool1")
        valid_single_host = single_host_scheduler.get_nodes_under_test()
        self.assertEqual(valid_single_host, ["node1"])

        node_scheduler = KubernetesScheduler(nodes="node1")
        self.assertEqual(node_scheduler.get_nodes_under_test(), ["node1"])

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_not_ready_nodes",
        return_value={"node2"},
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_busy_nodes",
        return_value=set(),
    )
    @mock.patch("chs.schedulers.kubernetes.KubernetesScheduler._get_tpu_slices")
    def test_get_nodes_under_test_dynamic_tpu_slice(
        self,
        mock_tpu_slices,
        mock_busy,
        mock_not_ready,
        mock_logger,
    ):
        """Test dynamic TPU nodes need a whole pool or their own sub-slice."""
        self.kube.add_object(
            "",
            "nodes",
            {
                "metadata": {
                    "name": "node1",
                    "labels": {
                        "cloud.google.com/gke-nodepool": "dynpool",
                        "cloud.google.com/gke-tpu-accelerator": "tpu7x",
                    },
                },
                "status": {"allocatable": {"google.com/tpu": "4"}},
            },
        )
        self.kube.add_object(
            "",
            "nodes",
            {
                "metadata": {
                    "name": "node2",
                    "labels": {
                        "cloud.google.com/gke-nodepool": "dynpool",
                        "cloud.google.com/gke-tpu-accelerator": "tpu7x",
                    },
                },
                "status": {"allocatable": {"google.com/tpu": "4"}},
            },
        )
        mock_tpu_slices.return_value = TpuSliceData(
            pools={"dynpool": ["node1", "node2"]},
            node_to_pool={"node1": "dynpool", "node2": "dynpool"},
            pool_to_topology={"dynpool": "4x4x4"},
            dynamic_slice_partitions={"dynpool": "dyn-part-1"},
            pool_to_accelerator={"dynpool": "tpu7x"},
        )
        scheduler = KubernetesScheduler(nodepools="dynpool")
        scheduler.require_whole_slices = False

        valid_nodes = scheduler.get_nodes_under_test()
        self.assertEqual(valid_nodes, [])
        self.assertEqual(scheduler.partially_busy_slices, {})
        mock_logger.warning.assert_any_call(
            "Cannot run tests on dynamic TPU pool '%s': dynamic slicing "
            "requires the entire pool to be available to provision the "
            "Slice CR, but the following node(s) are busy or unavailable: %s",
            "dynpool",
            "node2",
        )

        mock_tpu_slices.return_value = TpuSliceData(
            pools={"staticpool": ["node2"]},
            node_to_pool={"node2": "staticpool"},
            pool_to_topology={"staticpool": "2x2"},
            pool_to_accelerator={"staticpool": "tpu7x"},
            all_tpu_nodes={"node1", "node2"},
            provision_only_nodes={"node1"},
        )
        mixed_scheduler = KubernetesScheduler(nodes="node1,node2")
        mock_not_ready.return_value = set()
        self.assertEqual(mixed_scheduler.get_nodes_under_test(), ["node2"])
        self.assertEqual(mixed_scheduler._subslice_nodes, set())
        mock_logger.warning.assert_any_call(
            "Excluded node(s) %s from testing because their %s TPU "
            "partition is not HEALTHY or not labelled on the node.",
            "node1",
            DYNAMIC_SINGLE_HOST_TOPOLOGY,
        )

        mock_tpu_slices.return_value = TpuSliceData(
            all_tpu_nodes={"node1"},
            provision_only_nodes={"node1"},
            single_host_slices={"node1": _single_host_slice("sub-part-1")},
        )
        only_dynamic_scheduler = KubernetesScheduler(nodes="node1")
        self.assertEqual(
            only_dynamic_scheduler.get_nodes_under_test(), ["node1"]
        )
        self.assertEqual(only_dynamic_scheduler._subslice_nodes, {"node1"})
        self.assertEqual(only_dynamic_scheduler.partially_busy_slices, {})
        for call_args in mock_logger.info.call_args_list:
            self.assertNotIn("existing workloads", call_args[0][0])

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_not_ready_nodes",
        return_value={"d3"},
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_busy_nodes",
        return_value=set(),
    )
    @mock.patch("chs.schedulers.kubernetes.KubernetesScheduler._get_tpu_slices")
    def test_get_nodes_under_test_targets_each_dynamic_node_as_a_subslice(
        self, mock_tpu_slices, mock_busy, mock_not_ready, mock_logger
    ):
        for name in ("d1", "d2", "d3"):
            self.kube.add_object(
                "",
                "nodes",
                _tpu_node(
                    name,
                    "dynpool",
                    {DYNAMIC_SINGLE_HOST_TOPOLOGY: f"sub-{name}"},
                ),
            )
        mock_tpu_slices.return_value = _dynamic_pool_data(
            {"d1": "sub-d1", "d2": "sub-d2", "d3": "sub-d3"}
        )
        scheduler = KubernetesScheduler(nodes="d1,d2,d3")

        self.assertEqual(scheduler.get_nodes_under_test(), ["d1", "d2"])
        self.assertEqual(scheduler._subslice_nodes, {"d1", "d2"})
        self.assertEqual(scheduler.partially_busy_slices, {})
        mock_logger.warning.assert_not_called()

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_not_ready_nodes",
        return_value=set(),
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_busy_nodes",
        return_value=set(),
    )
    @mock.patch("chs.schedulers.kubernetes.KubernetesScheduler._get_tpu_slices")
    def test_get_nodes_under_test_excludes_dynamic_nodes_unfit_for_a_subslice(
        self, mock_tpu_slices, mock_busy, mock_not_ready, mock_logger
    ):
        for node in (
            _tpu_node("bare", "dynpool"),
            _in_tpu_slice(
                _tpu_node(
                    "taken-sub", "dynpool", {DYNAMIC_SINGLE_HOST_TOPOLOGY: "s1"}
                )
            ),
            _in_tpu_slice(
                _tpu_node(
                    "taken-cube",
                    "dynpool",
                    {
                        DYNAMIC_SINGLE_HOST_TOPOLOGY: "s2",
                        SUPPORTED_DYNAMIC_SLICE_TOPOLOGY: "cube-1",
                    },
                ),
                SUPPORTED_DYNAMIC_SLICE_TOPOLOGY,
            ),
            _tpu_node("free", "dynpool", {DYNAMIC_SINGLE_HOST_TOPOLOGY: "s3"}),
        ):
            self.kube.add_object("", "nodes", node)
        mock_tpu_slices.return_value = _dynamic_pool_data(
            {"bare": None, "taken-sub": "s1", "taken-cube": "s2", "free": "s3"}
        )
        scheduler = KubernetesScheduler(nodes="bare,taken-sub,taken-cube,free")

        self.assertEqual(scheduler.get_nodes_under_test(), ["free"])
        self.assertEqual(scheduler._subslice_nodes, {"free"})
        self.assertEqual(
            mock_logger.warning.call_args_list,
            [
                mock.call(
                    "Excluded node(s) %s from testing because their %s TPU "
                    "partition is not HEALTHY or not labelled on the node.",
                    "bare",
                    DYNAMIC_SINGLE_HOST_TOPOLOGY,
                ),
                mock.call(
                    "Excluded node(s) %s from testing because they are "
                    "already part of a TPU slice.",
                    "taken-sub, taken-cube",
                ),
            ],
        )

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_not_ready_nodes",
        return_value=set(),
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_busy_nodes",
        return_value=set(),
    )
    @mock.patch("chs.schedulers.kubernetes.KubernetesScheduler._get_tpu_slices")
    def test_get_nodes_under_test_excludes_nodes_in_a_slice_by_label_only(
        self, mock_tpu_slices, mock_busy, mock_not_ready, mock_logger
    ):
        nodes = ("d1", "d2", "d3", "d4")
        for name in nodes:
            node = _tpu_node(
                name, "dynpool", {DYNAMIC_SINGLE_HOST_TOPOLOGY: f"sub-{name}"}
            )
            if name != "d4":
                _in_tpu_slice(node)
            self.kube.add_object("", "nodes", node)
        mock_tpu_slices.return_value = _dynamic_pool_data(
            {name: f"sub-{name}" for name in nodes}
        )
        scheduler = KubernetesScheduler(nodes="d1,d2,d3,d4")

        self.assertEqual(scheduler.get_nodes_under_test(), ["d4"])
        self.assertEqual(scheduler._subslice_nodes, {"d4"})
        mock_logger.warning.assert_called_once_with(
            "Excluded node(s) %s from testing because they are already "
            "part of a TPU slice.",
            "d1, d2, d3",
        )
        self.assertEqual(
            self.kube.calls_to("list_custom_objects"), [_LIST_CHS_SLICES]
        )

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_not_ready_nodes",
        return_value=set(),
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_busy_nodes",
        return_value=set(),
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._nodes_unfit_for_subslice"
    )
    def test_get_nodes_under_test_skips_subslice_checks_without_dynamic_nodes(
        self, mock_unfit, mock_busy, mock_not_ready, mock_logger
    ):
        for name in ("s1", "s2"):
            self.kube.add_object(
                "", "nodes", _in_tpu_slice(_tpu_node(name, "staticpool"), "2x2")
            )
        scheduler = KubernetesScheduler(nodes="s1,s2")

        self.assertCountEqual(scheduler.get_nodes_under_test(), ["s1", "s2"])
        self.assertEqual(scheduler._subslice_nodes, set())
        mock_unfit.assert_not_called()

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_not_ready_nodes",
        return_value=set(),
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_busy_nodes",
        return_value=set(),
    )
    def test_get_nodes_under_test_targets_healthy_single_host_partitions(
        self, mock_busy, mock_not_ready, mock_logger
    ):
        for node in (
            _dynamic_tpu_node("healthy", "HEALTHY"),
            _dynamic_tpu_node("unhealthy", "UNHEALTHY"),
            _dynamic_tpu_node("unlabeled", None),
        ):
            self.kube.add_object("", "nodes", node)
        scheduler = KubernetesScheduler(nodes="healthy,unhealthy,unlabeled")

        self.assertEqual(scheduler.get_nodes_under_test(), ["healthy"])
        self.assertEqual(scheduler._subslice_nodes, {"healthy"})
        mock_logger.warning.assert_called_once_with(
            "Excluded node(s) %s from testing because their %s TPU "
            "partition is not HEALTHY or not labelled on the node.",
            "unhealthy, unlabeled",
            DYNAMIC_SINGLE_HOST_TOPOLOGY,
        )

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_not_ready_nodes",
        return_value=set(),
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_busy_nodes",
        return_value=set(),
    )
    @mock.patch("chs.schedulers.kubernetes.KubernetesScheduler._get_tpu_slices")
    def test_get_nodes_under_test_nodepools_targets_the_whole_dynamic_pool(
        self, mock_tpu_slices, mock_busy, mock_not_ready, mock_logger
    ):
        for name in ("d1", "d2"):
            self.kube.add_object(
                "",
                "nodes",
                _tpu_node(
                    name,
                    "dynpool",
                    {DYNAMIC_SINGLE_HOST_TOPOLOGY: f"sub-{name}"},
                ),
            )
        mock_tpu_slices.return_value = _dynamic_pool_data(
            {"d1": "sub-d1", "d2": "sub-d2"}
        )
        scheduler = KubernetesScheduler(nodepools="dynpool")

        self.assertCountEqual(scheduler.get_nodes_under_test(), ["d1", "d2"])
        self.assertEqual(scheduler._subslice_nodes, set())

    def test_slices_to_provision_keys_subslice_nodes_by_node_name(
        self, mock_logger
    ):
        for name, pool in (
            ("d1", "dynpool"),
            ("d2", "dynpool"),
            ("s1", "staticpool"),
            ("s2", "staticpool"),
        ):
            self.kube.add_object("", "nodes", _tpu_node(name, pool))
        scheduler = KubernetesScheduler()
        scheduler._subslice_nodes = {"d1", "d2"}

        self.assertEqual(
            scheduler._slices_to_provision(["d1", "s1", "d2", "s2"]),
            {"d1": ["d1"], "d2": ["d2"], "staticpool": ["s1", "s2"]},
        )
        self.assertEqual(
            scheduler._group_nodes_by_slice(["d1", "s1", "d2", "s2"]),
            {"dynpool": ["d1", "d2"], "staticpool": ["s1", "s2"]},
        )

    def test_tpu_slice_data_purge_pool_keeps_single_host_slices(
        self, mock_logger
    ):
        single_host_slices = {
            "d1": _single_host_slice("s1"),
            "e1": _single_host_slice("s3"),
        }
        data = TpuSliceData(
            pools={"dynpool": ["d1", "d2"], "otherpool": ["e1"]},
            node_to_pool={"d1": "dynpool", "d2": "dynpool", "e1": "otherpool"},
            pool_to_topology={"dynpool": "4x4x4", "otherpool": "4x4x4"},
            dynamic_slice_partitions={"dynpool": "c1", "otherpool": "c2"},
            pool_to_accelerator={"dynpool": "tpu7x", "otherpool": "tpu7x"},
            provision_only_nodes={"d1", "d2", "e1"},
            single_host_slices=dict(single_host_slices),
        )

        data.purge_pool("dynpool")

        self.assertEqual(
            data,
            TpuSliceData(
                pools={"otherpool": ["e1"]},
                node_to_pool={"e1": "otherpool"},
                pool_to_topology={"otherpool": "4x4x4"},
                dynamic_slice_partitions={"otherpool": "c2"},
                pool_to_accelerator={"otherpool": "tpu7x"},
                provision_only_nodes={"d1", "d2", "e1"},
                single_host_slices=single_host_slices,
            ),
        )

    def test_slice_name_keeps_keys_that_fit(self, mock_logger):
        longest_key = "n" * 39

        self.assertEqual(_slice_name("pool1"), "chs-slice-pool1")
        self.assertEqual(_slice_name(longest_key), f"chs-slice-{longest_key}")

    def test_slice_name_shortens_long_keys_uniquely(self, mock_logger):
        prefix = "gke-my-cluster-tpu7x-dynamic-pool-" + "a" * 20 + "-3f2a9c1b"
        keys = [f"{prefix}-x7k2", f"{prefix}-q9m4"]
        names = [_slice_name(key) for key in keys]

        for key, name in zip(keys, names):
            self.assertLessEqual(len(name), 49)
            self.assertRegex(name, r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
            match = re.fullmatch(r"chs-slice-(.+)-([0-9a-f]{8})", name)
            self.assertIsNotNone(match)
            self.assertTrue(key.endswith(match.group(1)))
            self.assertTrue(match.group(1).endswith(key[-4:]))
            self.assertEqual(_slice_name(key), name)
        self.assertNotEqual(names[0], names[1])

    def _assert_preflight_rejects(
        self, scheduler: KubernetesScheduler, node: str, version: str
    ) -> None:
        with self.assertRaises(ex.ChsValidationError) as cm:
            scheduler.run_preflight_checks([{"name": "test1"}])
        self.assertEqual(
            str(cm.exception),
            f"Node '{node}' of a provision-only TPU node pool runs GKE "
            f"'{version}', but --nodes needs GKE 1.36.0-gke.3712000 or later "
            "to create single-host slices. Upgrade its node pool, or test the "
            "whole pool with --nodepools.",
        )

    def test_run_preflight_checks_rejects_dynamic_nodes_too_old_to_subslice(
        self, mock_logger
    ):
        self.kube.add_object(
            "", "nodes", _versioned_tpu_node("new", "v1.36.0-gke.3712000")
        )
        for name, version in (
            ("old1", "v1.35.2-gke.100"),
            ("old2", "v1.36.0-gke.3711999"),
        ):
            self.kube.add_object(
                "", "nodes", _versioned_tpu_node(name, version)
            )
            with self.subTest(version=version):
                self._assert_preflight_rejects(
                    KubernetesScheduler(nodes=f"new,{name}"), name, version
                )

    def test_run_preflight_checks_rejects_dynamic_nodes_of_unknown_versions(
        self, mock_logger
    ):
        for name, version in (
            ("no-gke", "v1.35.0"),
            ("garbage", "unknown"),
            ("missing", None),
        ):
            self.kube.add_object(
                "", "nodes", _versioned_tpu_node(name, version)
            )
            with self.subTest(version=version):
                self._assert_preflight_rejects(
                    KubernetesScheduler(nodes=name), name, version or ""
                )

    def test_run_preflight_checks_passes_dynamic_nodes_that_can_subslice(
        self, mock_logger
    ):
        for name, version in (
            ("min", "v1.36.0-gke.3712000"),
            ("newer", "v1.36.4-gke.2046003"),
            ("rapid", "1.37.1-gke.2097000"),
        ):
            self.kube.add_object(
                "", "nodes", _versioned_tpu_node(name, version)
            )
        tests = [{"name": "test1"}]

        self.assertIs(
            KubernetesScheduler(nodes="min,newer,rapid").run_preflight_checks(
                tests
            ),
            tests,
        )

    def test_run_preflight_checks_ignores_old_static_nodes(self, mock_logger):
        for name, version in (("s1", "v1.30.0-gke.1"), ("s2", None)):
            self.kube.add_object(
                "",
                "nodes",
                _versioned_tpu_node(name, version, provision_only=False),
            )
        tests = [{"name": "test1"}]

        self.assertIs(
            KubernetesScheduler(nodes="s1,s2").run_preflight_checks(tests),
            tests,
        )

    def test_run_preflight_checks_skips_nodepools_on_old_dynamic_pools(
        self, mock_logger
    ):
        self.kube.add_object(
            "", "nodes", _versioned_tpu_node("d1", "v1.35.2-gke.100")
        )
        tests = [{"name": "test1"}]

        scheduler = KubernetesScheduler(nodepools="dynpool")

        self.assertIs(scheduler.run_preflight_checks(tests), tests)
        self.assertEqual(self.kube.calls, [])

    def test_gke_version(self, mock_logger):
        for version, expected in (
            ("v1.36.0-gke.3712000", (1, 36, 0, 3712000)),
            ("1.36.4-gke.2046003", (1, 36, 4, 2046003)),
            ("1.37.1-gke.2097000", (1, 37, 1, 2097000)),
            ("1.11.10-gke.5", (1, 11, 10, 5)),
            ("v1.36.10-gke.1200-extra", (1, 36, 10, 1200)),
            ("v1.36.0", ()),
            ("gke-1.36.0", ()),
            ("unknown", ()),
            ("", ()),
        ):
            with self.subTest(version=version):
                self.assertEqual(_gke_version(version), expected)
        self.assertGreater(
            _gke_version("v1.36.10-gke.1"),
            _gke_version("v1.36.9-gke.9999999"),
        )
        self.assertLess(
            _gke_version("1.11.10-gke.5"), _gke_version("1.36.0-gke.1")
        )
        self.assertLess(_gke_version("unknown"), _gke_version("0.0.0-gke.0"))

    def test_partition_ids_reads_only_tpu7x_partition_shapes(self, mock_logger):
        prefix = "cloud.google.com/gke-tpu-partition-"
        node = {
            "metadata": {
                "labels": {
                    f"{prefix}2x2x1-id": "p-2x2x1",
                    f"{prefix}2x2x2-id": "p-2x2x2",
                    f"{prefix}2x2x4-id": "p-2x2x4",
                    f"{prefix}2x4x4-id": "p-2x4x4",
                    f"{prefix}4x4x4-id": "p-4x4x4",
                    f"{prefix}2x2-id": "p-2d",
                    f"{prefix}8x8x8-id": "p-8x8x8",
                    f"{prefix}4x4x4-state": "HEALTHY",
                    "cloud.google.com/gke-nodepool": "dynpool",
                }
            }
        }

        self.assertEqual(
            _partition_ids(node),
            {"p-2x2x1", "p-2x2x2", "p-2x2x4", "p-2x4x4", "p-4x4x4"},
        )
        self.assertEqual(_partition_ids({"metadata": {}}), set())

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler.get_topology",
        return_value={},
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler.get_nodes_under_test",
        return_value=["d1", "s1"],
    )
    def test_write_ramble_config_emits_subslice_and_pool_topologies(
        self, mock_nodes, mock_topology, mock_logger
    ):
        self.kube.add_object("", "nodes", _tpu_node("d1", "dynpool"))
        self.kube.add_object("", "nodes", _tpu_node("s1", "staticpool"))
        data = TpuSliceData(
            pools={"staticpool": ["s1"]},
            node_to_pool={"s1": "staticpool"},
            pool_to_topology={"staticpool": "2x2"},
            provision_only_nodes={"d1"},
            single_host_slices={"d1": _single_host_slice("sub-d1")},
        )
        scheduler = KubernetesScheduler(nodes="d1,s1")
        scheduler._tpu_slice_data = data
        scheduler._subslice_nodes = {"d1"}
        workspace_config_dir = os.path.join(self.tmp_dir, "configs")
        os.makedirs(workspace_config_dir)

        scheduler.write_ramble_config(
            workspace_config_dir,
            [{"name": "test1"}],
            "tpu7x-standard-4t",
            run_id="run-a",
        )

        with open(os.path.join(workspace_config_dir, "ramble.yaml")) as f:
            variables = yaml.safe_load(f)["ramble"]["variables"]
        self.assertEqual(
            variables["tpu_slices"], {"dynpool": ["d1"], "staticpool": ["s1"]}
        )
        self.assertEqual(
            variables["tpu_slice_topologies"],
            {"dynpool": DYNAMIC_SINGLE_HOST_TOPOLOGY, "staticpool": "2x2"},
        )
        self.assertEqual(variables["tpu_dynamic_pools"], [])
        self.assertEqual(variables["tpu_single_host_pools"], ["dynpool"])

    def test_is_own_slice_finds_only_slices_of_this_run(self, mock_logger):
        for pool, run_id in (("mine", "run-a"), ("theirs", "run-b")):
            self.kube.add_object(
                "accelerator.gke.io",
                "slices",
                _chs_slice(pool, "ACTIVE", run_id=run_id),
            )
        scheduler = KubernetesScheduler()
        scheduler._run_id = "run-a"

        self.assertTrue(scheduler._is_own_slice("chs-slice-mine"))
        self.assertFalse(scheduler._is_own_slice("chs-slice-theirs"))
        self.assertEqual(
            self.kube.calls_to("list_custom_objects"),
            [_LIST_RUN_A_SLICES, _LIST_RUN_A_SLICES],
        )

    def test_is_own_slice_is_false_if_slices_cannot_be_listed(
        self, mock_logger
    ):
        self.kube.add_object(
            "accelerator.gke.io", "slices", _chs_slice("mine", "ACTIVE")
        )
        self.kube.fail_next("list_custom_objects", k8s_client.KubeError("boom"))
        scheduler = KubernetesScheduler()
        scheduler._run_id = "run-a"

        self.assertFalse(scheduler._is_own_slice("chs-slice-mine"))
        mock_logger.warning.assert_called_once()

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_not_ready_nodes",
        return_value=set(),
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_busy_nodes",
        return_value=set(),
    )
    @mock.patch("chs.schedulers.kubernetes.KubernetesScheduler._get_tpu_slices")
    def test_get_nodes_under_test_excludes_unhealthy_slice_nodes(
        self,
        mock_tpu_slices,
        mock_busy,
        mock_not_ready,
        mock_logger,
    ):
        """Test unhealthy-slice nodes are excluded when no pool is healthy."""
        for name in ("cube-a-0", "cube-a-1", "cube-a-2"):
            self.kube.add_object(
                "",
                "nodes",
                {
                    "metadata": {
                        "name": name,
                        "labels": {
                            "cloud.google.com/gke-nodepool": "cube-a",
                            "cloud.google.com/gke-tpu-accelerator": "tpu7x",
                        },
                    },
                    "status": {"allocatable": {"google.com/tpu": "4"}},
                },
            )
        # An unhealthy pool is purged from pools/node_to_pool but its nodes
        # remain in all_tpu_nodes.
        unhealthy_only = TpuSliceData(
            all_tpu_nodes={"cube-a-0", "cube-a-1", "cube-a-2"},
        )
        expected_warning = (
            "Excluded TPU node '%s' from testing because "
            "its parent slice is unhealthy."
        )

        for require_whole_slices in (False, True):
            for scheduler in (
                KubernetesScheduler(nodepools="cube-a"),
                KubernetesScheduler(nodes="cube-a-0,cube-a-1"),
                KubernetesScheduler(),
            ):
                mock_tpu_slices.return_value = unhealthy_only
                mock_logger.reset_mock()
                scheduler.require_whole_slices = require_whole_slices
                self.assertEqual(scheduler.get_nodes_under_test(), [])
                self.assertEqual(scheduler.partially_busy_slices, {})
                mock_logger.warning.assert_any_call(
                    expected_warning, "cube-a-0"
                )

        # With a healthy pool also in scope, only the healthy pool survives.
        mock_tpu_slices.return_value = TpuSliceData(
            pools={"cube-b": ["cube-b-0"]},
            node_to_pool={"cube-b-0": "cube-b"},
            pool_to_topology={"cube-b": "4x4x4"},
            all_tpu_nodes={"cube-a-0", "cube-a-1", "cube-a-2", "cube-b-0"},
        )
        self.kube.add_object(
            "",
            "nodes",
            {
                "metadata": {
                    "name": "cube-b-0",
                    "labels": {
                        "cloud.google.com/gke-nodepool": "cube-b",
                        "cloud.google.com/gke-tpu-accelerator": "tpu7x",
                    },
                },
                "status": {"allocatable": {"google.com/tpu": "4"}},
            },
        )
        scheduler = KubernetesScheduler(nodepools="cube-a,cube-b")
        scheduler.require_whole_slices = True
        self.assertEqual(scheduler.get_nodes_under_test(), ["cube-b-0"])

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_not_ready_nodes",
        return_value=set(),
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_busy_nodes",
        return_value=set(),
    )
    @mock.patch("chs.schedulers.kubernetes.KubernetesScheduler._get_tpu_slices")
    def test_get_nodes_under_test_skips_gpu_nodes_targeted_by_other_runs(
        self,
        mock_tpu_slices,
        mock_busy,
        mock_not_ready,
        mock_logger,
    ):
        targeted = [f"node{i}" for i in range(2, 9)]
        for name in ["node1", *targeted]:
            self.kube.add_object("", "nodes", {"metadata": {"name": name}})
        self.kube.add_object(
            "batch",
            "jobs",
            _run_job("chs-run-b-x-a1b2c3", "run-b", targeted),
            namespace=CHS_NAMESPACE,
        )
        mock_tpu_slices.return_value = TpuSliceData(
            pools={}, node_to_pool={}, pool_to_topology={}
        )
        scheduler = KubernetesScheduler(nodes=",".join(["node1", *targeted]))

        self.assertEqual(scheduler.get_nodes_under_test(), ["node1"])
        mock_logger.warning.assert_called_once_with(
            "Skipping %d node(s) targeted by CHS run '%s'.", 7, "run-b"
        )

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_not_ready_nodes",
        return_value=set(),
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_busy_nodes",
        return_value=set(),
    )
    @mock.patch("chs.schedulers.kubernetes.KubernetesScheduler._get_tpu_slices")
    def test_get_nodes_under_test_skips_tpu_slice_targeted_by_other_runs(
        self,
        mock_tpu_slices,
        mock_busy,
        mock_not_ready,
        mock_logger,
    ):
        for name in ("t1", "t2"):
            self.kube.add_object(
                "",
                "nodes",
                {
                    "metadata": {
                        "name": name,
                        "labels": {
                            "cloud.google.com/gke-nodepool": "pool1",
                            "cloud.google.com/gke-tpu-accelerator": (
                                "tpu-v4-podslice"
                            ),
                        },
                    },
                    "status": {"allocatable": {"google.com/tpu": "4"}},
                },
            )
        for name in ("t3", "t4"):
            self.kube.add_object(
                "",
                "nodes",
                {
                    "metadata": {
                        "name": name,
                        "labels": {
                            "cloud.google.com/gke-nodepool": "pool2",
                            "cloud.google.com/gke-tpu-accelerator": (
                                "tpu-v4-podslice"
                            ),
                        },
                    },
                    "status": {"allocatable": {"google.com/tpu": "4"}},
                },
            )
        self.kube.add_object(
            "batch",
            "jobs",
            _run_job("chs-run-b-x-a1b2c3", "run-b", ["t2"]),
            namespace=CHS_NAMESPACE,
        )
        mock_tpu_slices.return_value = TpuSliceData(
            pools={"pool1": ["t1", "t2"], "pool2": ["t3", "t4"]},
            node_to_pool={
                "t1": "pool1",
                "t2": "pool1",
                "t3": "pool2",
                "t4": "pool2",
            },
            pool_to_topology={"pool1": "2x2", "pool2": "2x2"},
        )
        scheduler = KubernetesScheduler(nodepools="pool1,pool2")
        scheduler.require_whole_slices = True
        self.assertCountEqual(
            scheduler.get_nodes_under_test(),
            ["t3", "t4"],
        )
        self.assertEqual(
            mock_logger.warning.call_args_list[0],
            mock.call(
                "Skipping %d node(s) targeted by CHS run '%s'.", 1, "run-b"
            ),
        )
        self.assertIn("pool1", mock_logger.warning.call_args_list[1][0])

        node_scheduler = KubernetesScheduler(nodes="t1,t3")
        self.assertCountEqual(
            node_scheduler.get_nodes_under_test(), ["t1", "t3"]
        )

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_not_ready_nodes",
        return_value=set(),
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._get_busy_nodes",
        return_value=set(),
    )
    @mock.patch("chs.schedulers.kubernetes.KubernetesScheduler._get_tpu_slices")
    def test_get_nodes_under_test_raises_if_runs_cannot_be_listed(
        self,
        mock_tpu_slices,
        mock_busy,
        mock_not_ready,
        mock_logger,
    ):
        self.kube.add_object("", "nodes", {"metadata": {"name": "node1"}})
        mock_tpu_slices.return_value = TpuSliceData(
            pools={}, node_to_pool={}, pool_to_topology={}
        )
        self.kube.fail_next(
            "list_jobs", k8s_client.KubeError("forbidden", status=403)
        )
        scheduler = KubernetesScheduler(nodes="node1")

        with self.assertRaisesRegex(ChsSchedulerException, "cannot verify"):
            scheduler.get_nodes_under_test()

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
                    },
                    "status": {"allocatable": {"nvidia.com/gpu": "8"}},
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
    workflow_manager: kubernetes-job
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
        """Test write_ramble_config writes defaults with suspend_job."""
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
                run_id="run-a",
            )

        output_path = os.path.join(workspace_config_dir, "ramble.yaml")

        with open(output_path, "r") as f:
            data = yaml.safe_load(f)

        variables = data["ramble"]["variables"]
        self.assertEqual(variables["suspend_job"], "true")

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler.get_nodes_under_test",
        return_value=["node1"],
    )
    def test_write_ramble_config_sets_run_variables(
        self, mock_filter, mock_logger
    ):
        workspace_config_dir = os.path.join(self.tmp_dir, "configs")
        os.makedirs(workspace_config_dir, exist_ok=True)

        with mock.patch.object(
            KubernetesScheduler,
            "_discover_networks",
            return_value=(["default"], ["rdma-0"]),
        ):
            KubernetesScheduler(nodes="node1").write_ramble_config(
                workspace_config_dir,
                [{"name": "test1"}],
                "a3-megagpu-8g",
                run_id="run-a",
            )

        with open(os.path.join(workspace_config_dir, "ramble.yaml")) as f:
            variables = yaml.safe_load(f)["ramble"]["variables"]
        self.assertEqual(variables["chs_run_id"], "run-a")
        self.assertEqual(variables["worker_service_name"], "chs-run-a")

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
                        "allocatable": {"nvidia.com/gpu": "8"},
                        "conditions": [{"type": "Ready", "status": "True"}],
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
                "status": {
                    "allocatable": {"nvidia.com/gpu": "8"},
                    "conditions": [{"type": "Ready", "status": "True"}],
                },
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
                "status": {
                    "allocatable": {"nvidia.com/gpu": "8"},
                    "conditions": [{"type": "Ready", "status": "True"}],
                },
            },
        )

        topology = scheduler.get_topology()

        self.assertEqual(len(topology), 2)
        self.assertEqual(
            topology["default-block"]["default-subblock"], ["node1"]
        )
        self.assertEqual(topology["block2"]["default-subblock"], ["node2"])

    @mock.patch("chs.utils.gcp.query_gce_instances_info")
    def test_get_topology_falls_back_to_gce_physical_host_topology(
        self, mock_query_info, mock_logger
    ):
        """Test get_topology queries GCE topology when labels are missing."""
        scheduler = KubernetesScheduler(nodes="node1,node2")
        self.mock_adc.return_value = (mock.Mock(), "adc-project")
        self.kube.add_object(
            "",
            "nodes",
            {
                "metadata": {"name": "node1", "labels": {}},
                "status": {
                    "allocatable": {"nvidia.com/gpu": "8"},
                    "conditions": [{"type": "Ready", "status": "True"}],
                },
            },
        )
        self.kube.add_object(
            "",
            "nodes",
            {
                "metadata": {"name": "node2", "labels": {}},
                "status": {
                    "allocatable": {"nvidia.com/gpu": "8"},
                    "conditions": [{"type": "Ready", "status": "True"}],
                },
            },
        )
        mock_query_info.return_value = {
            "node1": {
                "machine_type": "a3-megagpu-8g",
                "instance_id": "111",
                "block": "gce-block-1",
                "subblock": "gce-subblock-1",
            },
            "node2": {
                "machine_type": "a3-megagpu-8g",
                "instance_id": "222",
            },
        }

        machine_types = scheduler.detect_nodes_hardware(["node1", "node2"])
        topology = scheduler.get_topology()

        self.assertEqual(machine_types, ["a3-megagpu-8g"])
        mock_query_info.assert_called_once_with(
            "adc-project", ["node1", "node2"]
        )
        self.assertEqual(
            topology,
            {
                "gce-block-1": {"gce-subblock-1": ["node1"]},
                "default-block": {"default-subblock": ["node2"]},
            },
        )

    def test_resolve_raw_nodes_skips_non_accelerator_nodes(self, mock_logger):
        """Test resolve_raw_nodes skips 0-accelerator nodes."""
        self.kube.add_object(
            "",
            "nodes",
            {
                "metadata": {
                    "name": "cp-cpu-node",
                    "labels": {"node-role.kubernetes.io/control-plane": ""},
                },
                "status": {"allocatable": {"cpu": "16"}},
            },
        )
        self.kube.add_object(
            "",
            "nodes",
            {
                "metadata": {
                    "name": "cp-tpu-node",
                    "labels": {"node-role.kubernetes.io/control-plane": "true"},
                },
                "status": {"allocatable": {"google.com/tpu": "4"}},
            },
        )
        self.kube.add_object(
            "",
            "nodes",
            {
                "metadata": {"name": "no-status-node", "labels": {}},
            },
        )
        self.kube.add_object(
            "",
            "nodes",
            {
                "metadata": {"name": "cpu-node", "labels": {}},
                "status": {
                    "allocatable": {
                        "cpu": "32",
                        "memory": "128Gi",
                        "nvidia.com/gpu": "0",
                    }
                },
            },
        )
        self.kube.add_object(
            "",
            "nodes",
            {
                "metadata": {"name": "gpu-node", "labels": {}},
                "status": {"allocatable": {"nvidia.com/gpu": "8"}},
            },
        )
        self.kube.add_object(
            "",
            "nodes",
            {
                "metadata": {"name": "tpu-node", "labels": {}},
                "status": {"allocatable": {"google.com/tpu": "4"}},
            },
        )

        scheduler = KubernetesScheduler()
        nodes = scheduler.resolve_raw_nodes()
        self.assertEqual(nodes, ["cp-tpu-node", "gpu-node", "tpu-node"])

    def test_cancel_all_runs_without_jobs(self, mock_logger):
        """Test cancel_all_runs deletes the slices and the namespace."""
        scheduler = KubernetesScheduler()

        scheduler.cancel_all_runs()

        self.assertEqual(
            self.kube.calls,
            [
                Call(
                    "list_jobs",
                    {
                        "namespace": CHS_NAMESPACE,
                        "label_selector": f"{_MANAGED_BY}=chs",
                    },
                ),
                _LIST_CHS_SLICES,
                _DELETE_CHS_NAMESPACE,
            ],
        )

    def test_cancel_jobs_deletes_secrets_and_compute_domains(self, mock_logger):
        """Test cancel_jobs deletes only cancelled jobs' secrets and CDs."""
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
        for name in ("chs-job1-cd", "chs-job3-cd"):
            self.kube.add_object(
                "resource.nvidia.com",
                "computedomains",
                {"metadata": {"name": name}},
                namespace=CHS_NAMESPACE,
            )
        scheduler = KubernetesScheduler()

        scheduler.cancel_jobs(["chs-job1", "chs-job2"])

        self.assertEqual(
            self.kube.calls[0],
            Call(
                "delete_jobs",
                {
                    "object_names": ["chs-job1", "chs-job2"],
                    "namespace": CHS_NAMESPACE,
                    "wait_timeout_seconds": DELETE_WAIT_TIMEOUT_SEC,
                },
            ),
        )
        self.assertEqual(self.kube.stored_names("batch", "jobs"), ["chs-job3"])
        self.assertEqual(
            self.kube.stored_names("resource.nvidia.com", "computedomains"),
            ["chs-job3-cd"],
        )
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

    def test_cancel_jobs_cleans_dependents_when_deleting_fails(
        self, mock_logger
    ):
        self.kube.add_object(
            "",
            "secrets",
            {"metadata": {"name": "chs-job1-ssh"}},
            namespace=CHS_NAMESPACE,
        )
        self.kube.fail_next(
            "delete_jobs", k8s_client.KubeError("forbidden", status=403)
        )

        KubernetesScheduler().cancel_jobs(["chs-job1"])

        self.assertEqual(self.kube.stored_names("", "secrets"), [])
        mock_logger.error.assert_called_once()

    def test_cancel_run_raises_when_listing_jobs_fails(self, mock_logger):
        self.kube.add_object(
            "",
            "secrets",
            {
                "metadata": {
                    "name": "chs-run-a-job-ssh",
                    "labels": {_MANAGED_BY: "chs", _INSTANCE: "run-a"},
                }
            },
            namespace=CHS_NAMESPACE,
        )
        self.kube.fail_next(
            "list_jobs", k8s_client.KubeError("forbidden", status=403)
        )

        with self.assertRaisesRegex(
            ex.ChsCancellationException, "Failed to list the Jobs matching"
        ):
            KubernetesScheduler().cancel_run("run-a")

        self.assertEqual(
            self.kube.stored_names("", "secrets"), ["chs-run-a-job-ssh"]
        )
        self.assertEqual(self.kube.calls_to("delete_jobs"), [])

    def test_poll_jobs_logs_failure_to_delete_skipped_jobs(self, mock_logger):
        self.kube.add_object(
            "batch", "jobs", _job_object("job1", ["node1"]), namespace="chs"
        )
        self.kube.add_object(
            "",
            "pods",
            _accelerator_pod("other", "node1"),
            namespace=CHS_NAMESPACE,
        )
        error = k8s_client.KubeError("forbidden", status=403)
        self.kube.fail_next("delete_jobs", error)

        statuses = KubernetesScheduler().poll_jobs(["job1"])

        self.assertEqual(statuses, {"job1": "DEADLINE"})
        mock_logger.error.assert_called_once_with(
            "Encountered an error while attempting to "
            "delete active Jobs. Error: %s",
            error,
        )

    def test_cancel_all_runs_deletes_secrets(self, mock_logger):
        """Test that cancel_all_runs deletes CHS secrets when the
        namespace, which would delete them too, cannot be deleted."""
        forbidden = k8s_client.KubeError("forbidden", status=403)
        self.kube.fail_next("delete_namespace", forbidden)
        self.kube.add_object(
            "", "namespaces", {"metadata": {"name": CHS_NAMESPACE}}
        )
        self.kube.add_object(
            "",
            "secrets",
            {
                "metadata": {
                    "name": "chs-job1-ssh",
                    "labels": {_MANAGED_BY: "chs", _INSTANCE: "run-a"},
                }
            },
            namespace=CHS_NAMESPACE,
        )
        self.kube.add_object(
            "",
            "secrets",
            {"metadata": {"name": "user-secret"}},
            namespace=CHS_NAMESPACE,
        )
        scheduler = KubernetesScheduler()

        with self.assertRaisesRegex(
            ex.ChsCancellationException,
            r"^Failed to delete the CHS chs namespace\. Run "
            r"`chs cancel --all` again to retry\.$",
        ):
            scheduler.cancel_all_runs()

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
        self.assertEqual(self.kube.stored_names("", "secrets"), ["user-secret"])

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

    def test_resolve_project_id_kubernetes_falls_back_to_adc(self, mock_logger):
        """Test _resolve_project_id falls back to ADC without providerID."""
        self.kube.add_object("", "nodes", {"metadata": {"name": "node1"}})
        self.mock_adc.return_value = (mock.Mock(), "adc-fallback-project")
        scheduler = KubernetesScheduler()
        self.assertEqual(
            scheduler._resolve_project_id("node1"), "adc-fallback-project"
        )

    @mock.patch("chs.utils.gcp.query_gce_instances_info")
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
                    "zone": "us-central1-a",
                },
                "node2": {
                    "machine_type": "a3-megagpu-8g",
                    "instance_id": "67890",
                    "zone": "us-central1-b",
                },
            }
            machine_types = scheduler.detect_nodes_hardware(["node1", "node2"])
            self.assertEqual(machine_types, ["a3-megagpu-8g"])
            self.assertEqual(
                scheduler.node_to_instance_id,
                {"node1": "12345", "node2": "67890"},
            )
            self.assertEqual(
                scheduler.node_to_zone,
                {"node1": "us-central1-a", "node2": "us-central1-b"},
            )
            self.assertEqual(scheduler.project_id, "my-project")

    @mock.patch("chs.utils.gcp.query_gce_instances_info")
    def test_detect_nodes_hardware_reads_zone_and_project_on_gke(
        self, mock_query_info, mock_logger
    ):
        """GKE nodes skip the GCE API, so the providerID gives zone and
        project."""
        scheduler = KubernetesScheduler()
        for name, zone, instance_id in (
            ("node1", "us-central1-a", "111"),
            ("node2", "us-central1-b", "222"),
        ):
            self.kube.add_object(
                "",
                "nodes",
                {
                    "metadata": {
                        "name": name,
                        "labels": {
                            "node.kubernetes.io/instance-type": (
                                "a3-megagpu-8g"
                            ),
                        },
                        "annotations": {
                            "container.googleapis.com/instance_id": (
                                instance_id
                            ),
                        },
                    },
                    "spec": {
                        "providerID": f"gce://hpc-toolkit-dev/{zone}/{name}"
                    },
                    "status": {"allocatable": {"nvidia.com/gpu": "8"}},
                },
            )

        machine_types = scheduler.detect_nodes_hardware(["node1", "node2"])

        self.assertEqual(machine_types, ["a3-megagpu-8g"])
        self.assertEqual(
            scheduler.node_to_instance_id, {"node1": "111", "node2": "222"}
        )
        self.assertEqual(
            scheduler.node_to_zone,
            {"node1": "us-central1-a", "node2": "us-central1-b"},
        )
        self.assertEqual(scheduler.project_id, "hpc-toolkit-dev")
        mock_query_info.assert_not_called()

    @mock.patch("chs.utils.gcp.query_gce_instances_info")
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

    @mock.patch("chs.utils.gcp.query_gce_instances_info")
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

    @mock.patch("chs.utils.gcp.query_gce_instances_info")
    def test_detect_nodes_hardware_fallback_to_labels(
        self, mock_query_info, mock_logger
    ):
        scheduler = KubernetesScheduler()
        self.kube.add_object(
            "",
            "nodes",
            {
                "metadata": {"name": "node1"},
                "status": {"allocatable": {"nvidia.com/gpu": "8"}},
            },
        )
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
                },
                "status": {"allocatable": {"nvidia.com/gpu": "4"}},
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
        mock_open.assert_called_once_with("/tmp/exp/job.yaml", "r")

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

    def test_list_slice_names_returns_only_matching_slices(self, mock_logger):
        self.kube.add_object(
            "accelerator.gke.io", "slices", _chs_slice("pool1", "Ready")
        )
        self.kube.add_object(
            "accelerator.gke.io",
            "slices",
            {"metadata": {"name": "user-slice"}},
        )
        scheduler = KubernetesScheduler()

        self.assertEqual(
            scheduler._list_slice_names(f"{_MANAGED_BY}=chs"),
            ["chs-slice-pool1"],
        )
        self.assertEqual(self.kube.calls, [_LIST_CHS_SLICES])
        mock_logger.warning.assert_not_called()

    def test_list_slice_names_crd_not_found(self, mock_logger):
        # Without the Slice CRD there is nothing to delete, and a list
        # failing with 404 is silently ignored.
        self.kube.unregister_type("accelerator.gke.io", "slices")
        scheduler = KubernetesScheduler()

        self.assertEqual(scheduler._list_slice_names(f"{_MANAGED_BY}=chs"), [])
        self.assertEqual(
            self.kube.calls_to("list_custom_objects"), [_LIST_CHS_SLICES]
        )
        mock_logger.warning.assert_not_called()

    def test_list_slice_names_returns_none_if_listing_fails(self, mock_logger):
        error = k8s_client.KubeError("forbidden", status=403)
        self.kube.fail_next("list_custom_objects", error)
        scheduler = KubernetesScheduler()

        self.assertIsNone(scheduler._list_slice_names(f"{_MANAGED_BY}=chs"))
        mock_logger.warning.assert_called_once()
        self.assertIn(
            "Failed to list the dynamic TPU slices",
            str(mock_logger.warning.call_args.args[1]),
        )

    def test_list_slice_names_skips_slices_without_name(self, mock_logger):
        self.kube.add_object(
            "accelerator.gke.io", "slices", _chs_slice("pool1", "Ready")
        )
        self.kube.add_object(
            "accelerator.gke.io", "slices", _chs_slice("nameless", "Ready")
        )
        del self.kube.objects[
            ("accelerator.gke.io", "slices", None, "chs-slice-nameless")
        ]["metadata"]["name"]
        scheduler = KubernetesScheduler()

        self.assertEqual(
            scheduler._list_slice_names(f"{_MANAGED_BY}=chs"),
            ["chs-slice-pool1"],
        )

    def test_delete_slices_deletes_only_named_slices(self, mock_logger):
        self.kube.add_object(
            "accelerator.gke.io", "slices", _chs_slice("pool1", "Ready")
        )
        self.kube.add_object(
            "accelerator.gke.io",
            "slices",
            {"metadata": {"name": "user-slice"}},
        )
        scheduler = KubernetesScheduler()

        self.assertTrue(scheduler._delete_slices(["chs-slice-pool1"]))

        self.assertEqual(
            self.kube.calls,
            [
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
        self.assertEqual(
            self.kube.stored_names("accelerator.gke.io", "slices"),
            ["user-slice"],
        )
        mock_logger.warning.assert_not_called()

    def test_delete_slices_failure_is_logged_and_reported(self, mock_logger):
        error = k8s_client.KubeError("timed out deleting slices")
        self.kube.fail_next("delete_custom_objects", error)
        scheduler = KubernetesScheduler()

        self.assertFalse(scheduler._delete_slices(["chs-slice-pool1"]))
        mock_logger.warning.assert_called_once_with(
            "Failed to delete dynamic TPU slices. "
            "Manual cleanup may be required. Error: %s",
            error,
        )

    def test_delete_slices_without_names_makes_no_api_call(self, mock_logger):
        self.assertTrue(KubernetesScheduler()._delete_slices([]))
        self.assertEqual(self.kube.calls, [])

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

    def test_apply_scheduler_resources_creates_run_worker_service(
        self, mock_logger
    ):
        scheduler = KubernetesScheduler()
        with tempfile.TemporaryDirectory() as workspace_config_dir:
            scheduler.setup_scheduler_resources(
                workspace_config_dir, run_id="run-a"
            )
        with mock.patch.object(
            scheduler, "get_nodes_under_test", return_value=[]
        ):
            scheduler.apply_scheduler_resources()
            scheduler.apply_scheduler_resources()

        service = self.kube.objects[
            ("", "services", CHS_NAMESPACE, "chs-run-a")
        ]
        self.assertEqual(
            service["metadata"]["labels"],
            {_MANAGED_BY: "chs", _INSTANCE: "run-a"},
        )
        self.assertEqual(
            service["spec"],
            {
                "clusterIP": "None",
                "publishNotReadyAddresses": True,
                "selector": {**_WORKER_LABELS, _INSTANCE: "run-a"},
            },
        )
        self.assertEqual(self.kube.stored_names("", "services"), ["chs-run-a"])

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
        scheduler._run_id = "run-a"
        self.kube.add_object(
            "accelerator.gke.io", "slices", _chs_slice("mypool", "ACTIVE")
        )
        success, failed = scheduler._wait_for_slices_active(["mypool"])
        self.assertEqual(self.kube.calls, [_LIST_RUN_A_SLICES])
        self.assertEqual(success, ["mypool"])
        self.assertEqual(failed, [])
        mock_sleep.assert_not_called()

    @mock.patch("time.sleep")
    @mock.patch("time.time", side_effect=[0, 1])
    def test_wait_for_slices_active_failed(
        self, mock_time, mock_sleep, mock_logger
    ):
        scheduler = KubernetesScheduler()
        scheduler._run_id = "run-a"
        self.kube.add_object(
            "accelerator.gke.io", "slices", _chs_slice("mypool", "FAILED")
        )
        success, failed = scheduler._wait_for_slices_active(["mypool"])
        self.assertEqual(self.kube.calls, [_LIST_RUN_A_SLICES])
        self.assertEqual(success, [])
        self.assertEqual(failed, ["mypool"])
        mock_sleep.assert_not_called()

    @mock.patch("time.sleep")
    @mock.patch("time.time", side_effect=[0, 10, 310])
    def test_wait_for_slices_active_timeout(
        self, mock_time, mock_sleep, mock_logger
    ):
        scheduler = KubernetesScheduler()
        scheduler._run_id = "run-a"
        self.kube.add_object(
            "accelerator.gke.io", "slices", _chs_slice("mypool", "CREATING")
        )
        success, failed = scheduler._wait_for_slices_active(
            ["mypool"], timeout_sec=300
        )
        self.assertEqual(success, [])
        self.assertEqual(failed, ["mypool"])
        self.assertEqual(self.kube.calls, [_LIST_RUN_A_SLICES])
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
    def test_apply_scheduler_resources_failed_nodepools(
        self,
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

    def _two_pool_scheduler(self) -> KubernetesScheduler:
        scheduler = KubernetesScheduler()
        scheduler._run_id = "run-a"
        scheduler.workspace_config_dir = self.tmp_dir
        scheduler._resolved_nodes = ["node1", "node2"]
        scheduler._tpu_slice_data = TpuSliceData(
            pools={"good": ["node1"], "bad": ["node2"]},
            node_to_pool={"node1": "good", "node2": "bad"},
            pool_to_topology={"good": "2x2", "bad": "2x2"},
            dynamic_slice_partitions={"good": "p-good", "bad": "p-bad"},
            pool_to_accelerator={"good": "tpu7x", "bad": "tpu7x"},
        )
        return scheduler

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._group_nodes_by_slice",
        return_value={"good": ["node1"], "bad": ["node2"]},
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._wait_for_slices_active",
        return_value=(["good"], ["bad"]),
    )
    def test_apply_scheduler_resources_deletes_failed_slices(
        self, mock_wait, mock_group, mock_logger
    ):
        scheduler = self._two_pool_scheduler()

        scheduler.apply_scheduler_resources()

        self.assertEqual(scheduler._resolved_nodes, ["node1"])
        self.assertEqual(
            self.kube.calls_to("delete_custom_objects"),
            [
                Call(
                    "delete_custom_objects",
                    {
                        "group": "accelerator.gke.io",
                        "version": "v1beta1",
                        "plural": "slices",
                        "object_names": ["chs-slice-bad"],
                        "wait_timeout_seconds": 300,
                    },
                )
            ],
        )
        self.assertEqual(
            self.kube.stored_names("accelerator.gke.io", "slices"),
            ["chs-slice-good"],
        )

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._group_nodes_by_slice",
        return_value={"good": ["node1"], "bad": ["node2"]},
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._wait_for_slices_active",
        return_value=(["good"], ["bad"]),
    )
    def test_apply_scheduler_resources_logs_failure_to_delete_failed_slices(
        self, mock_wait, mock_group, mock_logger
    ):
        error = k8s_client.KubeError("forbidden", status=403)
        self.kube.fail_next("delete_custom_objects", error)
        scheduler = self._two_pool_scheduler()

        scheduler.apply_scheduler_resources()

        self.assertEqual(scheduler._resolved_nodes, ["node1"])
        self.assertEqual(
            [
                call.kwargs["object_names"]
                for call in self.kube.calls_to("delete_custom_objects")
            ],
            [["chs-slice-bad"]],
        )
        mock_logger.warning.assert_any_call(
            "Failed to delete dynamic TPU slices. "
            "Manual cleanup may be required. Error: %s",
            error,
        )

    def test_existing_slice_states_maps_partitions_to_slice_states(
        self, mock_logger
    ):
        scheduler = KubernetesScheduler()
        for name, partition_ids, reason in (
            ("active", ["part-1"], "ACTIVE"),
            ("activating", ["part-3", "part-4"], "ACTIVATING"),
            ("pending", ["part-5"], None),
        ):
            tpu_slice = _foreign_slice(name, partition_ids[0], reason)
            tpu_slice["spec"]["partitionIds"] = partition_ids
            self.kube.add_object("accelerator.gke.io", "slices", tpu_slice)
        deleting = _foreign_slice("deleting", "part-2", "ACTIVE")
        deleting["metadata"]["deletionTimestamp"] = "2023-10-01T00:00:00Z"
        self.kube.add_object("accelerator.gke.io", "slices", deleting)

        self.assertEqual(
            scheduler._existing_slice_states(),
            {
                "part-1": "ACTIVE",
                "part-3": "ACTIVATING",
                "part-4": "ACTIVATING",
                "part-5": None,
            },
        )
        self.assertEqual(
            self.kube.calls,
            [
                Call(
                    "list_custom_objects",
                    {
                        "group": "accelerator.gke.io",
                        "version": "v1beta1",
                        "plural": "slices",
                        "label_selector": None,
                    },
                )
            ],
        )

    def test_existing_slice_states_skips_other_runs_slices(self, mock_logger):
        scheduler = KubernetesScheduler()
        scheduler._run_id = "run-a"
        for name, labels, partition_id in (
            ("user-slice", {}, "part-user"),
            ("own", {_MANAGED_BY: "chs", _INSTANCE: "run-a"}, "part-own"),
            ("other", {_MANAGED_BY: "chs", _INSTANCE: "run-b"}, "part-b"),
        ):
            tpu_slice = _foreign_slice(name, partition_id, "ACTIVE")
            tpu_slice["metadata"]["labels"] = labels
            self.kube.add_object("accelerator.gke.io", "slices", tpu_slice)

        self.assertEqual(
            scheduler._existing_slice_states(),
            {"part-user": "ACTIVE", "part-own": "ACTIVE"},
        )

    def test_existing_slice_states_skips_slices_that_failed_to_form(
        self, mock_logger
    ):
        for name, reason in (
            ("failed", "FAILED"),
            ("creation-failed", "SliceCreationFailed"),
            ("active", "ACTIVE"),
            ("pending", None),
        ):
            self.kube.add_object(
                "accelerator.gke.io",
                "slices",
                _foreign_slice(name, f"part-{name}", reason),
            )

        self.assertEqual(
            KubernetesScheduler()._existing_slice_states(),
            {"part-active": "ACTIVE", "part-pending": None},
        )

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._wait_for_slices_active",
        return_value=(["dynpool"], []),
    )
    def test_provision_dynamic_slices_replaces_foreign_slices_that_failed(
        self, mock_wait, mock_logger
    ):
        self.kube.add_object(
            "accelerator.gke.io",
            "slices",
            _foreign_slice("user-cube", "dynpool-cube", "FAILED"),
        )
        scheduler = KubernetesScheduler()
        scheduler._run_id = "run-a"
        scheduler.workspace_config_dir = self.tmp_dir
        scheduler._tpu_slice_data = _dynamic_pool_data({"d1": "sub-d1"})

        success, failed = scheduler._provision_dynamic_slices(
            {"dynpool": ["d1"]}
        )

        self.assertEqual((success, failed), (["dynpool"], []))
        self.assertIn(
            "chs-slice-dynpool",
            self.kube.stored_names("accelerator.gke.io", "slices"),
        )
        mock_wait.assert_called_once_with(["dynpool"], 300)

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._wait_for_slices_active"
    )
    def test_provision_dynamic_slices_reuses_active_foreign_slices(
        self, mock_wait, mock_logger
    ):
        self.kube.add_object(
            "accelerator.gke.io",
            "slices",
            _foreign_slice("user-cube", "dynpool-cube", "ACTIVE"),
        )
        scheduler = KubernetesScheduler()
        scheduler._run_id = "run-a"
        scheduler.workspace_config_dir = self.tmp_dir
        scheduler._tpu_slice_data = _dynamic_pool_data({"d1": "sub-d1"})

        success, failed = scheduler._provision_dynamic_slices(
            {"dynpool": ["d1"]}
        )

        self.assertEqual((success, failed), (["dynpool"], []))
        self.assertEqual(self.kube.calls_to("create_custom_object"), [])
        mock_wait.assert_not_called()

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._wait_for_slices_active"
    )
    def test_provision_dynamic_slices_reports_partitions_of_inactive_slices(
        self, mock_wait, mock_logger
    ):
        states = {
            "d1": "ACTIVATING",
            "d2": "ACTIVE_DEGRADED",
            "d3": "INCOMPLETE",
            "d4": None,
        }
        for node, state in states.items():
            self.kube.add_object(
                "accelerator.gke.io",
                "slices",
                _foreign_slice(f"user-{node}", f"sub-{node}", state),
            )
        scheduler = self._subslice_scheduler(*states)

        success, failed = scheduler._provision_dynamic_slices(
            {node: [node] for node in states}
        )

        self.assertEqual((success, failed), ([], list(states)))
        self.assertEqual(self.kube.calls_to("create_custom_object"), [])
        self.assertEqual(self.kube.calls_to("delete_custom_objects"), [])
        self.assertCountEqual(
            self.kube.stored_names("accelerator.gke.io", "slices"),
            [f"user-{node}" for node in states],
        )
        mock_wait.assert_not_called()
        self.assertEqual(
            mock_logger.warning.call_args_list,
            [
                mock.call(
                    "Cannot provision a dynamic slice for '%s': its TPU "
                    "partition is held by an existing slice in state %s.",
                    node,
                    state or "unknown",
                )
                for node, state in states.items()
            ],
        )

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._wait_for_slices_active",
        return_value=(["d2"], []),
    )
    def test_provision_dynamic_slices_reuses_only_active_single_host_slices(
        self, mock_wait, mock_logger
    ):
        for node, state in (("d1", "ACTIVE"), ("d2", "FAILED")):
            self.kube.add_object(
                "accelerator.gke.io",
                "slices",
                _foreign_slice(f"user-{node}", f"sub-{node}", state),
            )
        scheduler = self._subslice_scheduler("d1", "d2")

        success, failed = scheduler._provision_dynamic_slices(
            {"d1": ["d1"], "d2": ["d2"]}
        )

        self.assertEqual((success, failed), (["d2", "d1"], []))
        self.assertEqual(
            [
                call.kwargs["body"]["metadata"]["name"]
                for call in self.kube.calls_to("create_custom_object")
            ],
            ["chs-slice-d2"],
        )
        mock_wait.assert_called_once_with(["d2"], 300)
        self.assertEqual(self.kube.calls_to("delete_custom_objects"), [])

    def test_existing_slice_states_without_listable_slices(self, mock_logger):
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
            self.assertEqual(scheduler._existing_slice_states(), {})
        with self.subTest("Slice CRD not installed"):
            self.kube.unregister_type("accelerator.gke.io", "slices")
            self.assertEqual(scheduler._existing_slice_states(), {})

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._existing_slice_states"
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._wait_for_slices_active"
    )
    def test_provision_dynamic_slices_all_reused(
        self, mock_wait, mock_existing_states, mock_logger
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
        mock_existing_states.return_value = {"part1": "ACTIVE"}
        success, failed = scheduler._provision_dynamic_slices(target_pools)
        self.assertEqual(success, ["pool1"])
        self.assertEqual(failed, [])
        # Nothing is applied.
        self.assertEqual(self.kube.calls, [])
        mock_wait.assert_not_called()

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._existing_slice_states"
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._wait_for_slices_active"
    )
    def test_provision_dynamic_slices_none_reused(
        self, mock_wait, mock_existing_states, mock_logger
    ):
        scheduler = KubernetesScheduler()
        scheduler._run_id = "run-a"
        target_pools = {"pool1": ["node1"]}

        scheduler.workspace_config_dir = self.tmp_dir
        scheduler._tpu_slice_data = TpuSliceData(
            pools=target_pools,
            node_to_pool={"node1": "pool1"},
            pool_to_topology={"pool1": "2x2"},
            dynamic_slice_partitions={"pool1": "part1"},
            pool_to_accelerator={"pool1": "tpu-v4-podslice"},
        )
        mock_existing_states.return_value = {}
        mock_wait.return_value = (["pool1"], [])
        success, failed = scheduler._provision_dynamic_slices(target_pools)
        self.assertEqual(success, ["pool1"])
        self.assertEqual(failed, [])
        applied = self.kube.objects[
            ("accelerator.gke.io", "slices", None, "chs-slice-pool1")
        ]
        self.assertEqual(len(self.kube.calls), 1)
        self.assertEqual(self.kube.calls[0].method, "create_custom_object")
        self.assertEqual(self.kube.calls[0].kwargs["plural"], "slices")
        self.assertEqual(
            applied["metadata"]["labels"],
            {_MANAGED_BY: "chs", _INSTANCE: "run-a"},
        )
        self.assertEqual(applied["spec"]["partitionIds"], ["part1"])
        mock_wait.assert_called_once_with(["pool1"], 300)

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._existing_slice_states",
        return_value={},
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._wait_for_slices_active",
        return_value=(["pool1", "pool2"], []),
    )
    def test_provision_dynamic_slices_writes_separated_documents(
        self, mock_wait, mock_existing_states, mock_logger
    ):
        scheduler = KubernetesScheduler()
        scheduler._run_id = "run-a"
        scheduler.workspace_config_dir = self.tmp_dir
        target_pools = {"pool1": ["node1"], "pool2": ["node2"]}
        scheduler._tpu_slice_data = TpuSliceData(
            pools=target_pools,
            pool_to_topology={"pool1": "2x2", "pool2": "2x2"},
            dynamic_slice_partitions={"pool1": "part1", "pool2": "part2"},
            pool_to_accelerator={
                "pool1": "tpu-v4-podslice",
                "pool2": "tpu-v4-podslice",
            },
        )

        scheduler._provision_dynamic_slices(target_pools)

        with open(os.path.join(self.tmp_dir, "chs-slices.yaml")) as f:
            content = f.read()
        self.assertIn("\n---\n", content)
        self.assertEqual(
            [s["metadata"]["name"] for s in yaml.safe_load_all(content)],
            ["chs-slice-pool1", "chs-slice-pool2"],
        )

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._existing_slice_states",
        return_value={},
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._wait_for_slices_active"
    )
    def test_provision_dynamic_slices_never_takes_over_existing_slice(
        self, mock_wait, mock_existing_states, mock_logger
    ):
        existing = {
            "metadata": {
                "name": "chs-slice-pool1",
                "labels": {_MANAGED_BY: "chs", _INSTANCE: "run-b"},
            },
            "spec": {"partitionIds": ["part1"]},
        }
        self.kube.add_object("accelerator.gke.io", "slices", existing)
        scheduler = KubernetesScheduler()
        scheduler._run_id = "run-a"
        scheduler.workspace_config_dir = self.tmp_dir
        scheduler._tpu_slice_data = TpuSliceData(
            pools={"pool1": ["node1"]},
            dynamic_slice_partitions={"pool1": "part1"},
        )

        with self.assertRaisesRegex(
            ChsSchedulerException,
            "Dynamic slice 'chs-slice-pool1' already exists; another CHS run "
            "may be using pool1\\.",
        ):
            scheduler._provision_dynamic_slices({"pool1": ["node1"]})

        stored = self.kube.objects[
            ("accelerator.gke.io", "slices", None, "chs-slice-pool1")
        ]
        self.assertEqual(stored["metadata"]["labels"][_INSTANCE], "run-b")
        self.assertEqual(self.kube.calls_to("apply_custom_object"), [])
        mock_wait.assert_not_called()

    def _subslice_scheduler(self, *nodes: str) -> KubernetesScheduler:
        scheduler = KubernetesScheduler()
        scheduler._run_id = "run-a"
        scheduler.workspace_config_dir = self.tmp_dir
        scheduler._tpu_slice_data = _dynamic_pool_data(
            {node: f"sub-{node}" for node in nodes}
        )
        scheduler._subslice_nodes = set(nodes)
        return scheduler

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._existing_slice_states",
        return_value={},
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._wait_for_slices_active",
        return_value=(["d1", "d2"], []),
    )
    def test_provision_dynamic_slices_creates_a_subslice_per_node(
        self, mock_wait, mock_existing_states, mock_logger
    ):
        scheduler = self._subslice_scheduler("d1", "d2")

        success, failed = scheduler._provision_dynamic_slices(
            {"d1": ["d1"], "d2": ["d2"]}
        )

        self.assertEqual((success, failed), (["d1", "d2"], []))
        for node in ("d1", "d2"):
            stored = self.kube.objects[
                ("accelerator.gke.io", "slices", None, f"chs-slice-{node}")
            ]
            self.assertEqual(
                stored["metadata"]["labels"],
                {_MANAGED_BY: "chs", _INSTANCE: "run-a"},
            )
            self.assertEqual(
                stored["spec"],
                {
                    "type": "tpu7x",
                    "topology": DYNAMIC_SINGLE_HOST_TOPOLOGY,
                    "partitionIds": [f"sub-{node}"],
                },
            )
        mock_wait.assert_called_once_with(["d1", "d2"], 300)

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._existing_slice_states",
        return_value={},
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._wait_for_slices_active",
        return_value=(["d3"], []),
    )
    def test_provision_dynamic_slices_reports_subslices_that_fail_to_create(
        self, mock_wait, mock_existing_states, mock_logger
    ):
        _add_chs_slice(self.kube, "d1", "run-b", ["sub-d1"])
        self.kube.fail_next(
            "create_custom_object",
            None,
            k8s_client.KubeError("invalid slice", status=422),
        )
        scheduler = self._subslice_scheduler("d1", "d2", "d3")

        with mock.patch("chs.schedulers.kubernetes._API_MAX_CONCURRENCY", 1):
            success, failed = scheduler._provision_dynamic_slices(
                {"d1": ["d1"], "d2": ["d2"], "d3": ["d3"]}
            )

        self.assertEqual((success, failed), (["d3"], ["d1", "d2"]))
        mock_wait.assert_called_once_with(["d3"], 300)
        self.assertEqual(len(self.kube.calls_to("create_custom_object")), 3)
        self.assertEqual(self.kube.calls_to("delete_custom_objects"), [])
        stored = self.kube.objects[
            ("accelerator.gke.io", "slices", None, "chs-slice-d1")
        ]
        self.assertEqual(stored["metadata"]["labels"][_INSTANCE], "run-b")
        warnings = [str(c.args[1]) for c in mock_logger.warning.call_args_list]
        self.assertEqual(len(warnings), 2)
        self.assertIn("another CHS run may be using d1.", warnings[0])
        self.assertIn(
            "Failed to create dynamic slice 'chs-slice-d2'", warnings[1]
        )

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._existing_slice_states",
        return_value={},
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._wait_for_slices_active"
    )
    @mock.patch("time.sleep")
    def test_provision_dynamic_slices_raises_if_a_pool_slice_fails_to_create(
        self, mock_sleep, mock_wait, mock_existing_states, mock_logger
    ):
        self.kube.fail_always(
            "create_custom_object",
            k8s_client.KubeError("internal error", status=500),
        )
        scheduler = KubernetesScheduler()
        scheduler._run_id = "run-a"
        scheduler.workspace_config_dir = self.tmp_dir
        scheduler._tpu_slice_data = _dynamic_pool_data({"d1": "sub-d1"})

        with self.assertRaisesRegex(
            ChsSchedulerException,
            "^Failed to create dynamic slice 'chs-slice-dynpool'",
        ):
            scheduler._provision_dynamic_slices({"dynpool": ["d1"]})
        mock_wait.assert_not_called()
        self.assertEqual(
            len(self.kube.calls_to("create_custom_object")), _APPLY_ATTEMPTS
        )

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._group_nodes_by_slice",
        return_value={"d1": ["d1"], "d2": ["d2"], "staticpool": ["s1"]},
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._provision_dynamic_slices",
        return_value=(["d2"], ["d1"]),
    )
    def test_apply_scheduler_resources_drops_failed_subslices(
        self, mock_provision, mock_group, mock_logger
    ):
        for node, reason in (("d1", "FAILED"), ("d2", "ACTIVE")):
            self.kube.add_object(
                "accelerator.gke.io", "slices", _chs_slice(node, reason)
            )
        scheduler = self._subslice_scheduler("d1", "d2")
        scheduler._resolved_nodes = ["d1", "d2", "s1"]

        scheduler.apply_scheduler_resources()

        self.assertEqual(scheduler._resolved_nodes, ["d2", "s1"])
        self.assertEqual(scheduler._subslice_nodes, {"d2"})
        self.assertIn("dynpool", scheduler._tpu_slice_data.pools)
        self.assertEqual(
            self.kube.stored_names("accelerator.gke.io", "slices"),
            ["chs-slice-d1", "chs-slice-d2"],
        )
        self.assertEqual(self.kube.calls_to("list_custom_objects"), [])
        self.assertEqual(self.kube.calls_to("delete_custom_objects"), [])
        mock_logger.warning.assert_called_once_with(
            "Excluded node(s) %s from testing because their %s "
            "single-host slices could not be provisioned. Tests will "
            "run on the remaining nodes.",
            "d1",
            DYNAMIC_SINGLE_HOST_TOPOLOGY,
        )

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._group_nodes_by_slice",
        return_value={"d1": ["d1"], "d2": ["d2"], "d3": ["d3"]},
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._wait_for_slices_active",
        return_value=(["d3"], ["d2"]),
    )
    def test_apply_scheduler_resources_deletes_only_own_failed_subslices(
        self, mock_wait, mock_group, mock_logger
    ):
        foreign = _chs_slice("d1", "ACTIVE", run_id="run-b")
        foreign["spec"] = {"partitionIds": ["sub-d1"]}
        self.kube.add_object("accelerator.gke.io", "slices", foreign)
        scheduler = self._subslice_scheduler("d1", "d2", "d3")
        scheduler._resolved_nodes = ["d1", "d2", "d3"]

        scheduler.apply_scheduler_resources()

        self.assertEqual(scheduler._resolved_nodes, ["d3"])
        self.assertEqual(scheduler._subslice_nodes, {"d3"})
        self.assertEqual(
            [
                call.kwargs["object_names"]
                for call in self.kube.calls_to("delete_custom_objects")
            ],
            [["chs-slice-d2"]],
        )
        self.assertCountEqual(
            self.kube.stored_names("accelerator.gke.io", "slices"),
            ["chs-slice-d1", "chs-slice-d3"],
        )
        stored = self.kube.objects[
            ("accelerator.gke.io", "slices", None, "chs-slice-d1")
        ]
        self.assertEqual(stored["metadata"]["labels"][_INSTANCE], "run-b")

    def test_drop_failed_slices_raises_if_every_subslice_failed(
        self, mock_logger
    ):
        scheduler = self._subslice_scheduler("d1", "d2")
        scheduler._resolved_nodes = ["d1", "d2"]

        with self.assertRaisesRegex(
            ChsSchedulerException,
            "^All targeted dynamic slices failed to provision: d1, d2\\.$",
        ):
            scheduler._drop_failed_slices(
                ["d1", "d2"], {"d1": ["d1"], "d2": ["d2"]}
            )
        self.assertEqual(scheduler._resolved_nodes, [])
        self.assertEqual(scheduler._subslice_nodes, set())

    @mock.patch("time.sleep")
    def test_with_retries_retries_throttling_and_server_errors(
        self, mock_sleep, mock_logger
    ):
        call = mock.Mock(
            side_effect=[
                k8s_client.KubeError("Too Many Requests", status=429),
                k8s_client.KubeError("Internal Server Error", status=500),
                k8s_client.KubeError("connection reset"),
                None,
            ]
        )

        _with_retries(call, "Slice 'chs-slice-d1'")

        self.assertEqual(call.call_count, 4)
        self.assertEqual(
            [c.args[0] for c in mock_sleep.call_args_list], [1, 2, 4]
        )

    @mock.patch("time.sleep")
    def test_with_retries_does_not_retry_client_errors(
        self, mock_sleep, mock_logger
    ):
        for status in (409, 403, 422):
            with self.subTest(status=status):
                error = k8s_client.KubeError("rejected", status=status)
                call = mock.Mock(side_effect=error)

                with self.assertRaises(k8s_client.KubeError) as cm:
                    _with_retries(call, "Slice 'chs-slice-d1'")

                self.assertIs(cm.exception, error)
                call.assert_called_once_with()
        mock_sleep.assert_not_called()

    @mock.patch("time.sleep")
    def test_with_retries_gives_up_after_the_last_attempt(
        self, mock_sleep, mock_logger
    ):
        error = k8s_client.KubeError("Service Unavailable", status=503)
        call = mock.Mock(side_effect=error)

        with self.assertRaises(k8s_client.KubeError) as cm:
            _with_retries(call, "Slice 'chs-slice-d1'")

        self.assertIs(cm.exception, error)
        self.assertEqual(call.call_count, _APPLY_ATTEMPTS)
        self.assertEqual(mock_sleep.call_count, _APPLY_ATTEMPTS - 1)

    def _slice_manifests(self, *keys: str) -> dict[str, dict]:
        return {
            key: {
                "apiVersion": "accelerator.gke.io/v1beta1",
                "kind": "Slice",
                "metadata": {"name": _slice_name(key)},
            }
            for key in keys
        }

    @mock.patch("time.sleep")
    def test_create_slice_retries_throttled_creates(
        self, mock_sleep, mock_logger
    ):
        self.kube.fail_next(
            "create_custom_object",
            k8s_client.KubeError("Too Many Requests", status=429),
        )
        manifest = self._slice_manifests("d1")["d1"]

        KubernetesScheduler()._create_slice(manifest, "d1")

        self.assertEqual(len(self.kube.calls_to("create_custom_object")), 2)
        mock_sleep.assert_called_once_with(1)
        self.assertEqual(
            self.kube.stored_names("accelerator.gke.io", "slices"),
            ["chs-slice-d1"],
        )

    @mock.patch("time.sleep")
    def test_create_slice_accepts_a_conflict_with_its_own_retried_create(
        self, mock_sleep, mock_logger
    ):
        _add_chs_slice(self.kube, "d1", "run-a", ["sub-d1"])
        self.kube.fail_next(
            "create_custom_object",
            k8s_client.KubeError("connection reset", status=500),
        )
        scheduler = KubernetesScheduler()
        scheduler._run_id = "run-a"

        scheduler._create_slice(self._slice_manifests("d1")["d1"], "d1")

        self.assertEqual(len(self.kube.calls_to("create_custom_object")), 2)
        self.assertEqual(
            self.kube.calls_to("list_custom_objects"), [_LIST_RUN_A_SLICES]
        )
        mock_logger.warning.assert_not_called()

    @mock.patch("time.sleep")
    def test_create_slice_rejects_a_conflict_with_another_runs_slice(
        self, mock_sleep, mock_logger
    ):
        _add_chs_slice(self.kube, "d1", "run-b", ["sub-d1"])
        self.kube.fail_next(
            "create_custom_object",
            k8s_client.KubeError("connection reset", status=500),
        )
        scheduler = KubernetesScheduler()
        scheduler._run_id = "run-a"

        with self.assertRaisesRegex(
            ChsSchedulerException,
            "^Dynamic slice 'chs-slice-d1' already exists; another CHS run "
            "may be using d1\\.",
        ):
            scheduler._create_slice(self._slice_manifests("d1")["d1"], "d1")

        self.assertEqual(len(self.kube.calls_to("create_custom_object")), 2)
        stored = self.kube.objects[
            ("accelerator.gke.io", "slices", None, "chs-slice-d1")
        ]
        self.assertEqual(stored["metadata"]["labels"][_INSTANCE], "run-b")

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._existing_slice_states",
        return_value={},
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._wait_for_slices_active",
        return_value=(["d1"], []),
    )
    @mock.patch("time.sleep")
    def test_provision_dynamic_slices_waits_on_its_own_retried_slice(
        self, mock_sleep, mock_wait, mock_existing_states, mock_logger
    ):
        _add_chs_slice(self.kube, "d1", "run-a", ["sub-d1"])
        self.kube.fail_next(
            "create_custom_object",
            k8s_client.KubeError("connection reset", status=500),
        )
        scheduler = self._subslice_scheduler("d1")

        success, failed = scheduler._provision_dynamic_slices({"d1": ["d1"]})

        self.assertEqual((success, failed), (["d1"], []))
        mock_wait.assert_called_once_with(["d1"], 300)
        self.assertEqual(self.kube.calls_to("delete_custom_objects"), [])
        mock_logger.warning.assert_not_called()

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._existing_slice_states",
        return_value={},
    )
    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._wait_for_slices_active",
        return_value=([], []),
    )
    @mock.patch("time.sleep")
    def test_provision_dynamic_slices_reports_conflicts_with_other_runs(
        self, mock_sleep, mock_wait, mock_existing_states, mock_logger
    ):
        _add_chs_slice(self.kube, "d1", "run-b", ["sub-d1"])
        self.kube.fail_next(
            "create_custom_object",
            k8s_client.KubeError("connection reset", status=500),
        )
        scheduler = self._subslice_scheduler("d1")

        success, failed = scheduler._provision_dynamic_slices({"d1": ["d1"]})

        self.assertEqual((success, failed), ([], ["d1"]))
        mock_wait.assert_called_once_with([], 300)
        self.assertEqual(self.kube.calls_to("delete_custom_objects"), [])
        self.assertIn(
            "another CHS run may be using d1.",
            str(mock_logger.warning.call_args.args[1]),
        )

    def test_create_slices_returns_failed_subslices_in_manifest_order(
        self, mock_logger
    ):
        for key in ("d1", "d3"):
            _add_chs_slice(self.kube, key, "run-b", [f"sub-{key}"])
        keys = ("d4", "d3", "d2", "d1")
        scheduler = self._subslice_scheduler(*keys)

        failed = scheduler._create_slices(self._slice_manifests(*keys))

        self.assertEqual(failed, ["d3", "d1"])
        self.assertCountEqual(
            self.kube.stored_names("accelerator.gke.io", "slices"),
            ["chs-slice-d1", "chs-slice-d2", "chs-slice-d3", "chs-slice-d4"],
        )
        for key in ("d1", "d3"):
            stored = self.kube.objects[
                ("accelerator.gke.io", "slices", None, f"chs-slice-{key}")
            ]
            self.assertEqual(stored["metadata"]["labels"][_INSTANCE], "run-b")
        self.assertEqual(mock_logger.warning.call_count, 2)

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
    def test__get_tpu_slices_missing_gke_tpu_accelerator_label(
        self, mock_fetch, mock_logger
    ):
        """Test _get_tpu_slices falls back to allocatable TPU resources."""
        mock_fetch.return_value = [
            {
                "metadata": {
                    "name": "tpu-node",
                    "labels": {
                        "cloud.google.com/gke-nodepool": "pool-tpu",
                        "cloud.google.com/gke-tpu-topology": "2x2",
                    },
                },
                "status": {"allocatable": {"google.com/tpu": "4"}},
            },
            {
                "metadata": {
                    "name": "gpu-node",
                    "labels": {
                        "cloud.google.com/gke-nodepool": "pool-gpu",
                    },
                },
                "status": {"allocatable": {"nvidia.com/gpu": "8"}},
            },
        ]

        scheduler = KubernetesScheduler()
        res = scheduler._get_tpu_slices()

        self.assertEqual(res.pools, {"pool-tpu": ["tpu-node"]})
        self.assertEqual(res.all_tpu_nodes, {"tpu-node"})

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._fetch_raw_node_resources"
    )
    def test__get_tpu_slices_marks_single_host_pools(
        self, mock_fetch, mock_logger
    ):
        mock_fetch.return_value = [
            {
                "metadata": {
                    "name": name,
                    "labels": {
                        "cloud.google.com/gke-nodepool": pool,
                        "cloud.google.com/gke-tpu-topology": topology,
                    },
                },
                "status": {"allocatable": {"google.com/tpu": "4"}},
            }
            for name, pool, topology in (
                ("n1", "single", "2x2x1"),
                ("n2", "single", "2x2x1"),
                ("n3", "multi", "2x2x2"),
                ("n4", "multi", "2x2x2"),
            )
        ] + [
            {
                "metadata": {
                    "name": "d1",
                    "labels": {
                        "cloud.google.com/gke-nodepool": "dyn",
                        "cloud.google.com/gke-tpu-topology": "2x2x1",
                        "cloud.google.com/gke-accelerator-topology-mode": (
                            "PROVISION_ONLY"
                        ),
                        (
                            "cloud.google.com/gke-tpu-partition-"
                            f"{SUPPORTED_DYNAMIC_SLICE_TOPOLOGY}-state"
                        ): "HEALTHY",
                    },
                },
                "status": {"allocatable": {"google.com/tpu": "4"}},
            }
        ]

        res = KubernetesScheduler()._get_tpu_slices()

        self.assertIn("dyn", res.pools)
        self.assertEqual(res.single_host_pools, {"single"})

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
        self.assertEqual(res.provision_only_nodes, set())
        self.assertEqual(res.single_host_slices, {})
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
    def test__get_tpu_slices_records_single_host_slices_with_nodes_arg(
        self, mock_fetch, mock_logger
    ):
        mock_fetch.return_value = [
            _dynamic_tpu_node("healthy", "HEALTHY"),
            _dynamic_tpu_node("unhealthy", "UNHEALTHY", pool_state="HEALTHY"),
            _dynamic_tpu_node("unlabeled", None),
        ]

        res = KubernetesScheduler(
            nodes="healthy,unhealthy,unlabeled"
        )._get_tpu_slices()

        self.assertEqual(
            res.provision_only_nodes, {"healthy", "unhealthy", "unlabeled"}
        )
        self.assertEqual(
            res.single_host_slices,
            {"healthy": _single_host_slice("sub-healthy")},
        )
        self.assertEqual(res.pools, {})
        self.assertEqual(res.node_to_pool, {})
        self.assertEqual(res.dynamic_slice_partitions, {})
        mock_logger.warning.assert_not_called()

    @mock.patch(
        "chs.schedulers.kubernetes.KubernetesScheduler._fetch_raw_node_resources"
    )
    def test__get_tpu_slices_records_no_single_host_slices_with_nodepools_arg(
        self, mock_fetch, mock_logger
    ):
        mock_fetch.return_value = [
            _dynamic_tpu_node("d1", "HEALTHY", pool_state="HEALTHY"),
            _dynamic_tpu_node("d2", "UNHEALTHY", pool_state="HEALTHY"),
        ]

        res = KubernetesScheduler(nodepools="dynpool")._get_tpu_slices()

        self.assertEqual(res.pools, {"dynpool": ["d1", "d2"]})
        self.assertEqual(res.dynamic_slice_partitions, {"dynpool": "cube-1"})
        self.assertEqual(res.provision_only_nodes, set())
        self.assertEqual(res.single_host_slices, {})
        mock_logger.warning.assert_not_called()

    def test_record_pool_node_records_static_node_by_its_topology(
        self, mock_logger
    ):
        data = TpuSliceData()
        unhealthy_pools = set()
        node = _in_tpu_slice(_tpu_node("s1", "staticpool"), "2x2")

        _record_pool_node(data, unhealthy_pools, "s1", _labels(node))

        self.assertEqual(
            data,
            TpuSliceData(
                pools={"staticpool": ["s1"]},
                node_to_pool={"s1": "staticpool"},
                pool_to_topology={"staticpool": "2x2"},
            ),
        )
        self.assertEqual(unhealthy_pools, set())

    def test_record_pool_node_purges_static_pool_with_a_node_without_topology(
        self, mock_logger
    ):
        data = TpuSliceData()
        unhealthy_pools = set()
        nodes = (
            _in_tpu_slice(_tpu_node("s1", "staticpool"), "2x2"),
            _tpu_node("s2", "staticpool"),
            _in_tpu_slice(_tpu_node("s3", "staticpool"), "2x2"),
        )

        for node in nodes:
            _record_pool_node(
                data, unhealthy_pools, node["metadata"]["name"], _labels(node)
            )

        self.assertEqual(data, TpuSliceData())
        self.assertEqual(unhealthy_pools, {"staticpool"})
        mock_logger.warning.assert_not_called()

    def test_record_pool_node_records_healthy_dynamic_pool_as_a_cube(
        self, mock_logger
    ):
        data = TpuSliceData()
        unhealthy_pools = set()
        node = _dynamic_tpu_node("d1", "UNHEALTHY", pool_state="HEALTHY")

        _record_pool_node(data, unhealthy_pools, "d1", _labels(node))

        self.assertEqual(
            data,
            TpuSliceData(
                pools={"dynpool": ["d1"]},
                node_to_pool={"d1": "dynpool"},
                pool_to_topology={"dynpool": SUPPORTED_DYNAMIC_SLICE_TOPOLOGY},
                dynamic_slice_partitions={"dynpool": "cube-1"},
                pool_to_accelerator={"dynpool": "tpu7x"},
            ),
        )
        self.assertEqual(unhealthy_pools, set())

    def test_record_pool_node_warns_once_about_unhealthy_dynamic_pool(
        self, mock_logger
    ):
        data = TpuSliceData()
        unhealthy_pools = set()

        for name in ("d1", "d2"):
            node = _dynamic_tpu_node(name, "HEALTHY", pool_state="UNHEALTHY")
            _record_pool_node(data, unhealthy_pools, name, _labels(node))

        self.assertEqual(data, TpuSliceData())
        self.assertEqual(unhealthy_pools, {"dynpool"})
        mock_logger.warning.assert_called_once_with(
            "Skipping nodepool '%s' because the %s "
            "topology is currently unhealthy.",
            "dynpool",
            SUPPORTED_DYNAMIC_SLICE_TOPOLOGY,
        )

    def test_record_subslice_node_records_only_healthy_single_host_partitions(
        self, mock_logger
    ):
        data = TpuSliceData()
        missing_id = _dynamic_tpu_node("missing-id", "HEALTHY")
        del _labels(missing_id)[
            f"cloud.google.com/gke-tpu-partition-"
            f"{DYNAMIC_SINGLE_HOST_TOPOLOGY}-id"
        ]
        nodes = (
            _dynamic_tpu_node("healthy", "HEALTHY"),
            _dynamic_tpu_node("unhealthy", "UNHEALTHY", pool_state="HEALTHY"),
            missing_id,
        )

        for node in nodes:
            _record_subslice_node(data, node["metadata"]["name"], _labels(node))

        self.assertEqual(
            data,
            TpuSliceData(
                provision_only_nodes={"healthy", "unhealthy", "missing-id"},
                single_host_slices={
                    "healthy": _single_host_slice("sub-healthy")
                },
            ),
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

    def test_poll_jobs_skips_job_whose_node_is_taken(self, mock_logger):
        """Test Jobs on a taken node are skipped and the others start."""
        for name, node in (
            ("job1", "node1"),
            ("job2", "node2"),
            ("job3", "node1"),
        ):
            self.kube.add_object(
                "batch",
                "jobs",
                _job_object(name, [node]),
                namespace=CHS_NAMESPACE,
            )
        self.kube.add_object(
            "", "pods", _accelerator_pod("train", "node1"), namespace="team-a"
        )
        scheduler = KubernetesScheduler()

        statuses = scheduler.poll_jobs(["job1", "job2", "job3"])

        self.assertEqual(
            statuses,
            {"job1": "DEADLINE", "job2": "PENDING", "job3": "DEADLINE"},
        )
        self.assertEqual(
            [call.kwargs["name"] for call in self.kube.calls_to("patch_job")],
            ["job2"],
        )
        self.assertEqual(len(self.kube.calls_to("list_pods")), 2)
        self.assertEqual(
            mock_logger.warning.call_args_list,
            [
                mock.call(
                    "Job '%s' was not started because node %s is now used by "
                    "another workload (pod %s).",
                    name,
                    "node1",
                    "team-a/train",
                )
                for name in ("job1", "job3")
            ],
        )

    def test_poll_jobs_starts_job_on_free_node_of_skipped_job(
        self, mock_logger
    ):
        """Test a skipped Job does not hold its free nodes in that poll."""
        self.kube.add_object(
            "batch",
            "jobs",
            _job_object("pair", ["node1", "node2"]),
            namespace=CHS_NAMESPACE,
        )
        self.kube.add_object(
            "batch",
            "jobs",
            _job_object("single", ["node1"]),
            namespace=CHS_NAMESPACE,
        )
        self.kube.add_object(
            "", "pods", _accelerator_pod("train", "node2"), namespace="team-a"
        )
        scheduler = KubernetesScheduler()

        statuses = scheduler.poll_jobs(["pair", "single"])

        self.assertEqual(statuses, {"pair": "DEADLINE", "single": "PENDING"})
        self.assertEqual(
            [call.kwargs["name"] for call in self.kube.calls_to("patch_job")],
            ["single"],
        )

    def test_poll_jobs_checks_each_job_when_its_turn_comes(self, mock_logger):
        """Test a Job that cannot start yet is checked when its turn comes."""
        self.kube.add_object(
            "batch",
            "jobs",
            _job_object("running", ["node2"], suspended=False),
            namespace=CHS_NAMESPACE,
        )
        self.kube.add_object(
            "batch",
            "jobs",
            _job_object("job1", ["node1"]),
            namespace=CHS_NAMESPACE,
        )
        self.kube.add_object(
            "batch",
            "jobs",
            _job_object("pair", ["node1", "node2"]),
            namespace=CHS_NAMESPACE,
        )
        self.kube.add_object(
            "", "pods", _accelerator_pod("train", "node1"), namespace="team-a"
        )
        scheduler = KubernetesScheduler()

        first = scheduler.poll_jobs(["job1", "pair"])
        del self.kube.objects[("batch", "jobs", CHS_NAMESPACE, "running")]
        del self.kube.objects[("", "pods", "team-a", "train")]
        second = scheduler.poll_jobs(["pair"])

        self.assertEqual(first, {"job1": "DEADLINE", "pair": "PENDING"})
        self.assertEqual(second, {"pair": "PENDING"})
        self.assertEqual(
            [call.kwargs["name"] for call in self.kube.calls_to("patch_job")],
            ["pair"],
        )

    def test_poll_jobs_skips_all_waiting_jobs_of_taken_node(self, mock_logger):
        """Test the queued Jobs of a taken node are skipped in one poll."""
        for name in ("job1", "job2", "job3"):
            self.kube.add_object(
                "batch",
                "jobs",
                _job_object(name, ["node1"]),
                namespace=CHS_NAMESPACE,
            )
        self.kube.add_object(
            "", "pods", _accelerator_pod("train", "node1"), namespace="team-a"
        )
        scheduler = KubernetesScheduler()

        statuses = scheduler.poll_jobs(["job1", "job2", "job3"])

        self.assertEqual(
            statuses,
            {"job1": "DEADLINE", "job2": "DEADLINE", "job3": "DEADLINE"},
        )
        self.assertEqual(self.kube.calls_to("patch_job"), [])
        self.assertEqual(len(self.kube.calls_to("list_pods")), 1)

    def test_poll_jobs_deletes_skipped_jobs(self, mock_logger):
        """Test a skipped Job and its SSH Secret are deleted."""
        for name in ("job1", "job2"):
            self.kube.add_object(
                "batch",
                "jobs",
                _job_object(name, [name.replace("job", "node")]),
                namespace=CHS_NAMESPACE,
            )
            self.kube.add_object(
                "",
                "secrets",
                {"metadata": {"name": f"{name}-ssh"}},
                namespace=CHS_NAMESPACE,
            )
        self.kube.add_object(
            "", "pods", _accelerator_pod("train", "node1"), namespace="team-a"
        )
        scheduler = KubernetesScheduler()

        scheduler.poll_jobs(["job1", "job2"])

        self.assertEqual(
            self.kube.stored_names("batch", "jobs", CHS_NAMESPACE), ["job2"]
        )
        self.assertEqual(
            self.kube.stored_names("", "secrets", CHS_NAMESPACE), ["job2-ssh"]
        )

    def test_poll_jobs_skips_multi_node_job_if_one_node_is_taken(
        self, mock_logger
    ):
        """Test a gang is not started with one of its nodes taken."""
        self.kube.add_object(
            "batch",
            "jobs",
            _job_object("pair", ["node1", "node2"]),
            namespace=CHS_NAMESPACE,
        )
        self.kube.add_object(
            "", "pods", _accelerator_pod("train", "node2"), namespace="team-a"
        )
        scheduler = KubernetesScheduler()

        self.assertEqual(scheduler.poll_jobs(["pair"]), {"pair": "DEADLINE"})
        self.assertEqual(self.kube.calls_to("patch_job"), [])
        mock_logger.warning.assert_called_once_with(
            "Job '%s' was not started because node %s is now used by "
            "another workload (pod %s).",
            "pair",
            "node2",
            "team-a/train",
        )

    def test_poll_jobs_ignores_chs_pods_on_nodes(self, mock_logger):
        """Test CHS's own pods, such as terminating ones, do not block."""
        self.kube.add_object(
            "batch",
            "jobs",
            _job_object("job1", ["node1"]),
            namespace=CHS_NAMESPACE,
        )
        self.kube.add_object(
            "",
            "pods",
            _accelerator_pod("old-test", "node1", dict(_WORKER_LABELS)),
            namespace=CHS_NAMESPACE,
        )
        scheduler = KubernetesScheduler()

        statuses = scheduler.poll_jobs(["job1"])

        self.assertEqual(statuses, {"job1": "PENDING"})
        self.assertEqual(
            [call.kwargs["name"] for call in self.kube.calls_to("patch_job")],
            ["job1"],
        )
        mock_logger.warning.assert_not_called()

    def test_poll_jobs_counts_unlabeled_pods_in_chs_namespace(
        self, mock_logger
    ):
        """Test a pod in the chs namespace without the CHS labels blocks."""
        for labels in ({}, {"app.kubernetes.io/component": "worker"}):
            with self.subTest(labels=labels):
                kube = install_fake_client(self)
                kube.add_object(
                    "batch",
                    "jobs",
                    _job_object("job1", ["node1"]),
                    namespace=CHS_NAMESPACE,
                )
                kube.add_object(
                    "",
                    "pods",
                    _accelerator_pod("other", "node1", labels),
                    namespace=CHS_NAMESPACE,
                )
                scheduler = KubernetesScheduler()

                statuses = scheduler.poll_jobs(["job1"])

                self.assertEqual(statuses, {"job1": "DEADLINE"})
                self.assertEqual(kube.calls_to("patch_job"), [])

    @mock.patch("time.sleep", return_value=None)
    def test_poll_jobs_starts_jobs_if_node_check_fails(
        self, mock_sleep, mock_logger
    ):
        """Test a node check that fails after retries never blocks a test."""
        self.kube.add_object(
            "batch",
            "jobs",
            _job_object("job1", ["node1"]),
            namespace=CHS_NAMESPACE,
        )
        error = k8s_client.KubeError("Too Many Requests", status=429)
        self.kube.fail_next("list_pods", error, error, error)
        scheduler = KubernetesScheduler()

        statuses = scheduler.poll_jobs(["job1"])

        self.assertEqual(statuses, {"job1": "PENDING"})
        self.assertEqual(
            [call.kwargs["name"] for call in self.kube.calls_to("patch_job")],
            ["job1"],
        )
        self.assertEqual(len(self.kube.calls_to("list_pods")), 3)
        self.assertEqual(mock_sleep.call_count, 2)
        mock_logger.warning.assert_not_called()
        self.assertIn(
            "Failed to check node '%s' for other workloads, so it is "
            "treated as free. Error: %s",
            [call.args[0] for call in mock_logger.debug.call_args_list],
        )

    @mock.patch("time.sleep", return_value=None)
    def test_poll_jobs_retries_failed_node_check(self, mock_sleep, mock_logger):
        """Test a failed node check is retried before the Job starts."""
        self.kube.add_object(
            "batch",
            "jobs",
            _job_object("job1", ["node1"]),
            namespace=CHS_NAMESPACE,
        )
        self.kube.add_object(
            "", "pods", _accelerator_pod("train", "node1"), namespace="team-a"
        )
        self.kube.fail_next(
            "list_pods",
            k8s_client.KubeError("Too Many Requests", status=429),
        )
        scheduler = KubernetesScheduler()

        statuses = scheduler.poll_jobs(["job1"])

        self.assertEqual(statuses, {"job1": "DEADLINE"})
        self.assertEqual(self.kube.calls_to("patch_job"), [])
        self.assertEqual(len(self.kube.calls_to("list_pods")), 2)
        mock_sleep.assert_called_once()

    def test_poll_jobs_checks_nodes_only_when_starting_jobs(self, mock_logger):
        """Test polls that start nothing make no node checks."""
        self.kube.add_object(
            "batch",
            "jobs",
            _job_object("running", ["node1"], suspended=False),
            namespace=CHS_NAMESPACE,
        )
        self.kube.add_object(
            "batch",
            "jobs",
            _job_object("waiting", ["node1"]),
            namespace=CHS_NAMESPACE,
        )
        scheduler = KubernetesScheduler()

        scheduler.poll_jobs(["running", "waiting"])

        self.assertEqual(self.kube.calls_to("list_pods"), [])
        self.assertEqual(self.kube.calls_to("patch_job"), [])

    def test_get_unschedulable_jobs_returns_job_blocked_by_other_workload(
        self, mock_logger
    ):
        """Test a waiting pod whose node another workload uses counts now."""
        self.kube.add_object(
            "",
            "pods",
            _waiting_pod("job1-1", "job1", "node1"),
            namespace=CHS_NAMESPACE,
        )
        self.kube.add_object(
            "", "pods", _accelerator_pod("train", "node1"), namespace="team-a"
        )
        scheduler = KubernetesScheduler()

        self.assertEqual(scheduler.get_unschedulable_jobs(["job1"]), ["job1"])
        mock_logger.warning.assert_called_once_with(
            "Job '%s' cannot schedule all of its pods because node %s "
            "is now used by another workload (pod %s).",
            "job1",
            "node1",
            "team-a/train",
        )

    def test_get_unschedulable_jobs_waits_for_chs_pods_to_leave(
        self, mock_logger
    ):
        """Test a pod waiting only for CHS's own pod gets the full timeout."""
        self.kube.add_object(
            "",
            "pods",
            _waiting_pod("job1-1", "job1", "node1"),
            namespace=CHS_NAMESPACE,
        )
        self.kube.add_object(
            "",
            "pods",
            _accelerator_pod("old-test", "node1", dict(_WORKER_LABELS)),
            namespace=CHS_NAMESPACE,
        )
        scheduler = KubernetesScheduler()

        self.assertEqual(scheduler.get_unschedulable_jobs(["job1"]), [])
        mock_logger.warning.assert_not_called()

    @mock.patch("time.sleep", return_value=None)
    def test_get_unschedulable_jobs_waits_if_node_check_fails(
        self, mock_sleep, mock_logger
    ):
        """Test a failed node check falls back to the normal timeout."""
        self.kube.add_object(
            "",
            "pods",
            _waiting_pod("job1-1", "job1", "node1"),
            namespace=CHS_NAMESPACE,
        )
        self.kube.add_object(
            "", "pods", _accelerator_pod("train", "node1"), namespace="team-a"
        )
        error = k8s_client.KubeError("Too Many Requests", status=429)
        self.kube.fail_next("list_pods", None, None, error, error, error)
        scheduler = KubernetesScheduler()

        self.assertEqual(scheduler.get_unschedulable_jobs(["job1"]), [])
        self.assertEqual(mock_sleep.call_count, 2)
        mock_logger.warning.assert_not_called()
        self.assertIn(
            "Failed to check node '%s' for other workloads, so it is "
            "treated as free. Error: %s",
            [call.args[0] for call in mock_logger.debug.call_args_list],
        )

    def test_get_unschedulable_jobs_waits_if_job_pods_cannot_be_listed(
        self, mock_logger
    ):
        """Test no node is checked if the waiting Job's pods are unknown."""
        self.kube.add_object(
            "",
            "pods",
            _waiting_pod("job1-1", "job1", "node1"),
            namespace=CHS_NAMESPACE,
        )
        self.kube.add_object(
            "", "pods", _accelerator_pod("train", "node1"), namespace="team-a"
        )
        self.kube.fail_next(
            "list_pods",
            None,
            k8s_client.KubeError("Too Many Requests", status=429),
        )
        scheduler = KubernetesScheduler()

        self.assertEqual(scheduler.get_unschedulable_jobs(["job1"]), [])
        self.assertEqual(len(self.kube.calls_to("list_pods")), 2)
        mock_logger.warning.assert_not_called()
        self.assertIn(
            "Failed to list the pods of waiting Jobs to check their nodes. "
            "Error: %s",
            [call.args[0] for call in mock_logger.debug.call_args_list],
        )

    def test_get_unschedulable_jobs_checks_only_nodes_without_job_pods(
        self, mock_logger
    ):
        """Test only the nodes a waiting pod can still use are checked."""
        for index, node in enumerate(("node1", "node2")):
            self.kube.add_object(
                "",
                "pods",
                _accelerator_pod(
                    f"gang-{index}",
                    node,
                    {
                        "batch.kubernetes.io/job-name": "gang",
                        **_WORKER_LABELS,
                    },
                ),
                namespace=CHS_NAMESPACE,
            )
        waiting = _waiting_pod("gang-2", "gang", "node1")
        waiting["spec"] = _job_object("gang", ["node1", "node2", "node3"])[
            "spec"
        ]["template"]["spec"]
        self.kube.add_object("", "pods", waiting, namespace=CHS_NAMESPACE)
        self.kube.add_object(
            "", "pods", _accelerator_pod("train", "node3"), namespace="team-a"
        )
        scheduler = KubernetesScheduler()

        self.assertEqual(scheduler.get_unschedulable_jobs(["gang"]), ["gang"])
        node_checks = [
            call.kwargs["field_selector"]
            for call in self.kube.calls_to("list_pods")
            if call.kwargs.get("field_selector", "").startswith(
                "spec.nodeName="
            )
        ]
        self.assertEqual(
            node_checks,
            [
                "spec.nodeName=node3,status.phase!=Succeeded,status.phase!=Failed"
            ],
        )

    def test_poll_jobs_keeps_failure_of_job_that_waited_briefly(
        self, mock_logger
    ):
        """Test a real failure after a short wait is reported as is."""
        self.kube.add_object(
            "",
            "pods",
            _waiting_pod("job1-1", "job1", "node1"),
            namespace=CHS_NAMESPACE,
        )
        scheduler = KubernetesScheduler()
        self.assertEqual(scheduler.get_unschedulable_jobs(["job1"]), [])
        self.kube.add_object(
            "batch",
            "jobs",
            _finished_job("job1", "node1", "Failed", "PodFailurePolicy"),
            namespace=CHS_NAMESPACE,
        )

        self.assertEqual(scheduler.poll_jobs(["job1"]), {"job1": "UNKNOWN"})

    def _write_experiment(self, name: str, *objects: dict) -> str:
        experiment_dir = os.path.join(self.tmp_dir, "experiments", name)
        os.makedirs(experiment_dir)
        with open(os.path.join(experiment_dir, "job.yaml"), "w") as f:
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


def _run_job(
    name: str, run_id: str, nodes: list[str], **kwargs: Any
) -> dict[str, Any]:
    """Returns a Job of the run pinned to nodes."""
    job = _job_object(name, nodes, **kwargs)
    job["metadata"]["labels"] = {_MANAGED_BY: "chs", _INSTANCE: run_id}
    return job


_PARTITION_ID_LABEL = (
    f"cloud.google.com/gke-tpu-partition-{SUPPORTED_DYNAMIC_SLICE_TOPOLOGY}-id"
)


def _add_chs_slice(
    kube: Any, pool: str, run_id: str | None, partition_ids: list[str]
) -> None:
    """Stores a CHS dynamic Slice of the run over the partitions."""
    labels = {_MANAGED_BY: "chs"}
    if run_id is not None:
        labels[_INSTANCE] = run_id
    kube.add_object(
        "accelerator.gke.io",
        "slices",
        {
            "metadata": {"name": f"chs-slice-{pool}", "labels": labels},
            "spec": {"partitionIds": partition_ids},
        },
    )


def _partition_node(name: str, partition_id: str) -> dict[str, Any]:
    """Returns a Node object in the TPU partition."""
    return {
        "metadata": {
            "name": name,
            "labels": {_PARTITION_ID_LABEL: partition_id},
        }
    }


@mock.patch("chs.schedulers.kubernetes.logger")
class TestKubernetesRunsTargetingNodes(unittest.TestCase):
    """Tests finding the other runs whose Jobs or Slices target nodes."""

    def setUp(self):
        super().setUp()
        self.kube = install_fake_client(self)
        self.scheduler = KubernetesScheduler()

    def _add_job(self, job: dict[str, Any]) -> None:
        self.kube.add_object("batch", "jobs", job, namespace=CHS_NAMESPACE)

    def test_find_runs_targeting_nodes_includes_all_existing_jobs(
        self, mock_logger
    ):
        self._add_job(_run_job("chs-run-b-nccl-a1b2c3", "run-b", ["n1"]))
        self._add_job(
            _run_job("chs-run-b-dcgm-b2c3d4", "run-b", ["n3"], suspended=False)
        )
        running = _run_job(
            "chs-run-c-gpu-c3d4e5", "run-c", ["n2"], suspended=False
        )
        running["status"] = {"active": 1}
        self._add_job(running)
        for name, condition, node in (
            ("chs-run-d-x-d4e5f6", "Complete", "n4"),
            ("chs-run-d-y-e5f6a7", "Failed", "n5"),
        ):
            finished = _finished_job(name, node, condition, "")
            finished["metadata"]["labels"] = {
                _MANAGED_BY: "chs",
                _INSTANCE: "run-d",
            }
            self._add_job(finished)
        self._add_job(_run_job("chs-run-e-far-a7b8c9", "run-e", ["n9"]))

        conflicts = self.scheduler.find_runs_targeting_nodes(
            ["n1", "n2", "n3", "n4", "n5", "n6"]
        )

        # Finished Jobs keep their nodes until their run is torn down, so a
        # run between tiers still claims them.
        self.assertEqual(
            conflicts,
            {"run-b": ["n1", "n3"], "run-c": ["n2"], "run-d": ["n4", "n5"]},
        )

    def test_find_runs_targeting_nodes_includes_terminating_jobs(
        self, mock_logger
    ):
        job = _run_job("chs-run-b-x-a1b2c3", "run-b", ["n1"], suspended=False)
        job["metadata"]["deletionTimestamp"] = "2026-10-14T09:00:00Z"
        job["status"] = {"active": 1}
        self._add_job(job)

        self.assertEqual(
            self.scheduler.find_runs_targeting_nodes(["n1"]),
            {"run-b": ["n1"]},
        )

    def test_find_runs_targeting_nodes_attributes_jobs_by_instance_label(
        self, mock_logger
    ):
        self._add_job(_run_job("chs-x-a1b2c3", "run-b", ["n1"]))
        self._add_job(_run_job("chs-run-a-x-c3d4e5", "run-a-2", ["n3"]))
        self._add_job(_job_object("chs-run-z-unlabeled-d4e5f6", ["n4"]))

        self.assertEqual(
            self.scheduler.find_runs_targeting_nodes(["n1", "n3", "n4"]),
            {"run-b": ["n1"], "run-a-2": ["n3"]},
        )

    def test_find_runs_targeting_nodes_ignores_non_chs_and_nameless_jobs(
        self, mock_logger
    ):
        third_party = _job_object("other-app-job", ["n1"])
        third_party["metadata"]["labels"] = {_INSTANCE: "my-release"}
        self._add_job(third_party)
        self._add_job(_run_job("nameless", "run-b", ["n2"]))
        del self.kube.objects[("batch", "jobs", CHS_NAMESPACE, "nameless")][
            "metadata"
        ]["name"]
        self._add_job(_run_job("chs-run-c-x-a1b2c3", "run-c", ["n3"]))

        self.assertEqual(
            self.scheduler.find_runs_targeting_nodes(["n1", "n2", "n3"]),
            {"run-c": ["n3"]},
        )
        self.assertEqual(
            self.kube.calls_to("list_jobs"),
            [
                Call(
                    "list_jobs",
                    {
                        "namespace": CHS_NAMESPACE,
                        "label_selector": f"{_MANAGED_BY}=chs",
                    },
                )
            ],
        )

    def test_find_runs_targeting_nodes_raises_if_listing_fails(
        self, mock_logger
    ):
        for error in (
            k8s_client.KubeError("connection refused"),
            k8s_client.KubeError("forbidden", status=403),
        ):
            with self.subTest(status=error.status):
                self.kube.fail_next("list_jobs", error)
                with self.assertRaisesRegex(
                    ChsSchedulerException, "cannot verify"
                ):
                    self.scheduler.find_runs_targeting_nodes(["n1"])

    def test_find_runs_targeting_nodes_includes_slices_of_other_runs(
        self, mock_logger
    ):
        for name, partition_id in (
            ("t1", "part-1"),
            ("t2", "part-1"),
            ("t3", "part-2"),
            ("t4", "part-3"),
            ("t5", "part-4"),
        ):
            self.kube.add_object(
                "", "nodes", _partition_node(name, partition_id)
            )
        _add_chs_slice(self.kube, "pool-b", "run-b", ["part-1"])
        _add_chs_slice(self.kube, "pool-far", "run-c", ["part-4"])
        _add_chs_slice(self.kube, "pool-unlabeled", None, ["part-2"])

        conflicts = self.scheduler.find_runs_targeting_nodes(
            ["t1", "t2", "t3", "t4"]
        )

        self.assertEqual(conflicts, {"run-b": ["t1", "t2"]})

    def test_find_runs_targeting_nodes_includes_subslices_of_other_runs(
        self, mock_logger
    ):
        for name in ("t1", "t2"):
            self.kube.add_object(
                "",
                "nodes",
                _tpu_node(
                    name,
                    "dynpool",
                    {
                        DYNAMIC_SINGLE_HOST_TOPOLOGY: f"sub-{name}",
                        SUPPORTED_DYNAMIC_SLICE_TOPOLOGY: "cube-1",
                    },
                ),
            )
        _add_chs_slice(self.kube, "t1", "run-b", ["sub-t1"])

        self.assertEqual(
            self.scheduler.find_runs_targeting_nodes(["t1", "t2"]),
            {"run-b": ["t1"]},
        )

    def test_find_runs_targeting_nodes_includes_slices_that_failed_to_form(
        self, mock_logger
    ):
        for name, run_id, reason in (
            ("t1", "run-b", "FAILED"),
            ("t2", "run-c", "SliceCreationFailed"),
            ("t3", "run-d", "ACTIVE"),
        ):
            self.kube.add_object(
                "",
                "nodes",
                _tpu_node(
                    name,
                    "dynpool",
                    {DYNAMIC_SINGLE_HOST_TOPOLOGY: f"sub-{name}"},
                ),
            )
            tpu_slice = _chs_slice(name, reason, run_id=run_id)
            tpu_slice["spec"] = {"partitionIds": [f"sub-{name}"]}
            self.kube.add_object("accelerator.gke.io", "slices", tpu_slice)

        self.assertEqual(
            self.scheduler.find_runs_targeting_nodes(["t1", "t2", "t3"]),
            {"run-b": ["t1"], "run-c": ["t2"], "run-d": ["t3"]},
        )

    def test_find_runs_targeting_nodes_uses_cached_nodes(self, mock_logger):
        self.scheduler._resolved_node_resources = [
            _partition_node("t1", "part-1")
        ]
        _add_chs_slice(self.kube, "pool-b", "run-b", ["part-1"])

        self.assertEqual(
            self.scheduler.find_runs_targeting_nodes(["t1"]),
            {"run-b": ["t1"]},
        )
        self.assertEqual(self.kube.calls_to("list_nodes"), [])

    def test_find_runs_targeting_nodes_without_slice_crd(self, mock_logger):
        self.kube.unregister_type("accelerator.gke.io", "slices")

        self.assertEqual(self.scheduler.find_runs_targeting_nodes(["n1"]), {})

    def test_find_runs_targeting_nodes_raises_if_slice_listing_fails(
        self, mock_logger
    ):
        self.kube.fail_next(
            "list_custom_objects", k8s_client.KubeError("forbidden", status=403)
        )

        with self.assertRaisesRegex(ChsSchedulerException, "cannot verify"):
            self.scheduler.find_runs_targeting_nodes(["n1"])

    def test_find_runs_targeting_nodes_without_nodes(self, mock_logger):
        self.assertEqual(self.scheduler.find_runs_targeting_nodes([]), {})
        self.assertEqual(self.kube.calls, [])

    def test_list_slices_without_slice_crd_returns_no_slices(self, mock_logger):
        self.kube.unregister_type("accelerator.gke.io", "slices")

        self.assertEqual(self.scheduler._list_slices(), [])

    def test_list_slices_raises_scheduler_exception_on_api_error(
        self, mock_logger
    ):
        self.kube.fail_next(
            "list_custom_objects", k8s_client.KubeError("forbidden", status=403)
        )

        with self.assertRaisesRegex(
            ChsSchedulerException, "Failed to list the dynamic TPU slices"
        ):
            self.scheduler._list_slices()

    def test_cancel_run_deletes_the_rest_if_listing_slices_fails(
        self, mock_logger
    ):
        self._add_job(_run_job("chs-run-a-x-a1b2c3", "run-a", ["n1"]))
        self.kube.fail_next(
            "list_custom_objects", k8s_client.KubeError("forbidden", status=403)
        )

        with self.assertRaisesRegex(
            ex.ChsCancellationException,
            "^Failed to delete the dynamic slices of CHS run 'run-a'\\.",
        ):
            self.scheduler.cancel_run("run-a")
        self.assertEqual(self.kube.stored_names("batch", "jobs"), [])
        self.assertEqual(len(self.kube.calls_to("delete_service")), 1)


@mock.patch("chs.schedulers.kubernetes.logger")
class TestKubernetesRunScopedCleanup(unittest.TestCase):
    """Tests that cleanup of a run leaves concurrent runs alone."""

    def setUp(self):
        super().setUp()
        self.kube = install_fake_client(self)
        self.scheduler = KubernetesScheduler()

    def _add_slice(self, pool: str, run_id: str | None) -> None:
        _add_chs_slice(self.kube, pool, run_id, [f"part-{pool}"])

    def _slices(self) -> list[str]:
        return sorted(self.kube.stored_names("accelerator.gke.io", "slices"))

    def _add_run_objects(self, job: str, run_id: str) -> None:
        """Stores a Job of the run with its Secret and ComputeDomain."""
        self.kube.add_object(
            "batch",
            "jobs",
            _run_job(job, run_id, ["n1"]),
            namespace=CHS_NAMESPACE,
        )
        labels = {_MANAGED_BY: "chs", _INSTANCE: run_id}
        self.kube.add_object(
            "",
            "secrets",
            {"metadata": {"name": f"{job}-ssh", "labels": labels}},
            namespace=CHS_NAMESPACE,
        )
        self.kube.add_object(
            "resource.nvidia.com",
            "computedomains",
            {"metadata": {"name": f"{job}-cd", "labels": labels}},
            namespace=CHS_NAMESPACE,
        )

    def test_cancel_run_deletes_only_objects_of_that_exact_run(
        self, mock_logger
    ):
        self.kube.create_namespace_if_not_exists(name=CHS_NAMESPACE)
        for run_id in ("run-a", "run-a-x1y2"):
            self.kube.add_object(
                "",
                "services",
                {"metadata": {"name": f"chs-{run_id}"}},
                namespace=CHS_NAMESPACE,
            )
        # run-a-x1y2's Job names start with run-a's Job name prefix.
        self._add_run_objects("chs-run-a-nccl-a1b2c3", "run-a")
        self._add_run_objects("chs-run-a-x1y2-nccl-b2c3d4", "run-a-x1y2")
        self._add_slice("pool-a", "run-a")
        self._add_slice("pool-b", "run-a-x1y2")
        self._add_slice("pool-unlabeled", None)

        self.scheduler.cancel_run("run-a")

        other = "chs-run-a-x1y2-nccl-b2c3d4"
        self.assertEqual(self.kube.stored_names("batch", "jobs"), [other])
        self.assertEqual(
            self.kube.stored_names("", "secrets"), [f"{other}-ssh"]
        )
        self.assertEqual(
            self.kube.stored_names("resource.nvidia.com", "computedomains"),
            [f"{other}-cd"],
        )
        self.assertEqual(
            self._slices(), ["chs-slice-pool-b", "chs-slice-pool-unlabeled"]
        )
        self.assertEqual(
            self.kube.stored_names("", "services"), ["chs-run-a-x1y2"]
        )
        self.assertIn(
            ("", "namespaces", None, CHS_NAMESPACE), self.kube.objects
        )
        mock_logger.warning.assert_not_called()

    def test_cancel_run_cleans_up_even_if_deleting_jobs_fails(
        self, mock_logger
    ):
        self.kube.create_namespace_if_not_exists(name=CHS_NAMESPACE)
        self.kube.add_object(
            "",
            "services",
            {"metadata": {"name": "chs-run-a"}},
            namespace=CHS_NAMESPACE,
        )
        self._add_run_objects("chs-run-a-nccl-a1b2c3", "run-a")
        self._add_slice("pool-a", "run-a")
        self.kube.fail_next(
            "delete_jobs", k8s_client.KubeError("forbidden", status=403)
        )

        with self.assertRaisesRegex(
            ex.ChsCancellationException,
            r"^Failed to delete the Jobs of CHS run 'run-a'\. Run "
            r"`chs cancel run-a` again to retry\.$",
        ):
            self.scheduler.cancel_run("run-a")

        self.assertEqual(
            self.kube.stored_names("batch", "jobs"), ["chs-run-a-nccl-a1b2c3"]
        )
        self.assertEqual(self.kube.stored_names("", "secrets"), [])
        self.assertEqual(
            self.kube.stored_names("resource.nvidia.com", "computedomains"), []
        )
        self.assertEqual(self._slices(), [])
        self.assertEqual(self.kube.stored_names("", "services"), [])
        mock_logger.error.assert_called_once()

    def test_cancel_run_lists_every_kind_that_failed_to_delete(
        self, mock_logger
    ):
        self._add_run_objects("chs-run-a-nccl-a1b2c3", "run-a")
        self._add_slice("pool-a", "run-a")
        error = k8s_client.KubeError("forbidden", status=403)
        self.kube.fail_next("delete_jobs", error)
        self.kube.fail_next("delete_secret", error)
        self.kube.fail_next("delete_service", error)

        with self.assertRaisesRegex(
            ex.ChsCancellationException,
            r"^Failed to delete the Jobs, Secrets, worker Service of CHS run "
            r"'run-a'\.",
        ):
            self.scheduler.cancel_run("run-a")

        self.assertEqual(
            self.kube.stored_names("resource.nvidia.com", "computedomains"), []
        )
        self.assertEqual(self._slices(), [])

    def test_cancel_run_raises_if_listing_secrets_fails(self, mock_logger):
        self._add_run_objects("chs-run-a-nccl-a1b2c3", "run-a")
        self.kube.fail_next(
            "list_secrets", k8s_client.KubeError("forbidden", status=403)
        )

        with self.assertRaisesRegex(
            ex.ChsCancellationException,
            r"^Failed to delete the Secrets of CHS run 'run-a'\.",
        ):
            self.scheduler.cancel_run("run-a")

        self.assertEqual(self.kube.stored_names("batch", "jobs"), [])
        self.assertEqual(
            self.kube.stored_names("", "secrets"), ["chs-run-a-nccl-a1b2c3-ssh"]
        )

    def test_cancel_run_raises_if_slice_deletion_fails(self, mock_logger):
        self._add_slice("pool-a", "run-a")
        self.kube.fail_next(
            "delete_custom_objects",
            None,
            k8s_client.KubeError("timed out deleting slices"),
        )

        with self.assertRaisesRegex(
            ex.ChsCancellationException,
            r"^Failed to delete the dynamic slices of CHS run 'run-a'\.",
        ):
            self.scheduler.cancel_run("run-a")

        self.assertEqual(
            self.kube.calls_to("delete_custom_objects")[1].kwargs["plural"],
            "slices",
        )

    def test_cancel_run_returns_whether_the_run_had_jobs_or_slices(
        self, mock_logger
    ):
        self._add_run_objects("chs-run-a-nccl-a1b2c3", "run-a")
        self._add_slice("pool-b", "run-b")

        self.assertTrue(self.scheduler.cancel_run("run-a"))
        self.assertTrue(self.scheduler.cancel_run("run-b"))
        self.assertFalse(self.scheduler.cancel_run("run-a"))
        self.assertFalse(self.scheduler.cancel_run("run-unknown"))

    def test_cancel_run_logs_service_deletion_errors(self, mock_logger):
        error = k8s_client.KubeError("forbidden", status=403)
        self.kube.fail_next("delete_service", error)

        with self.assertRaisesRegex(
            ex.ChsCancellationException,
            r"^Failed to delete the worker Service of CHS run 'run-a'\.",
        ):
            self.scheduler.cancel_run("run-a")

        mock_logger.warning.assert_called_once_with(
            "Failed to delete the worker Service of run '%s'. Error: %s",
            "run-a",
            error,
        )

    def test_cancel_run_without_compute_domain_crd(self, mock_logger):
        self.kube.unregister_type("resource.nvidia.com", "computedomains")

        self.scheduler.cancel_run("run-a")

        mock_logger.warning.assert_not_called()

    def test_cancel_run_skips_objects_without_name(self, mock_logger):
        self._add_run_objects("chs-run-a-nccl-a1b2c3", "run-a")
        self._add_run_objects("nameless", "run-a")
        for key in (
            ("", "secrets", CHS_NAMESPACE, "nameless-ssh"),
            (
                "resource.nvidia.com",
                "computedomains",
                CHS_NAMESPACE,
                "nameless-cd",
            ),
        ):
            del self.kube.objects[key]["metadata"]["name"]

        self.scheduler.cancel_run("run-a")

        self.assertEqual(
            [
                call.kwargs["name"]
                for call in self.kube.calls_to("delete_secret")
            ],
            ["chs-run-a-nccl-a1b2c3-ssh"],
        )
        self.assertEqual(
            [
                call.kwargs["object_names"]
                for call in self.kube.calls_to("delete_custom_objects")
                if call.kwargs["plural"] == "computedomains"
            ],
            [["chs-run-a-nccl-a1b2c3-cd"]],
        )

    def test_cancel_run_deletes_secrets_independently(self, mock_logger):
        self._add_run_objects("chs-run-a-x-a1b2c3", "run-a")
        self._add_run_objects("chs-run-a-y-b2c3d4", "run-a")
        error = k8s_client.KubeError("forbidden", status=403)
        self.kube.fail_next("delete_secret", error)

        with self.assertRaisesRegex(
            ex.ChsCancellationException,
            r"^Failed to delete the Secrets of CHS run 'run-a'\.",
        ):
            self.scheduler.cancel_run("run-a")

        self.assertEqual(len(self.kube.calls_to("delete_secret")), 2)
        self.assertEqual(len(self.kube.stored_names("", "secrets")), 1)
        self.assertEqual(
            self.kube.stored_names("resource.nvidia.com", "computedomains"), []
        )
        mock_logger.warning.assert_called_once_with(
            "Failed to delete Kubernetes secret '%s'. Error: %s",
            mock.ANY,
            error,
        )

    def test_cancel_jobs_deletes_dependents_independently(self, mock_logger):
        for job in ("chs-run-a-x-a1b2c3", "chs-run-a-y-b2c3d4"):
            self._add_run_objects(job, "run-a")
        error = k8s_client.KubeError("forbidden", status=403)
        self.kube.fail_next("delete_secret", error)

        self.scheduler.cancel_jobs(["chs-run-a-x-a1b2c3", "chs-run-a-y-b2c3d4"])

        # The first Secret fails, but the second Secret and both
        # ComputeDomains are still deleted.
        self.assertEqual(
            self.kube.stored_names("", "secrets"), ["chs-run-a-x-a1b2c3-ssh"]
        )
        self.assertEqual(
            self.kube.stored_names("resource.nvidia.com", "computedomains"), []
        )
        mock_logger.warning.assert_called_once_with(
            "Failed to delete Kubernetes secret '%s'. Error: %s",
            "chs-run-a-x-a1b2c3-ssh",
            error,
        )

    def test_cancel_jobs_logs_compute_domain_deletion_errors(self, mock_logger):
        self._add_run_objects("chs-run-a-x-a1b2c3", "run-a")
        error = k8s_client.KubeError("forbidden", status=403)
        self.kube.fail_next("delete_custom_objects", error)

        self.scheduler.cancel_jobs(["chs-run-a-x-a1b2c3"])

        self.assertEqual(self.kube.stored_names("", "secrets"), [])
        mock_logger.warning.assert_called_once_with(
            "Failed to delete Kubernetes ComputeDomains. Error: %s", error
        )

    def test_cancel_all_runs_deletes_jobs_namespace_and_all_slices(
        self, mock_logger
    ):
        self.kube.create_namespace_if_not_exists(name=CHS_NAMESPACE)
        self._add_run_objects("chs-run-a-nccl-a1b2c3", "run-a")
        self._add_slice("pool-a", "run-a")
        self._add_slice("pool-unlabeled", None)

        self.scheduler.cancel_all_runs()

        self.assertEqual(
            [
                call.kwargs["object_names"]
                for call in self.kube.calls_to("delete_jobs")
            ],
            [["chs-run-a-nccl-a1b2c3"]],
        )
        self.assertEqual(self._slices(), [])
        self.assertEqual(self.kube.stored_names("", "namespaces"), [])

    def test_cancel_all_runs_attempts_every_deletion_and_lists_failures(
        self, mock_logger
    ):
        self.kube.create_namespace_if_not_exists(name=CHS_NAMESPACE)
        self._add_run_objects("chs-run-a-nccl-a1b2c3", "run-a")
        self._add_slice("pool-a", "run-a")
        error = k8s_client.KubeError("forbidden", status=403)
        self.kube.fail_next("delete_jobs", error)
        self.kube.fail_next("delete_namespace", error)

        with self.assertRaisesRegex(
            ex.ChsCancellationException,
            r"^Failed to delete the CHS Jobs, chs namespace\. Run "
            r"`chs cancel --all` again to retry\.$",
        ):
            self.scheduler.cancel_all_runs()

        self.assertEqual(self._slices(), [])
        self.assertEqual(self.kube.stored_names("", "secrets"), [])
        self.assertEqual(
            self.kube.stored_names("resource.nvidia.com", "computedomains"), []
        )

    def test_cancel_all_runs_deletes_the_rest_if_listing_slices_fails(
        self, mock_logger
    ):
        self.kube.create_namespace_if_not_exists(name=CHS_NAMESPACE)
        self._add_run_objects("chs-run-a-nccl-a1b2c3", "run-a")
        self.kube.fail_next(
            "list_custom_objects", k8s_client.KubeError("forbidden", status=403)
        )

        with self.assertRaisesRegex(
            ex.ChsCancellationException,
            "^Failed to delete the CHS dynamic slices\\.",
        ):
            self.scheduler.cancel_all_runs()

        self.assertEqual(self.kube.stored_names("batch", "jobs"), [])
        self.assertEqual(len(self.kube.calls_to("delete_namespace")), 1)

    def test_apply_scheduler_resources_deletes_no_existing_slices(
        self, mock_logger
    ):
        self._add_slice("pool-a", "run-old")
        self._add_slice("pool-b", "run-b")
        self.scheduler._tpu_slice_data = TpuSliceData(
            pools={"pool-a": ["a1"], "pool-b": ["b1"]},
            dynamic_slice_partitions={"pool-a": "part-a", "pool-b": "part-b"},
        )
        with (
            mock.patch.object(
                self.scheduler, "get_nodes_under_test", return_value=["a1"]
            ),
            mock.patch.object(
                self.scheduler,
                "_group_nodes_by_slice",
                return_value={"pool-a": ["a1"]},
            ),
            mock.patch.object(
                self.scheduler,
                "_provision_dynamic_slices",
                return_value=(["pool-a"], []),
            ),
        ):
            self.scheduler.apply_scheduler_resources()

        self.assertEqual(
            self._slices(), ["chs-slice-pool-a", "chs-slice-pool-b"]
        )
        self.assertEqual(self.kube.calls_to("delete_custom_objects"), [])

    @mock.patch("time.sleep")
    @mock.patch("time.time", side_effect=[0, 1, 310])
    def test_wait_for_slices_active_ignores_other_pools(
        self, mock_time, mock_sleep, mock_logger
    ):
        self.scheduler._run_id = "run-a"
        self.kube.add_object(
            "accelerator.gke.io",
            "slices",
            _chs_slice("other", "ACTIVE", run_id="run-b"),
        )
        self.kube.add_object(
            "accelerator.gke.io", "slices", _chs_slice("mypool", "CREATING")
        )

        success, failed = self.scheduler._wait_for_slices_active(["mypool"])

        self.assertEqual(success, [])
        self.assertEqual(failed, ["mypool"])
        mock_sleep.assert_called_once_with(10)


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

        self.assertEqual(detect_scheduler(), "standalone")

    @mock.patch("shutil.which")
    @mock.patch("os.path.exists")
    def test_detect_scheduler_fallback_to_standalone(
        self, mock_exists, mock_which
    ):
        """Test that standalone is detected if Slurm and K8s checks fail."""
        mock_which.return_value = False
        mock_exists.return_value = False

        with self.assertLogs("chs.schedulers", "INFO") as logs:
            self.assertEqual(detect_scheduler(), "standalone")
        self.assertIn(
            "No Slurm or Kubernetes environment detected", logs.output[0]
        )


class TestTpuSliceSummary(unittest.TestCase):
    def setUp(self):
        super().setUp()
        install_fake_client(self)

    def _scheduler(self, slices, **kwargs) -> KubernetesScheduler:
        scheduler = KubernetesScheduler(**kwargs)
        scheduler._tpu_slice_data = slices
        return scheduler

    def test_counts_partly_tested_slices_too(self):
        scheduler = self._scheduler(
            TpuSliceData(
                pools={"p1": ["n1", "n2"], "p2": ["n3", "n4"]},
                node_to_pool={"n1": "p1", "n2": "p1", "n3": "p2", "n4": "p2"},
                pool_to_topology={"p1": "2x2x2", "p2": "2x2x2"},
            )
        )

        summary = scheduler.get_tpu_slice_summary(["n1", "n2", "n3"])

        self.assertEqual(
            summary,
            TpuSliceSummary(
                slice_type="static",
                topologies={"2x2x2": 2},
                nodepool_count=2,
            ),
        )

    def test_any_dynamic_pool_makes_the_run_dynamic(self):
        scheduler = self._scheduler(
            TpuSliceData(
                pools={"dyn": ["n1", "n2"], "st": ["n3"]},
                node_to_pool={"n1": "dyn", "n2": "dyn", "n3": "st"},
                pool_to_topology={
                    "dyn": SUPPORTED_DYNAMIC_SLICE_TOPOLOGY,
                    "st": "2x2x1",
                },
                dynamic_slice_partitions={"dyn": "part-1"},
            ),
            nodepools="dyn,st",
        )

        summary = scheduler.get_tpu_slice_summary(["n1", "n2", "n3"])

        self.assertEqual(
            summary,
            TpuSliceSummary(
                slice_type="dynamic",
                topologies={"2x2x1": 1, SUPPORTED_DYNAMIC_SLICE_TOPOLOGY: 1},
                nodepool_count=2,
            ),
        )
        self.assertEqual(
            list(summary.topologies),
            ["2x2x1", SUPPORTED_DYNAMIC_SLICE_TOPOLOGY],
        )

    def test_no_nodepool_count_when_nodes_are_targeted(self):
        scheduler = self._scheduler(
            TpuSliceData(
                pools={"p1": ["n1", "n2"]},
                node_to_pool={"n1": "p1", "n2": "p1"},
                pool_to_topology={"p1": "2x2x2"},
            ),
            nodes="n1",
        )

        summary = scheduler.get_tpu_slice_summary(["n1"])

        self.assertEqual(summary.topologies, {"2x2x2": 1})
        self.assertIsNone(summary.nodepool_count)

    def test_each_node_of_a_single_host_pool_is_a_slice(self):
        scheduler = self._scheduler(
            TpuSliceData(
                pools={"sh": ["n1", "n2", "n3"], "mh": ["n4", "n5"]},
                node_to_pool={
                    "n1": "sh",
                    "n2": "sh",
                    "n3": "sh",
                    "n4": "mh",
                    "n5": "mh",
                },
                pool_to_topology={"sh": "2x2x1", "mh": "2x2x2"},
                single_host_pools={"sh"},
            ),
            nodes="n1,n2,n4",
        )

        summary = scheduler.get_tpu_slice_summary(["n1", "n2", "n4"])

        self.assertEqual(summary.topologies, {"2x2x1": 2, "2x2x2": 1})

    def test_dynamic_nodes_targeted_with_nodes_get_single_host_slices(self):
        scheduler = self._scheduler(
            TpuSliceData(provision_only_nodes={"d1", "d2", "d3"}),
            nodes="d1,d2",
        )
        scheduler._subslice_nodes = {"d1", "d2"}

        summary = scheduler.get_tpu_slice_summary(["d1", "d2"])

        self.assertEqual(
            summary,
            TpuSliceSummary(
                slice_type="dynamic",
                topologies={DYNAMIC_SINGLE_HOST_TOPOLOGY: 2},
                nodepool_count=None,
            ),
        )

    def test_none_without_tpu_nodes(self):
        self.assertIsNone(self._scheduler(None).get_tpu_slice_summary(["n1"]))
        self.assertIsNone(
            self._scheduler(TpuSliceData()).get_tpu_slice_summary(["gpu-1"])
        )
        self.assertIsNone(SlurmScheduler().get_tpu_slice_summary(["n1"]))


if __name__ == "__main__":
    unittest.main()
