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

"""Tests for the Kubernetes API call reductions in the Kubernetes scheduler."""

import concurrent.futures
import os
import tempfile
import threading
import unittest
from unittest import mock

from chs.constants import (
    CHS_NAMESPACE,
    DELETE_WAIT_TIMEOUT_SEC,
    NAMESPACE_DELETE_TIMEOUT_SEC,
)
from chs.exceptions import (
    ChsSchedulerException,
    ChsValidationError,
)
from chs.schedulers import kubernetes
from chs.schedulers.kubernetes import KubernetesScheduler, TpuSliceData
from chs.utils import k8s_client
from chs.utils.lifecycle import JobStatus
from tests.k8s_fakes import Call, install_fake_client

_INSTANCE_TYPE = "node.kubernetes.io/instance-type"
_BETA_INSTANCE_TYPE = "beta.kubernetes.io/instance-type"
_INSTANCE_ID = "container.googleapis.com/instance_id"
_CLUSTER_NAME = "container.googleapis.com/cluster_name"
_NODEPOOL = "cloud.google.com/gke-nodepool"
_JOBSET_GROUP = "jobset.x-k8s.io"
_JOBSET_VERSION = "v1alpha2"
_SLICE_GROUP = "accelerator.gke.io"
_SLICE_VERSION = "v1beta1"
_SLICE_CRD = "slices.accelerator.gke.io"
_SLICE_LIST = Call(
    "list_custom_objects",
    {
        "group": _SLICE_GROUP,
        "version": _SLICE_VERSION,
        "plural": "slices",
        "label_selector": "created-by=chs",
    },
)


def _pod(name, namespace=CHS_NAMESPACE):
    """Returns a minimal Kubernetes Pod object."""
    return {
        "metadata": {"name": name, "namespace": namespace},
        "spec": {"containers": [{"name": "test"}]},
    }


def _node(name, labels=None, annotations=None, provider_id=None):
    """Returns a minimal Kubernetes Node object."""
    node = {
        "kind": "Node",
        "metadata": {
            "name": name,
            "labels": dict(labels or {}),
            "annotations": dict(annotations or {}),
        },
    }
    if provider_id is not None:
        node["spec"] = {"providerID": provider_id}
    return node


def _node_pools_selector(*pools):
    """Returns the node label selector of `--nodepools`."""
    return f"{_NODEPOOL} in ({','.join(pools)})"


class _KubernetesTestBase(unittest.TestCase):
    """Fakes the Kubernetes API, mocks the scheduler logger and provides a
    temporary directory."""

    def setUp(self):
        super().setUp()
        self.kube = install_fake_client(self)
        self.mock_logger = self._patch(mock.patch.object(kubernetes, "logger"))
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_dir = tmp_dir.name

    def _patch(self, patcher):
        """Starts a patcher for the rest of the test and returns its mock."""
        started = patcher.start()
        self.addCleanup(patcher.stop)
        return started


class TestFetchJobLogConcurrency(_KubernetesTestBase):
    """Tests for fetching the pod logs of a JobSet concurrently."""

    def setUp(self):
        super().setUp()
        self.log_path = os.path.join(self.tmp_dir, "job.log")
        self.mock_stream = self._patch(
            mock.patch.object(k8s_client, "stream_pod_log")
        )
        self._reset_log_fetch_executor()

    def tearDown(self):
        self._reset_log_fetch_executor()
        super().tearDown()

    @staticmethod
    def _reset_log_fetch_executor():
        with kubernetes._LOG_FETCH_EXECUTOR_LOCK:
            executor = kubernetes._LOG_FETCH_EXECUTOR
            kubernetes._LOG_FETCH_EXECUTOR = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def _fetch(self, pod_names, job_id="chs-tpu-test"):
        """Runs fetch_job_log for a JobSet with the given pods."""
        scheduler = KubernetesScheduler()
        pod_objs = [_pod(name) for name in pod_names]
        with mock.patch.object(
            scheduler, "_get_pods", return_value=pod_objs
        ) as mock_get_pods:
            scheduler.fetch_job_log(job_id, self.log_path)
        return mock_get_pods

    def _read_log(self):
        with open(self.log_path, "rb") as f:
            return f.read()

    def _temp_files(self):
        return [
            name
            for name in os.listdir(self.tmp_dir)
            if name.startswith(kubernetes._LOG_TEMP_PREFIX)
        ]

    def test_logs_are_joined_in_pod_order(self):
        """Pods finishing in reverse order still yield pod-ordered bytes."""
        pods = [f"pod-{i}" for i in range(4)]
        logs = {pod: f"{pod}\n".encode() + b"\x00\xff\xfe\n" for pod in pods}
        finished = {pod: threading.Event() for pod in pods}
        finish_order = []

        def fake_stream(pod, out):
            name = pod["metadata"]["name"]
            index = pods.index(name)
            if index + 1 < len(pods):
                self.assertTrue(finished[pods[index + 1]].wait(timeout=10))
            out.write(logs[name])
            finish_order.append(name)
            finished[name].set()

        self.mock_stream.side_effect = fake_stream
        mock_get_pods = self._fetch(pods)

        mock_get_pods.assert_called_once_with(
            "chs-tpu-test", head_pod_only=False
        )
        self.assertEqual(finish_order, pods[::-1])
        self.assertEqual(self._read_log(), b"".join(logs[p] for p in pods))
        calls = self.mock_stream.call_args_list
        self.assertEqual(
            sorted(call.kwargs["pod"]["metadata"]["name"] for call in calls),
            sorted(pods),
        )
        for call in calls:
            self.assertEqual(call.args, ())
            self.assertEqual(set(call.kwargs), {"pod", "out"})
        self.assertEqual(
            len({call.kwargs["out"].name for call in calls}), len(pods)
        )
        self.assertEqual(os.listdir(self.tmp_dir), ["job.log"])

    def test_first_failure_in_pod_order_is_reported(self):
        """Pod order, not completion time, picks the reported failure."""
        pods = [f"pod-{i}" for i in range(4)]
        pod_3_failed = threading.Event()
        pod_1_error = k8s_client.KubeError("pod-1 log failed", status=500)
        pod_3_error = k8s_client.KubeError("pod-3 log failed", status=500)

        def fake_stream(pod, out):
            name = pod["metadata"]["name"]
            if name == "pod-3":
                pod_3_failed.set()
                raise pod_3_error
            if name == "pod-1":
                self.assertTrue(pod_3_failed.wait(timeout=10))
                raise pod_1_error
            out.write(name.encode())

        self.mock_stream.side_effect = fake_stream
        self._fetch(pods)

        self.assertEqual(self._read_log(), b"pod-0pod-2")
        self.mock_logger.error.assert_called_once()
        self.assertIs(self.mock_logger.error.call_args.args[-1], pod_1_error)
        self.assertEqual(self._temp_files(), [])

    def test_interrupted_fetch_propagates_and_cleans_up(self):
        """Ctrl+C is not logged as a failed fetch and leaves no temp files."""
        with self.subTest("multi-pod"):

            def fake_stream(pod, out):
                name = pod["metadata"]["name"]
                if name == "pod-1":
                    raise KeyboardInterrupt
                out.write(name.encode() + b"\n")

            self.mock_stream.side_effect = fake_stream
            with self.assertRaises(KeyboardInterrupt):
                self._fetch(["pod-0", "pod-1", "pod-2"], job_id="chs-tpu-job")

            self.mock_logger.error.assert_not_called()
            self.assertEqual(self._temp_files(), [])

        with self.subTest("single-pod"):
            self.mock_logger.reset_mock()
            self.mock_stream.side_effect = KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                self._fetch(["chs-nccl-0"], job_id="chs-nccl-job")

            self.mock_logger.error.assert_not_called()

    def test_gpu_job_fetches_head_pod_only(self):
        """Single-pod jobs stream directly to log_path without temp files."""

        def fake_stream(pod, out):
            out.write(b"head log\n")

        self.mock_stream.side_effect = fake_stream
        with (
            mock.patch.object(
                tempfile, "mkstemp", wraps=tempfile.mkstemp
            ) as mock_mkstemp,
            mock.patch.object(
                kubernetes,
                "_get_log_fetch_executor",
                wraps=kubernetes._get_log_fetch_executor,
            ) as mock_get_executor,
        ):
            mock_get_pods = self._fetch(
                ["chs-nccl-0-abc"], job_id="chs-nccl-test"
            )

        mock_get_pods.assert_called_once_with(
            "chs-nccl-test", head_pod_only=True
        )
        mock_mkstemp.assert_not_called()
        mock_get_executor.assert_not_called()
        self.assertIsNone(kubernetes._LOG_FETCH_EXECUTOR)
        self.mock_stream.assert_called_once()
        call = self.mock_stream.call_args
        self.assertEqual(call.args, ())
        self.assertEqual(set(call.kwargs), {"pod", "out"})
        self.assertEqual(call.kwargs["pod"], _pod("chs-nccl-0-abc"))
        self.assertEqual(call.kwargs["out"].name, self.log_path)
        self.assertEqual(self._read_log(), b"head log\n")
        self.assertEqual(os.listdir(self.tmp_dir), ["job.log"])

    def test_get_log_fetch_executor_returns_shared_instance(self):
        """_get_log_fetch_executor returns one shared pool across calls."""
        executor_1 = kubernetes._get_log_fetch_executor()
        executor_2 = kubernetes._get_log_fetch_executor()
        self.assertIs(executor_1, executor_2)
        self.assertEqual(
            executor_1._max_workers, kubernetes._LOG_FETCH_MAX_CONCURRENCY
        )

    def test_concurrent_fetches_share_one_process_wide_cap(self):
        """Concurrent fetch_job_log calls never exceed the 32-worker cap."""
        cap = kubernetes._LOG_FETCH_MAX_CONCURRENCY
        self.assertEqual(cap, 32)
        lock = threading.Lock()
        in_flight = 0
        peak = 0
        calls = 0
        saturated = threading.Event()

        def fake_stream(pod, out):
            nonlocal in_flight, peak, calls
            name = pod["metadata"]["name"]
            with lock:
                calls += 1
                in_flight += 1
                peak = max(peak, in_flight)
                if in_flight >= cap:
                    saturated.set()
            self.assertTrue(saturated.wait(timeout=5))
            out.write(name.encode() + b"\n")
            with lock:
                in_flight -= 1

        self.mock_stream.side_effect = fake_stream
        jobs = {
            f"chs-tpu-job-{j}": [f"job-{j}-pod-{i}" for i in range(20)]
            for j in range(3)
        }
        errors = []

        def fetch(job_id):
            scheduler = KubernetesScheduler()
            try:
                with mock.patch.object(
                    scheduler,
                    "_get_pods",
                    return_value=[_pod(p) for p in jobs[job_id]],
                ):
                    scheduler.fetch_job_log(
                        job_id, os.path.join(self.tmp_dir, f"{job_id}.log")
                    )
            except BaseException as e:
                errors.append(e)

        threads = [
            threading.Thread(target=fetch, args=(job_id,)) for job_id in jobs
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        self.assertEqual(errors, [])
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(peak, cap)
        self.assertEqual(calls, 3 * 20)
        for job_id, pods in jobs.items():
            with open(os.path.join(self.tmp_dir, f"{job_id}.log"), "rb") as f:
                self.assertEqual(
                    f.read(), b"".join(pod.encode() + b"\n" for pod in pods)
                )
        self.assertEqual(self._temp_files(), [])

    def test_no_pods_creates_no_file(self):
        """Without pods there is no API call and no log file."""
        self._fetch([])

        self.mock_stream.assert_not_called()
        self.assertEqual(os.listdir(self.tmp_dir), [])

    def test_os_errors_are_logged_and_clean_up_temp_files(self):
        """OSErrors and config errors during log fetch are logged."""
        with self.subTest("concatenation failure"):
            self.mock_stream.side_effect = lambda pod, out: out.write(b"log\n")
            with mock.patch.object(
                kubernetes.shutil,
                "copyfileobj",
                side_effect=OSError("disk full"),
            ):
                self._fetch(["pod-0", "pod-1"])
            self.mock_logger.error.assert_called_once()
            self.assertEqual(self._temp_files(), [])

        with self.subTest("missing kubeconfig"):
            self.mock_logger.reset_mock()
            self.mock_stream.side_effect = k8s_client.KubeError(
                "Kubernetes configuration could not be loaded: no kubeconfig"
            )
            self._fetch(["pod-0", "pod-1"], job_id="chs-tpu-job")
            self.mock_logger.error.assert_called_once()
            self.assertEqual(self._temp_files(), [])

    def test_failed_pod_does_not_stop_other_pods(self):
        """Queued pods are still fetched after pod 0 fails."""
        pods = [f"pod-{i}" for i in range(40)]
        pod_0_error = k8s_client.KubeError("stream of pod-0 broke")

        def fake_stream(pod, out):
            name = pod["metadata"]["name"]
            if name == "pod-0":
                raise pod_0_error
            out.write(name.encode() + b"\n")

        self.mock_stream.side_effect = fake_stream
        self._fetch(pods)

        self.assertEqual(self.mock_stream.call_count, len(pods))
        self.assertEqual(
            self._read_log(), b"".join(pod.encode() + b"\n" for pod in pods[1:])
        )
        self.mock_logger.error.assert_called_once()
        self.assertIs(self.mock_logger.error.call_args.args[-1], pod_0_error)
        self.assertEqual(self._temp_files(), [])

    def test_interrupt_cancels_unstarted_fetches(self):
        """Ctrl+C while waiting cancels queued pods 32..39 and writes no log."""
        cap = kubernetes._LOG_FETCH_MAX_CONCURRENCY
        self.assertEqual(cap, 32)
        pods = [f"pod-{i}" for i in range(40)]
        lock = threading.Lock()
        started = []
        pool_saturated = threading.Event()
        release_workers = threading.Event()
        wait_calls = []

        def interrupt_first_wait(fs):
            wait_calls.append(fs)
            if len(wait_calls) == 1:
                self.assertTrue(pool_saturated.wait(timeout=5))
                raise KeyboardInterrupt
            release_workers.set()
            return concurrent.futures.wait(fs)

        self._patch(
            mock.patch.object(
                kubernetes, "wait", side_effect=interrupt_first_wait
            )
        )

        def fake_stream(pod, out):
            with lock:
                started.append(pod["metadata"]["name"])
                if len(started) >= cap:
                    pool_saturated.set()
            self.assertTrue(release_workers.wait(timeout=5))

        self.mock_stream.side_effect = fake_stream
        with self.assertRaises(KeyboardInterrupt):
            self._fetch(pods)

        self.assertEqual(sorted(started), sorted(pods[:cap]))
        self.assertFalse(os.path.exists(self.log_path))
        self.assertEqual(self._temp_files(), [])

    def test_timeout_expired_is_logged_and_cleans_up_temp_files(self):
        """A pod log timeout is logged at error and removes temp files."""
        with self.subTest("multi-pod"):

            def fake_multi_stream(pod, out):
                name = pod["metadata"]["name"]
                namespace = pod["metadata"]["namespace"]
                if name == "pod-1":
                    raise k8s_client.KubeError(
                        f"GET /api/v1/namespaces/{namespace}/pods/{name}/log "
                        "timed out"
                    )
                out.write(name.encode() + b"\n")

            self.mock_stream.side_effect = fake_multi_stream
            self._fetch(["pod-0", "pod-1", "pod-2"], job_id="chs-tpu-timeout")

            self.assertEqual(self._read_log(), b"pod-0\npod-2\n")
            self.mock_logger.error.assert_called_once()
            self.assertIn(
                "chs-tpu-timeout", self.mock_logger.error.call_args.args
            )
            self.assertIsInstance(
                self.mock_logger.error.call_args.args[-1],
                k8s_client.KubeError,
            )
            self.assertEqual(self._temp_files(), [])

        with self.subTest("single-pod"):
            self.mock_logger.reset_mock()
            self.mock_stream.side_effect = k8s_client.KubeError(
                "GET pod log timed out"
            )
            self._fetch(["chs-nccl-0"], job_id="chs-nccl-timeout")

            self.mock_logger.error.assert_called_once()
            self.assertIn(
                "chs-nccl-timeout", self.mock_logger.error.call_args.args
            )
            self.assertIsInstance(
                self.mock_logger.error.call_args.args[-1],
                k8s_client.KubeError,
            )


class TestNodeObjectCache(_KubernetesTestBase):
    """Tests for serving node objects from the cached node list."""

    def setUp(self):
        super().setUp()
        self.scheduler = KubernetesScheduler()

    def test_cached_nodes_need_no_kubectl_and_preserve_order(self):
        """Nodes in the cached list are never fetched again and keep order."""
        node_0 = _node("node0", {_NODEPOOL: "pool-a"})
        node_1 = _node("node1", {_NODEPOOL: "pool-b"})
        node_2 = _node("node2", {_NODEPOOL: "pool-a"})
        self.scheduler._resolved_node_resources = [node_0, node_1, node_2]

        self.assertEqual(
            self.scheduler._get_node_objects(["node2", "node1", "node0"]),
            [node_2, node_1, node_0],
        )
        self.assertEqual(
            self.scheduler._group_nodes_by_slice(["node2", "node0", "node1"]),
            {"pool-a": ["node2", "node0"], "pool-b": ["node1"]},
        )
        with mock.patch.object(
            self.scheduler,
            "get_nodes_under_test",
            return_value=["node1", "node2"],
        ):
            self.assertEqual(
                self.scheduler.get_topology(),
                {"default-block": {"default-subblock": ["node1", "node2"]}},
            )
        self.assertEqual(self.kube.calls, [])

    def test_cold_cache_lists_nodes_once(self):
        """Without a cached list, the label-selected list is fetched once."""
        node_1 = self.kube.add_object("", "nodes", _node("node1"))
        node_2 = self.kube.add_object("", "nodes", _node("node2"))
        self.kube.add_object(
            "", "nodes", _node("sys-node", {_NODEPOOL: "system"})
        )

        self.assertEqual(
            self.scheduler._get_node_objects(["node1", "node2"]),
            [node_1, node_2],
        )
        self.assertEqual(
            self.kube.calls,
            [Call("list_nodes", {"label_selector": f"{_NODEPOOL}!=system"})],
        )
        self.assertEqual(
            self.scheduler._resolved_node_resources, [node_1, node_2]
        )

    def test_fetch_errors_keep_callers_error_handling(self):
        """Callers gracefully fall back when _fetch_raw_node_resources fails."""

        def assert_callers_handle_the_error():
            self.scheduler._resolved_node_resources = None
            with mock.patch.object(
                self.scheduler, "get_nodes_under_test", return_value=["n1"]
            ):
                self.assertEqual(self.scheduler.get_topology(), {})
            self.scheduler._resolved_node_resources = None
            self.assertEqual(
                self.scheduler._group_nodes_by_slice(["n1", "n2"]),
                {"failed-nodes": ["n1", "n2"]},
            )

        cases = [
            (
                "API error",
                k8s_client.KubeError(
                    "Error from server (InternalError): boom", status=500
                ),
            ),
            (
                "timeout",
                k8s_client.KubeError("GET /api/v1/nodes timed out"),
            ),
            (
                "no kube config",
                k8s_client.KubeError("no kubeconfig"),
            ),
        ]
        for label, error in cases:
            with self.subTest(label):
                self.kube.fail_next("list_nodes", error, error)
                assert_callers_handle_the_error()

    def test_project_id_comes_from_cached_provider_id(self):
        """A cached providerID spares the API call."""
        self.scheduler._resolved_node_resources = [
            _node("node1", provider_id="gce://proj-x/us-central1-a/node1")
        ]

        self.assertEqual(self.scheduler._resolve_project_id("node1"), "proj-x")
        self.assertEqual(self.kube.calls, [])

    def test_project_id_without_cached_provider_id(self):
        """Without cached providerID, only that node's providerID is queried."""
        self.kube.add_object(
            "",
            "nodes",
            _node("node1", provider_id="gce://proj-y/us-central1-a/node1"),
        )
        for cached in (None, [_node("node1")]):
            with self.subTest(cached=cached):
                self.scheduler._resolved_node_resources = cached
                self.kube.calls.clear()

                self.assertEqual(
                    self.scheduler._resolve_project_id("node1"), "proj-y"
                )
                self.assertEqual(
                    self.kube.calls,
                    [Call("get_node", {"name": "node1"})],
                )

    def test_invalid_cached_provider_id_is_rejected(self):
        """A cached providerID is validated like a fetched one."""
        self.scheduler._resolved_node_resources = [
            _node("node1", provider_id="gke://proj/us-central1-a/node1")
        ]

        with self.assertRaisesRegex(
            ChsSchedulerException, "Unexpected providerID format"
        ):
            self.scheduler._resolve_project_id("node1")
        self.assertEqual(self.kube.calls, [])

    def test_validation_error_from_list_propagates(self):
        """ChsValidationError from _fetch_raw_node_resources propagates."""
        scheduler = KubernetesScheduler(nodepools="pool-a,missing-pool")
        self.kube.add_object("", "nodes", _node("node1", {_NODEPOOL: "pool-a"}))

        with self.assertRaisesRegex(ChsValidationError, "missing-pool"):
            scheduler._get_node_objects(["node1"])
        self.assertEqual(
            self.kube.calls,
            [
                Call(
                    "list_nodes",
                    {
                        "label_selector": _node_pools_selector(
                            "pool-a", "missing-pool"
                        )
                    },
                )
            ],
        )
        self.assertIsNone(scheduler._resolved_node_resources)


class TestClusterNameMemoization(_KubernetesTestBase):
    """Tests for caching the cluster name on the scheduler instance."""

    def setUp(self):
        super().setUp()
        self.scheduler = KubernetesScheduler()
        self.mock_current_context = self._patch(
            mock.patch.object(k8s_client, "current_context", return_value=None)
        )

    def test_second_call_is_served_from_memory(self):
        """Only the first call looks the cluster name up."""
        self.mock_current_context.return_value = (
            "gke_my-proj_us-central1_my-cluster"
        )

        self.assertEqual(self.scheduler.get_cluster_name(), "my-cluster")
        self.assertEqual(self.scheduler.get_cluster_name(), "my-cluster")
        self.assertEqual(self.kube.calls, [])
        self.mock_current_context.assert_called_once_with()

    def test_failed_lookup_is_retried(self):
        """A None result is not cached."""
        self.assertIsNone(self.scheduler.get_cluster_name())
        self.assertEqual(self.mock_current_context.call_count, 1)
        self.assertIsNone(self.scheduler.get_cluster_name())
        self.assertEqual(self.mock_current_context.call_count, 2)
        self.assertEqual(self.kube.calls, [])

    def test_label_on_cached_nodes_skips_kubectl(self):
        """The cluster name label on a cached node spares any API call."""
        self.scheduler._resolved_node_resources = [
            _node("node1"),
            _node("node2", {_CLUSTER_NAME: "labeled-cluster"}),
        ]

        self.assertEqual(self.scheduler.get_cluster_name(), "labeled-cluster")
        self.assertEqual(self.kube.calls, [])
        self.mock_current_context.assert_not_called()

    def test_cached_nodes_without_label_fall_back_to_current_context(self):
        """Cached nodes without cluster label fall back to current_context."""
        self.scheduler._resolved_node_resources = [_node("node1")]
        self.mock_current_context.return_value = (
            "gke_my-proj_us-central1_ctx-cluster"
        )

        self.assertEqual(self.scheduler.get_cluster_name(), "ctx-cluster")
        self.assertEqual(self.kube.calls, [])
        self.mock_current_context.assert_called_once_with()


class TestDetectNodesHardwareFromNodeObjects(_KubernetesTestBase):
    """Tests for detecting node hardware from GKE node labels."""

    def setUp(self):
        super().setUp()
        self.scheduler = KubernetesScheduler()
        self.mock_query_gce = self._patch(
            mock.patch("chs.schedulers.base.query_gce_instances_info")
        )
        self._patch(mock.patch("chs.schedulers.base.logger"))

    def test_labeled_nodes_need_no_gce_lookup(self):
        """Labels and annotations answer everything without the GCE API."""
        self.scheduler._resolved_node_resources = [
            _node(
                "node1",
                {_INSTANCE_TYPE: "tpu7x-standard-4t"},
                {_INSTANCE_ID: "1111"},
            ),
            _node(
                "node2",
                {_INSTANCE_TYPE: "tpu7x-standard-4t"},
                {_INSTANCE_ID: "2222"},
            ),
        ]

        self.assertEqual(
            self.scheduler.detect_nodes_hardware(["node1", "node2"]),
            ["tpu7x-standard-4t"],
        )
        self.assertEqual(
            self.scheduler.node_to_instance_id,
            {"node1": "1111", "node2": "2222"},
        )
        self.mock_query_gce.assert_not_called()
        self.assertEqual(self.kube.calls, [])

    def test_incomplete_nodes_fall_back_to_gce(self):
        """Only nodes lacking the label or annotation go to the GCE API."""
        self.scheduler._resolved_node_resources = [
            _node(
                "node1", {_INSTANCE_TYPE: "a3-megagpu-8g"}, {_INSTANCE_ID: "1"}
            ),
            _node("node2", {_INSTANCE_TYPE: "a3-megagpu-8g"}),
            _node("node3", annotations={_INSTANCE_ID: "3"}),
            _node("node4", {_INSTANCE_TYPE: ""}, {_INSTANCE_ID: "4"}),
        ]
        self.mock_query_gce.return_value = {
            "node2": {"machine_type": "a3-megagpu-8g", "instance_id": "22"},
            "node3": {"machine_type": "a3-ultragpu-8g", "instance_id": "33"},
        }

        with mock.patch.object(
            self.scheduler, "_resolve_project_id", return_value="proj"
        ) as mock_resolve_project_id:
            machine_types = self.scheduler.detect_nodes_hardware(
                ["node1", "node2", "node3", "node4"]
            )

        mock_resolve_project_id.assert_called_once_with("node2")
        self.mock_query_gce.assert_called_once_with(
            "proj", ["node2", "node3", "node4"]
        )
        self.assertEqual(
            machine_types, ["a3-megagpu-8g", "a3-ultragpu-8g", "unknown"]
        )
        self.assertEqual(
            self.scheduler.node_to_instance_id,
            {"node1": "1", "node2": "22", "node3": "33"},
        )
        self.assertEqual(self.kube.calls, [])

    def test_machine_type_labels_are_normalized_and_beta_label_supported(self):
        """Label values are normalized and beta instance-type label works."""
        cases = [
            ({_INSTANCE_TYPE: "tpu7x-standard-4t"}, "tpu7x-standard-4t"),
            ({_INSTANCE_TYPE: "tpu7x-standard-4t-suffix"}, "tpu7x-standard-4t"),
            (
                {
                    _INSTANCE_TYPE: (
                        "zones/us-central1-a/machineTypes/a3-megagpu-8g"
                    )
                },
                "a3-megagpu-8g",
            ),
            ({_BETA_INSTANCE_TYPE: "a4-highgpu-8g"}, "a4-highgpu-8g"),
        ]
        for labels, expected in cases:
            with self.subTest(labels=labels):
                scheduler = KubernetesScheduler()
                scheduler._resolved_node_resources = [
                    _node("node1", labels, {_INSTANCE_ID: "1111"})
                ]
                self.assertEqual(
                    scheduler.detect_nodes_hardware(["node1"]), [expected]
                )
                self.assertEqual(
                    scheduler.node_to_instance_id, {"node1": "1111"}
                )
        self.mock_query_gce.assert_not_called()

    def test_label_only_node_resolves_from_label_when_gce_fails(self):
        """A label-only node uses its label when the GCE lookup raises."""
        self.scheduler._resolved_node_resources = [
            _node(
                "node1", {_INSTANCE_TYPE: "a3-megagpu-8g"}, {_INSTANCE_ID: "1"}
            ),
            _node("node2", {_INSTANCE_TYPE: "a5x-highgpu-4g-metal"}),
            _node("node3"),
        ]
        self.mock_query_gce.side_effect = RuntimeError("permission denied")

        with mock.patch.object(
            self.scheduler, "_resolve_project_id", return_value="proj"
        ):
            machine_types = self.scheduler.detect_nodes_hardware(
                ["node1", "node2", "node3"]
            )

        self.mock_query_gce.assert_called_once_with("proj", ["node2", "node3"])
        self.assertEqual(self.kube.calls, [])
        self.assertEqual(
            machine_types,
            ["a3-megagpu-8g", "a5x-highgpu-4g-metal", "unknown"],
        )
        self.assertEqual(
            self.scheduler.node_to_instance_id,
            {"node1": "1", "node2": "node2"},
        )

    def test_label_only_node_gets_label_type_and_gce_instance_id(self):
        """A label without annotation keeps its type and takes GCE's ID."""
        self.scheduler._resolved_node_resources = [
            _node("node1", {_INSTANCE_TYPE: "tpu7x-standard-4t-suffix"}),
        ]
        self.mock_query_gce.return_value = {
            "node1": {"machine_type": "tpu7x-standard-4t", "instance_id": "99"},
        }

        with mock.patch.object(
            self.scheduler, "_resolve_project_id", return_value="proj"
        ):
            machine_types = self.scheduler.detect_nodes_hardware(["node1"])

        self.mock_query_gce.assert_called_once_with("proj", ["node1"])
        self.assertEqual(machine_types, ["tpu7x-standard-4t"])
        self.assertEqual(self.scheduler.node_to_instance_id, {"node1": "99"})

    def test_unlabeled_node_unresolved_by_gce_is_unknown(self):
        """An unlabeled node that GCE does not return is reported unknown."""
        self.scheduler._resolved_node_resources = [
            _node(
                "node1", {_INSTANCE_TYPE: "a3-megagpu-8g"}, {_INSTANCE_ID: "1"}
            ),
            _node("node2", annotations={_INSTANCE_ID: "2"}),
        ]
        self.mock_query_gce.return_value = {}

        with mock.patch.object(
            self.scheduler, "_resolve_project_id", return_value="proj"
        ):
            machine_types = self.scheduler.detect_nodes_hardware(
                ["node1", "node2"]
            )

        self.mock_query_gce.assert_called_once_with("proj", ["node2"])
        self.assertEqual(machine_types, ["a3-megagpu-8g", "unknown"])
        self.assertEqual(self.scheduler.node_to_instance_id, {"node1": "1"})

    def test_unreadable_node_objects_fall_back_to_gce(self):
        """If node objects cannot be read, the GCE API handles all nodes."""
        self.mock_query_gce.return_value = {
            "node1": {"machine_type": "a3-megagpu-8g", "instance_id": "11"},
        }
        error = k8s_client.KubeError(
            "Error from server (InternalError): nodes unavailable", status=500
        )
        self.kube.fail_next("list_nodes", error)

        with mock.patch.object(
            self.scheduler, "_resolve_project_id", return_value="proj"
        ):
            machine_types = self.scheduler.detect_nodes_hardware(
                ["node1", "node2"]
            )

        self.assertEqual(
            [call.method for call in self.kube.calls], ["list_nodes"]
        )
        self.mock_query_gce.assert_called_once_with("proj", ["node1", "node2"])
        self.assertEqual(machine_types, ["a3-megagpu-8g", "unknown"])
        self.assertEqual(self.scheduler.node_to_instance_id, {"node1": "11"})

    def test_validation_error_propagates_without_gce_fallback(self):
        """ChsValidationError from _fetch_raw_node_resources propagates."""
        scheduler = KubernetesScheduler(nodepools="pool-a,missing-pool")
        self.kube.add_object("", "nodes", _node("node1", {_NODEPOOL: "pool-a"}))

        with self.assertRaisesRegex(ChsValidationError, "missing-pool"):
            scheduler.detect_nodes_hardware(["node1"])
        self.assertEqual(
            self.kube.calls,
            [
                Call(
                    "list_nodes",
                    {
                        "label_selector": _node_pools_selector(
                            "pool-a", "missing-pool"
                        )
                    },
                )
            ],
        )
        self.mock_query_gce.assert_not_called()
        self.assertIsNone(scheduler._resolved_node_resources)

    def test_no_nodes(self):
        """An empty node list needs no lookup at all."""
        self.assertEqual(self.scheduler.detect_nodes_hardware([]), [])
        self.assertEqual(self.kube.calls, [])
        self.mock_query_gce.assert_not_called()


class TestDynamicSliceCleanupDedup(_KubernetesTestBase):
    """Tests for skipping redundant dynamic slice cleanups."""

    def setUp(self):
        super().setUp()
        self.scheduler = KubernetesScheduler()

    def _slice_calls(self):
        """Returns the client calls that concern Slices."""
        return [
            call
            for call in self.kube.calls
            if (
                call.kwargs.get("group"),
                call.kwargs.get("version"),
                call.kwargs.get("plural"),
            )
            == (_SLICE_GROUP, _SLICE_VERSION, "slices")
        ]

    def test_back_to_back_cleanups_delete_slices_once(self):
        """cancel_jobs_by_prefix + cleanup deletes slices once statelessly."""
        self.kube.add_crd(_SLICE_CRD)
        for name, labels in (
            ("chs-slice", {"created-by": "chs"}),
            ("user-slice", {}),
        ):
            self.kube.add_object(
                _SLICE_GROUP,
                "slices",
                {"metadata": {"name": name, "labels": labels}},
            )
        self.kube.add_object(
            _JOBSET_GROUP,
            "jobsets",
            {"metadata": {"name": "chs-run1-nccl"}},
            namespace=CHS_NAMESPACE,
        )

        self.scheduler.cancel_jobs_by_prefix("chs-")
        self.scheduler.cleanup_scheduler_resources()

        self.assertEqual(
            self._slice_calls(),
            [
                _SLICE_LIST,
                Call(
                    "delete_custom_objects",
                    {
                        "group": _SLICE_GROUP,
                        "version": _SLICE_VERSION,
                        "plural": "slices",
                        "object_names": ["chs-slice"],
                        "wait_timeout_seconds": 300,
                    },
                ),
            ],
        )
        self.assertEqual(self.kube.stored_names(_JOBSET_GROUP, "jobsets"), [])
        self.assertEqual(
            self.kube.stored_names(_SLICE_GROUP, "slices"), ["user-slice"]
        )

    def test_cleanup_dynamic_slices_is_stateless(self):
        """_cleanup_dynamic_slices holds no flag and always queries cluster."""
        self.assertFalse(hasattr(self.scheduler, "_dynamic_slices_cleaned"))
        self.kube.add_crd(_SLICE_CRD)

        self.scheduler._cleanup_dynamic_slices()
        self.scheduler._cleanup_dynamic_slices()

        empty_delete = Call(
            "delete_custom_objects",
            {
                "group": _SLICE_GROUP,
                "version": _SLICE_VERSION,
                "plural": "slices",
                "object_names": [],
                "wait_timeout_seconds": 300,
            },
        )
        self.assertEqual(
            self._slice_calls(),
            [_SLICE_LIST, empty_delete, _SLICE_LIST, empty_delete],
        )

    def test_get_job_statuses_uses_namespaced_list_for_multiple_jobs(self):
        """get_job_statuses lists chs namespace once and filters."""
        completed = {"conditions": [{"type": "Completed", "status": "True"}]}
        for name, status in (
            ("chs-run-job1", completed),
            ("chs-run-job2", {"replicatedJobsStatus": [{"active": 1}]}),
            ("unrelated-jobset", completed),
        ):
            self.kube.add_object(
                _JOBSET_GROUP,
                "jobsets",
                {"metadata": {"name": name}, "status": status},
                namespace=CHS_NAMESPACE,
            )

        statuses = self.scheduler.get_job_statuses(
            ["chs-run-job1", "chs-run-job2"]
        )
        self.assertEqual(
            self.kube.calls,
            [
                Call(
                    "list_custom_objects",
                    {
                        "group": _JOBSET_GROUP,
                        "version": _JOBSET_VERSION,
                        "plural": "jobsets",
                        "namespace": CHS_NAMESPACE,
                    },
                )
            ],
        )
        self.assertNotIn("unrelated-jobset", statuses)
        self.assertEqual(
            statuses,
            {
                "chs-run-job1": JobStatus.COMPLETED.value,
                "chs-run-job2": JobStatus.RUNNING.value,
            },
        )

        self.kube.calls.clear()
        self.scheduler.get_job_statuses(["chs-run-job1"])
        self.assertEqual(
            self.kube.calls,
            [
                Call(
                    "list_custom_objects",
                    {
                        "group": _JOBSET_GROUP,
                        "version": _JOBSET_VERSION,
                        "plural": "jobsets",
                        "namespace": CHS_NAMESPACE,
                    },
                )
            ],
        )

    def test_apply_cleans_slices_only_for_dynamic_pools(self):
        """Slices are cleaned before provisioning only for dynamic pools."""
        slice_data = TpuSliceData(
            pool_to_topology={"static-pool": "2x2", "dyn-pool": "2x2"},
            dynamic_slice_partitions={"dyn-pool": "part-1"},
        )
        cases = [
            ("static pool", slice_data, {"static-pool": ["s1"]}, False),
            ("dynamic pool", slice_data, {"dyn-pool": ["d1"]}, True),
            ("GPU pool", None, {"default-slice": ["g1"]}, False),
        ]
        for label, tpu_slice_data, target_pools, cleans in cases:
            with self.subTest(label):
                scheduler = KubernetesScheduler()
                scheduler._tpu_slice_data = tpu_slice_data
                nodes = [n for pool in target_pools.values() for n in pool]
                with (
                    mock.patch.object(
                        scheduler, "get_nodes_under_test", return_value=nodes
                    ),
                    mock.patch.object(
                        scheduler,
                        "_group_nodes_by_slice",
                        return_value=target_pools,
                    ),
                    mock.patch.object(
                        scheduler, "_cleanup_dynamic_slices"
                    ) as mock_clean,
                    mock.patch.object(
                        scheduler,
                        "_provision_dynamic_slices",
                        return_value=([], []),
                    ) as mock_prov,
                    mock.patch.object(scheduler, "_ensure_namespace_exists"),
                ):
                    scheduler.apply_scheduler_resources()

                self.assertEqual(mock_clean.called, cleans)
                mock_prov.assert_called_once_with(target_pools)
        self.assertEqual(self.kube.calls, [])


class TestNamespaceSubsumesSecrets(_KubernetesTestBase):
    """Tests for letting namespace deletion delete the job secrets."""

    def setUp(self):
        super().setUp()
        self.scheduler = KubernetesScheduler()
        self.mock_cleanup_slices = self._patch(
            mock.patch.object(self.scheduler, "_cleanup_dynamic_slices")
        )
        self.mock_delete_secrets = self._patch(
            mock.patch.object(self.scheduler, "_delete_job_secrets")
        )

    def test_deleted_namespace_needs_no_secret_cleanup(self):
        """Secrets go with the namespace, so they are not deleted apart."""
        self.scheduler.cleanup_scheduler_resources()

        self.mock_cleanup_slices.assert_called_once_with()
        self.mock_delete_secrets.assert_not_called()
        self.assertEqual(
            self.kube.calls,
            [
                Call(
                    "delete_namespace",
                    {
                        "name": CHS_NAMESPACE,
                        "wait_timeout_seconds": NAMESPACE_DELETE_TIMEOUT_SEC,
                    },
                )
            ],
        )

    def test_failed_namespace_deletion_falls_back_to_secrets(self):
        """If the namespace survives, its secrets are deleted directly."""
        self.kube.fail_next(
            "delete_namespace",
            k8s_client.KubeError(
                "Error from server (Forbidden): forbidden", status=403
            ),
        )

        self.scheduler.cleanup_scheduler_resources()

        self.mock_delete_secrets.assert_called_once_with()
        self.mock_logger.warning.assert_called_once()

    def test_cancel_jobs_still_deletes_secrets(self):
        """cancel_jobs, also used on its own, keeps deleting secrets."""
        self.scheduler.cancel_jobs(["chs-job1"])

        self.assertEqual(
            self.kube.calls,
            [
                Call(
                    "delete_custom_objects",
                    {
                        "group": _JOBSET_GROUP,
                        "version": _JOBSET_VERSION,
                        "plural": "jobsets",
                        "object_names": ["chs-job1"],
                        "namespace": CHS_NAMESPACE,
                        "wait_timeout_seconds": DELETE_WAIT_TIMEOUT_SEC,
                    },
                )
            ],
        )
        self.mock_delete_secrets.assert_called_once_with(["chs-job1"])


if __name__ == "__main__":
    unittest.main()
