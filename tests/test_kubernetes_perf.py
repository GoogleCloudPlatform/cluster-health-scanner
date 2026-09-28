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

"""Tests for the kubectl call reductions in the Kubernetes scheduler.

Every kubectl call is mocked: none of these tests reaches a cluster.
"""

import concurrent.futures
import json
import os
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

from chs.constants import CHS_NAMESPACE, NAMESPACE_DELETE_TIMEOUT_SEC
from chs.exceptions import (
    ChsPreflightException,
    ChsSchedulerException,
    ChsValidationError,
)
from chs.preflights.kubernetes_preflight import KubernetesPreflightChecker
from chs.schedulers import kubernetes
from chs.schedulers.kubernetes import KubernetesScheduler, TpuSliceData

_INSTANCE_TYPE = "node.kubernetes.io/instance-type"
_BETA_INSTANCE_TYPE = "beta.kubernetes.io/instance-type"
_INSTANCE_ID = "container.googleapis.com/instance_id"
_CLUSTER_NAME = "container.googleapis.com/cluster_name"
_NODEPOOL = "cloud.google.com/gke-nodepool"
_CRD_CHECK = ["get", "crd", "slices.accelerator.gke.io", "--ignore-not-found"]
_SLICE_DELETE = [
    "delete",
    "slices.accelerator.gke.io",
    "-A",
    "-l",
    "created-by=chs",
    "--ignore-not-found",
]


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


def _node_list(*nodes):
    """Returns the kubectl JSON output for a list of nodes."""
    return json.dumps({"kind": "List", "items": list(nodes)})


class _KubernetesTestBase(unittest.TestCase):
    """Mocks the scheduler logger and provides a temporary directory."""

    def setUp(self):
        super().setUp()
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
        self.mock_run = self._patch(mock.patch.object(subprocess, "run"))
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
        with mock.patch.object(
            scheduler, "_get_pod_names", return_value=pod_names
        ) as mock_get_pod_names:
            scheduler.fetch_job_log(job_id, self.log_path)
        return mock_get_pod_names

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

        def fake_run(cmd, stdout, **kwargs):
            pod = cmd[2]
            index = pods.index(pod)
            # Each pod waits for the next one, so they finish in reverse.
            if index + 1 < len(pods):
                self.assertTrue(finished[pods[index + 1]].wait(timeout=10))
            stdout.write(logs[pod])
            finish_order.append(pod)
            finished[pod].set()

        self.mock_run.side_effect = fake_run
        mock_get_pod_names = self._fetch(pods)

        mock_get_pod_names.assert_called_once_with(
            "chs-tpu-test", head_pod_only=False
        )
        self.assertEqual(finish_order, pods[::-1])
        self.assertEqual(self._read_log(), b"".join(logs[p] for p in pods))
        self.assertEqual(
            sorted(call.args[0] for call in self.mock_run.call_args_list),
            [["kubectl", "logs", pod, "-n", CHS_NAMESPACE] for pod in pods],
        )
        for call in self.mock_run.call_args_list:
            self.assertEqual(set(call.kwargs), {"stdout", "check", "timeout"})
            self.assertTrue(call.kwargs["check"])
            self.assertEqual(call.kwargs["timeout"], 120)
        self.assertEqual(os.listdir(self.tmp_dir), ["job.log"])

    def test_first_failure_in_pod_order_is_reported(self):
        """Pod order, not completion time, picks the reported failure."""
        pods = [f"pod-{i}" for i in range(4)]
        pod_3_failed = threading.Event()

        def fake_run(cmd, stdout, **kwargs):
            pod = cmd[2]
            if pod == "pod-3":
                pod_3_failed.set()
                raise subprocess.CalledProcessError(3, cmd)
            if pod == "pod-1":
                self.assertTrue(pod_3_failed.wait(timeout=10))
                raise subprocess.CalledProcessError(1, cmd)
            stdout.write(pod.encode())

        self.mock_run.side_effect = fake_run
        self._fetch(pods)

        self.assertEqual(self._read_log(), b"pod-0pod-2")
        self.mock_logger.error.assert_called_once()
        error = self.mock_logger.error.call_args.args[-1]
        self.assertIn("pod-1", error)
        self.assertIn("exit status 1", error)
        self.assertEqual(self._temp_files(), [])

    def test_interrupted_fetch_is_logged_at_debug(self):
        """SIGINT-style exit codes keep today's debug-only message."""
        for returncode in (130, -2):
            with self.subTest(returncode=returncode):
                self.mock_logger.reset_mock()

                def fake_run(cmd, stdout, code=returncode, **kwargs):
                    raise subprocess.CalledProcessError(code, cmd)

                self.mock_run.side_effect = fake_run
                self._fetch(["pod-0", "pod-1"], job_id="chs-tpu-job")

                self.mock_logger.error.assert_not_called()
                self.mock_logger.debug.assert_any_call(
                    "Log fetching interrupted by user for JobSet %s.",
                    "chs-tpu-job",
                )

    def test_gpu_job_fetches_head_pod_only(self):
        """Single-pod jobs stream directly to log_path without temp files."""

        def fake_run(cmd, stdout, **kwargs):
            stdout.write(b"head log\n")

        self.mock_run.side_effect = fake_run
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
            mock_get_pod_names = self._fetch(
                ["chs-nccl-0-abc"], job_id="chs-nccl-test"
            )

        mock_get_pod_names.assert_called_once_with(
            "chs-nccl-test", head_pod_only=True
        )
        mock_mkstemp.assert_not_called()
        mock_get_executor.assert_not_called()
        self.assertIsNone(kubernetes._LOG_FETCH_EXECUTOR)
        self.mock_run.assert_called_once()
        self.assertEqual(
            self.mock_run.call_args.args[0],
            ["kubectl", "logs", "chs-nccl-0-abc", "-n", CHS_NAMESPACE],
        )
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

        def fake_run(cmd, stdout, **kwargs):
            nonlocal in_flight, peak, calls
            with lock:
                calls += 1
                in_flight += 1
                peak = max(peak, in_flight)
                if in_flight >= cap:
                    saturated.set()
            self.assertTrue(saturated.wait(timeout=5))
            stdout.write(cmd[2].encode() + b"\n")
            with lock:
                in_flight -= 1

        self.mock_run.side_effect = fake_run
        jobs = {
            f"chs-tpu-job-{j}": [f"job-{j}-pod-{i}" for i in range(20)]
            for j in range(3)
        }
        errors = []

        def fetch(job_id):
            scheduler = KubernetesScheduler()
            try:
                with mock.patch.object(
                    scheduler, "_get_pod_names", return_value=jobs[job_id]
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
        """Without pods there is no kubectl call and no log file."""
        self._fetch([])

        self.mock_run.assert_not_called()
        self.assertEqual(os.listdir(self.tmp_dir), [])

    def test_os_errors_are_logged_and_clean_up_temp_files(self):
        """OSErrors during temp file creation, kubectl, or copy are logged."""
        with self.subTest("concatenation failure"):
            self.mock_run.side_effect = lambda cmd, stdout, **kw: stdout.write(
                b"log\n"
            )
            with mock.patch.object(
                kubernetes.shutil,
                "copyfileobj",
                side_effect=OSError("disk full"),
            ):
                self._fetch(["pod-0", "pod-1"])
            self.mock_logger.error.assert_called_once()
            self.assertEqual(self._temp_files(), [])

        with self.subTest("missing kubectl"):
            self.mock_logger.reset_mock()
            self.mock_run.side_effect = FileNotFoundError(
                2, "No such file or directory", "kubectl"
            )
            self._fetch(["pod-0", "pod-1"])
            self.mock_logger.error.assert_called_once()
            self.assertEqual(self._temp_files(), [])

    def test_failed_pod_does_not_stop_other_pods(self):
        """Queued pods are still fetched after pod 0 fails."""
        pods = [f"pod-{i}" for i in range(40)]

        def fake_run(cmd, stdout, **kwargs):
            if cmd[2] == "pod-0":
                raise subprocess.CalledProcessError(1, cmd)
            stdout.write(cmd[2].encode() + b"\n")

        self.mock_run.side_effect = fake_run
        self._fetch(pods)

        self.assertEqual(self.mock_run.call_count, len(pods))
        self.assertEqual(
            self._read_log(), b"".join(pod.encode() + b"\n" for pod in pods[1:])
        )
        self.mock_logger.error.assert_called_once()
        self.assertIn("pod-0", self.mock_logger.error.call_args.args[-1])
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

        def fake_run(cmd, stdout, **kwargs):
            with lock:
                started.append(cmd[2])
                if len(started) >= cap:
                    pool_saturated.set()
            self.assertTrue(release_workers.wait(timeout=5))

        self.mock_run.side_effect = fake_run
        with self.assertRaises(KeyboardInterrupt):
            self._fetch(pods)

        self.assertEqual(sorted(started), sorted(pods[:cap]))
        self.assertFalse(os.path.exists(self.log_path))
        self.assertEqual(self._temp_files(), [])

    def test_timeout_expired_is_logged_and_cleans_up_temp_files(self):
        """A kubectl logs timeout is logged at error and removes temp files."""
        with self.subTest("multi-pod"):

            def fake_multi_run(cmd, stdout, **kwargs):
                if cmd[2] == "pod-1":
                    raise subprocess.TimeoutExpired(cmd, 120)
                stdout.write(cmd[2].encode() + b"\n")

            self.mock_run.side_effect = fake_multi_run
            self._fetch(["pod-0", "pod-1", "pod-2"], job_id="chs-tpu-timeout")

            self.assertEqual(self._read_log(), b"pod-0\npod-2\n")
            self.mock_logger.error.assert_called_once()
            self.assertIn(
                "chs-tpu-timeout", self.mock_logger.error.call_args.args
            )
            self.assertIsInstance(
                self.mock_logger.error.call_args.args[-1],
                subprocess.TimeoutExpired,
            )
            self.assertEqual(self._temp_files(), [])

        with self.subTest("single-pod"):
            self.mock_logger.reset_mock()
            self.mock_run.side_effect = subprocess.TimeoutExpired(
                ["kubectl", "logs"], 120
            )
            self._fetch(["chs-nccl-0"], job_id="chs-nccl-timeout")

            self.mock_logger.error.assert_called_once()
            self.assertIn(
                "chs-nccl-timeout", self.mock_logger.error.call_args.args
            )
            self.assertIsInstance(
                self.mock_logger.error.call_args.args[-1],
                subprocess.TimeoutExpired,
            )


class TestNodeObjectCache(_KubernetesTestBase):
    """Tests for serving node objects from the cached node list."""

    def setUp(self):
        super().setUp()
        self.scheduler = KubernetesScheduler()
        self.mock_kubectl = self._patch(
            mock.patch.object(KubernetesScheduler, "_run_kubectl")
        )

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
        self.mock_kubectl.assert_not_called()

    def test_cold_cache_lists_nodes_once(self):
        """Without a cached list, the label-selected list is fetched once."""
        node_1, node_2 = _node("node1"), _node("node2")
        self.mock_kubectl.return_value = _node_list(node_1, node_2)

        self.assertEqual(
            self.scheduler._get_node_objects(["node1", "node2"]),
            [node_1, node_2],
        )
        self.mock_kubectl.assert_called_once_with(
            [
                "get",
                "nodes",
                "-l",
                "cloud.google.com/gke-nodepool!=system",
                "-o",
                "json",
            ]
        )
        self.assertEqual(
            self.scheduler._resolved_node_resources, [node_1, node_2]
        )

    def test_fetch_errors_keep_callers_error_handling(self):
        """Callers gracefully fall back when _fetch_raw_node_resources fails."""
        cases = [
            ("kubectl error", subprocess.CalledProcessError(1, "kubectl")),
            ("invalid JSON", "not json"),
            ("no kubectl", FileNotFoundError("kubectl")),
        ]
        for label, outcome in cases:
            with self.subTest(label):
                self.scheduler._resolved_node_resources = None
                self.mock_kubectl.reset_mock(
                    return_value=True, side_effect=True
                )
                if isinstance(outcome, Exception):
                    self.mock_kubectl.side_effect = outcome
                else:
                    self.mock_kubectl.return_value = outcome

                with mock.patch.object(
                    self.scheduler, "get_nodes_under_test", return_value=["n1"]
                ):
                    self.assertEqual(self.scheduler.get_topology(), {})
                self.assertEqual(
                    self.scheduler._group_nodes_by_slice(["n1", "n2"]),
                    {"failed-nodes": ["n1", "n2"]},
                )

    def test_project_id_comes_from_cached_provider_id(self):
        """A cached providerID spares the kubectl call."""
        self.scheduler._resolved_node_resources = [
            _node("node1", provider_id="gce://proj-x/us-central1-a/node1")
        ]

        self.assertEqual(self.scheduler._resolve_project_id("node1"), "proj-x")
        self.mock_kubectl.assert_not_called()

    def test_project_id_without_cached_provider_id(self):
        """Without cached providerID, only that node's providerID is queried."""
        for cached in (None, [_node("node1")]):
            with self.subTest(cached=cached):
                self.scheduler._resolved_node_resources = cached
                self.mock_kubectl.reset_mock()
                self.mock_kubectl.return_value = (
                    "gce://proj-y/us-central1-a/node1\n"
                )

                self.assertEqual(
                    self.scheduler._resolve_project_id("node1"), "proj-y"
                )
                self.mock_kubectl.assert_called_once_with(
                    [
                        "get",
                        "node",
                        "node1",
                        "-o",
                        "jsonpath={.spec.providerID}",
                    ]
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
        self.mock_kubectl.assert_not_called()

    def test_validation_error_from_list_propagates(self):
        """ChsValidationError from _fetch_raw_node_resources propagates."""
        scheduler = KubernetesScheduler(nodepools="pool-a,missing-pool")
        self.mock_kubectl.return_value = _node_list(
            _node("node1", {_NODEPOOL: "pool-a"})
        )

        with self.assertRaisesRegex(ChsValidationError, "missing-pool"):
            scheduler._get_node_objects(["node1"])
        self.mock_kubectl.assert_called_once()
        self.assertIsNone(scheduler._resolved_node_resources)


class TestClusterNameMemoization(_KubernetesTestBase):
    """Tests for caching the cluster name on the scheduler instance."""

    def setUp(self):
        super().setUp()
        self.scheduler = KubernetesScheduler()
        self.mock_kubectl = self._patch(
            mock.patch.object(KubernetesScheduler, "_run_kubectl")
        )

    def test_second_call_is_served_from_memory(self):
        """Only the first call looks the cluster name up."""
        self.mock_kubectl.return_value = "gke_my-proj_us-central1_my-cluster"

        self.assertEqual(self.scheduler.get_cluster_name(), "my-cluster")
        self.assertEqual(self.scheduler.get_cluster_name(), "my-cluster")
        self.mock_kubectl.assert_called_once_with(
            ["config", "current-context"], timeout=5
        )

    def test_failed_lookup_is_retried(self):
        """A None result is not cached."""
        self.mock_kubectl.side_effect = subprocess.CalledProcessError(
            1, "kubectl"
        )

        self.assertIsNone(self.scheduler.get_cluster_name())
        self.assertEqual(self.mock_kubectl.call_count, 1)
        self.assertIsNone(self.scheduler.get_cluster_name())
        self.assertEqual(self.mock_kubectl.call_count, 2)

    def test_label_on_cached_nodes_skips_kubectl(self):
        """The cluster name label on a cached node spares any kubectl call."""
        self.scheduler._resolved_node_resources = [
            _node("node1"),
            _node("node2", {_CLUSTER_NAME: "labeled-cluster"}),
        ]

        self.assertEqual(self.scheduler.get_cluster_name(), "labeled-cluster")
        self.mock_kubectl.assert_not_called()

    def test_cached_nodes_without_label_fall_back_to_current_context(self):
        """Cached nodes without cluster label fall back to current-context."""
        self.scheduler._resolved_node_resources = [_node("node1")]
        self.mock_kubectl.return_value = "gke_my-proj_us-central1_ctx-cluster"

        self.assertEqual(self.scheduler.get_cluster_name(), "ctx-cluster")
        self.mock_kubectl.assert_called_once_with(
            ["config", "current-context"], timeout=5
        )


class TestDetectNodesHardwareFromNodeObjects(_KubernetesTestBase):
    """Tests for detecting node hardware from GKE node labels."""

    def setUp(self):
        super().setUp()
        self.scheduler = KubernetesScheduler()
        self.mock_kubectl = self._patch(
            mock.patch.object(KubernetesScheduler, "_run_kubectl")
        )
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
        self.mock_kubectl.assert_not_called()

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
        self.mock_kubectl.assert_not_called()
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

        with (
            mock.patch.object(
                self.scheduler,
                "_get_node_objects",
                side_effect=ChsSchedulerException("kubectl failed"),
            ),
            mock.patch.object(
                self.scheduler, "_resolve_project_id", return_value="proj"
            ),
        ):
            machine_types = self.scheduler.detect_nodes_hardware(
                ["node1", "node2"]
            )

        self.mock_query_gce.assert_called_once_with("proj", ["node1", "node2"])
        self.assertEqual(machine_types, ["a3-megagpu-8g", "unknown"])
        self.assertEqual(self.scheduler.node_to_instance_id, {"node1": "11"})

    def test_validation_error_propagates_without_gce_fallback(self):
        """ChsValidationError from _fetch_raw_node_resources propagates."""
        scheduler = KubernetesScheduler(nodepools="pool-a,missing-pool")
        self.mock_kubectl.return_value = _node_list(
            _node("node1", {_NODEPOOL: "pool-a"})
        )

        with self.assertRaisesRegex(ChsValidationError, "missing-pool"):
            scheduler.detect_nodes_hardware(["node1"])
        self.mock_kubectl.assert_called_once()
        self.mock_query_gce.assert_not_called()
        self.assertIsNone(scheduler._resolved_node_resources)

    def test_no_nodes(self):
        """An empty node list needs no lookup at all."""
        self.assertEqual(self.scheduler.detect_nodes_hardware([]), [])
        self.mock_kubectl.assert_not_called()
        self.mock_query_gce.assert_not_called()


class TestDynamicSliceCleanupDedup(_KubernetesTestBase):
    """Tests for skipping redundant dynamic slice cleanups."""

    def setUp(self):
        super().setUp()
        self.scheduler = KubernetesScheduler()
        self.mock_kubectl = self._patch(
            mock.patch.object(
                KubernetesScheduler, "_run_kubectl", return_value=""
            )
        )

    def _slice_calls(self):
        return [
            call.args[0]
            for call in self.mock_kubectl.call_args_list
            if "slices.accelerator.gke.io" in call.args[0]
        ]

    def test_back_to_back_cleanups_delete_slices_once(self):
        """cancel_jobs_by_prefix + cleanup deletes slices once statelessly."""

        def fake_kubectl(args, **kwargs):
            if args == _CRD_CHECK:
                return "slices.accelerator.gke.io"
            return ""

        self.mock_kubectl.side_effect = fake_kubectl

        self.scheduler.cancel_jobs_by_prefix("chs-")
        self.scheduler.cleanup_scheduler_resources()

        self.assertEqual(self._slice_calls(), [_CRD_CHECK, _SLICE_DELETE])

    def test_cleanup_dynamic_slices_is_stateless(self):
        """_cleanup_dynamic_slices holds no flag and always queries cluster."""
        self.assertFalse(hasattr(self.scheduler, "_dynamic_slices_cleaned"))
        self.mock_kubectl.side_effect = [
            "slices.accelerator.gke.io",
            "",
            "slices.accelerator.gke.io",
            "",
        ]

        self.scheduler._cleanup_dynamic_slices()
        self.scheduler._cleanup_dynamic_slices()

        self.assertEqual(
            self._slice_calls(),
            [_CRD_CHECK, _SLICE_DELETE, _CRD_CHECK, _SLICE_DELETE],
        )

    def test_get_job_statuses_uses_namespaced_list_for_multiple_jobs(self):
        """get_job_statuses lists chs namespace once for >1 job and filters."""
        self.mock_kubectl.return_value = json.dumps(
            {
                "kind": "List",
                "items": [
                    {
                        "metadata": {"name": "chs-run-job1"},
                        "status": {
                            "conditions": [
                                {"type": "Completed", "status": "True"}
                            ]
                        },
                    },
                    {
                        "metadata": {"name": "chs-run-job2"},
                        "status": {"replicatedJobsStatus": [{"active": 1}]},
                    },
                    {
                        "metadata": {"name": "unrelated-jobset"},
                        "status": {
                            "conditions": [
                                {"type": "Completed", "status": "True"}
                            ]
                        },
                    },
                ],
            }
        )

        statuses = self.scheduler.get_job_statuses(
            ["chs-run-job1", "chs-run-job2"]
        )
        self.mock_kubectl.assert_called_once_with(
            ["get", "jobset", "-o", "json", "--ignore-not-found"],
            namespaced=True,
        )
        self.assertNotIn("unrelated-jobset", statuses)
        self.assertEqual(set(statuses.keys()), {"chs-run-job1", "chs-run-job2"})

        self.mock_kubectl.reset_mock()
        self.scheduler.get_job_statuses(["chs-run-job1"])
        self.mock_kubectl.assert_called_once_with(
            [
                "get",
                "jobset",
                "chs-run-job1",
                "-o",
                "json",
                "--ignore-not-found",
            ],
            namespaced=True,
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


class TestNamespaceSubsumesSecrets(_KubernetesTestBase):
    """Tests for letting namespace deletion delete the job secrets."""

    def setUp(self):
        super().setUp()
        self.scheduler = KubernetesScheduler()
        self.mock_kubectl = self._patch(
            mock.patch.object(
                KubernetesScheduler, "_run_kubectl", return_value=""
            )
        )
        self.mock_cleanup_slices = self._patch(
            mock.patch.object(self.scheduler, "_cleanup_dynamic_slices")
        )
        self.mock_delete_secrets = self._patch(
            mock.patch.object(self.scheduler, "_delete_job_secrets")
        )
        self.mock_kueue = self._patch(
            mock.patch.object(kubernetes, "KueueManager")
        )

    def test_deleted_namespace_needs_no_secret_cleanup(self):
        """Secrets go with the namespace, so they are not deleted apart."""
        self.scheduler.cleanup_scheduler_resources()

        self.mock_cleanup_slices.assert_called_once_with()
        self.mock_kueue.return_value.cleanup.assert_called_once_with()
        self.mock_delete_secrets.assert_not_called()
        self.mock_kubectl.assert_called_once_with(
            ["delete", "namespace", CHS_NAMESPACE, "--ignore-not-found"],
            timeout=NAMESPACE_DELETE_TIMEOUT_SEC,
        )

    def test_failed_namespace_deletion_falls_back_to_secrets(self):
        """If the namespace survives, its secrets are deleted directly."""
        self.mock_kubectl.side_effect = subprocess.CalledProcessError(
            1, "kubectl delete namespace", stderr="forbidden"
        )

        self.scheduler.cleanup_scheduler_resources()

        self.mock_delete_secrets.assert_called_once_with()
        self.mock_logger.warning.assert_called_once()

    def test_cancel_jobs_still_deletes_secrets(self):
        """cancel_jobs, also used on its own, keeps deleting secrets."""
        self.scheduler.cancel_jobs(["chs-job1"])

        self.mock_delete_secrets.assert_called_once_with()


class TestKueuePreflightEmptyDeployments(unittest.TestCase):
    """Tests _check_kueue when no Kueue controller deployment is found."""

    def test_check_kueue_returns_false_when_no_deployments_found(self):
        """_check_kueue logs a warning and returns False on empty list."""
        checker = KubernetesPreflightChecker(mock.Mock())
        with (
            mock.patch.object(
                checker, "_discover_deployments", return_value=[]
            ),
            mock.patch(
                "chs.preflights.kubernetes_preflight.logger"
            ) as mock_logger,
        ):
            result = checker._check_kueue(
                ["customresourcedefinition.apiextensions.k8s.io/kueues.kueue"]
            )

        self.assertFalse(result)
        mock_logger.warning.assert_called_once()
        self.assertIn(
            "Kueue CRD is installed, but no Kueue controller deployment "
            "could be found.",
            mock_logger.warning.call_args.args[0],
        )

    def test_check_raises_when_kueue_crd_has_no_deployment(self):
        """check() raises ChsPreflightException when Kueue has no deployment."""
        checker = KubernetesPreflightChecker(mock.Mock())
        with (
            mock.patch.object(
                checker,
                "_get_installed_crds",
                return_value=[
                    "jobsets.jobset.sigs.k8s.io",
                    "kueues.kueue.sigs.k8s.io",
                ],
            ),
            mock.patch.object(checker, "_check_jobset", return_value=True),
            mock.patch.object(
                checker, "_discover_deployments", return_value=[]
            ),
            mock.patch("chs.preflights.kubernetes_preflight.logger"),
            self.assertRaisesRegex(
                ChsPreflightException, "Kueue is missing or incompatible"
            ),
        ):
            checker.check([{"name": "nccl-cluster"}])


if __name__ == "__main__":
    unittest.main()
