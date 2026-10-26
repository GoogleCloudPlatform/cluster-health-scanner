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

"""Unit tests for chs.utils.k8s_client."""

import inspect
import io
import json
import os
from types import SimpleNamespace
from typing import Any
import unittest
from unittest import mock

import kubernetes
from kubernetes.client.exceptions import ApiException
from kubernetes.config.config_exception import ConfigException
import urllib3

from chs.utils import k8s_client
from chs.utils.k8s_client import KubeError
from tests import k8s_fakes


class _FakeClock:
    """Deterministic monotonic clock and sleep stub."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _FakeResponse:
    """In-memory urllib3 HTTPResponse stand-in."""

    def __init__(
        self,
        payload: Any = None,
        *,
        chunks: list[bytes] | None = None,
        stream_error: BaseException | None = None,
        data_error: BaseException | None = None,
    ) -> None:
        self._data = (
            payload
            if isinstance(payload, bytes)
            else json.dumps(payload if payload is not None else {}).encode()
        )
        self._chunks = chunks if chunks is not None else [self._data]
        self._stream_error = stream_error
        self._data_error = data_error
        self.released = False
        self.closed = False

    @property
    def data(self) -> bytes:
        if self._data_error is not None:
            raise self._data_error
        return self._data

    def stream(self):
        for chunk in self._chunks:
            yield chunk
        if self._stream_error is not None:
            raise self._stream_error

    def release_conn(self) -> None:
        self.released = True

    def close(self) -> None:
        self.closed = True


def _api_error(
    status: int,
    reason: str = "Error",
    message: str | None = None,
    *,
    raw_body: Any = None,
) -> ApiException:
    err = ApiException(status=status, reason=reason)
    if raw_body is not None:
        err.body = raw_body
    elif message is not None:
        err.body = json.dumps({"message": message})
    return err


class SdkTest(unittest.TestCase):
    """Tests for k8s_client._sdk configuration loading and caching."""

    def setUp(self) -> None:
        super().setUp()
        allowed = k8s_fakes.real_client_allowed()
        allowed.__enter__()
        self.addCleanup(allowed.__exit__, None, None, None)

    def test_has_kubeconfig(self) -> None:
        with self.subTest("default ~/.kube/config"):
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(
                    os.path,
                    "exists",
                    side_effect=lambda p: p
                    == os.path.expanduser("~/.kube/config"),
                ),
            ):
                self.assertTrue(k8s_client.has_kubeconfig())
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(os.path, "exists", return_value=False),
            ):
                self.assertFalse(k8s_client.has_kubeconfig())

        with self.subTest("KUBECONFIG env paths"):
            env = {"KUBECONFIG": f"/missing{os.pathsep} /present "}
            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch.object(
                    os.path, "exists", side_effect=lambda p: p == "/present"
                ),
            ):
                self.assertTrue(k8s_client.has_kubeconfig())
            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch.object(os.path, "exists", return_value=False),
            ):
                self.assertFalse(k8s_client.has_kubeconfig())

    def test_loads_kubeconfig_with_persist_false_and_context_namespace(
        self,
    ) -> None:
        with (
            mock.patch.object(k8s_client, "has_kubeconfig", return_value=True),
            mock.patch.object(
                kubernetes.config, "load_kube_config"
            ) as load_kube_config,
            mock.patch.object(
                kubernetes.config,
                "list_kube_config_contexts",
                return_value=(
                    [],
                    {"name": "ctx-a", "context": {"namespace": "team-ns"}},
                ),
            ) as list_contexts,
            mock.patch.object(k8s_client.atexit, "register") as register_atexit,
        ):
            sdk = k8s_client._sdk()
            self.assertIs(k8s_client._sdk(), sdk)

        self.addCleanup(k8s_client._close_api_client, sdk.core.api_client)
        load_kube_config.assert_called_once()
        self.assertIs(
            load_kube_config.call_args.kwargs["persist_config"], False
        )
        list_contexts.assert_called_once_with()
        register_atexit.assert_called_once_with(
            k8s_client._close_api_client, sdk.core.api_client
        )
        config = load_kube_config.call_args.kwargs["client_configuration"]
        self.assertEqual(config.connection_pool_maxsize, 64)
        self.assertEqual(
            sdk.core.api_client.default_headers.get("Accept-Encoding"), "gzip"
        )
        self.assertEqual(sdk.context, "ctx-a")
        self.assertEqual(sdk.namespace, "team-ns")
        self.assertIsInstance(sdk.core, kubernetes.client.CoreV1Api)
        self.assertIsInstance(sdk.batch, kubernetes.client.BatchV1Api)
        self.assertIsInstance(sdk.custom, kubernetes.client.CustomObjectsApi)
        self.assertIs(sdk.api_error, ApiException)
        self.assertIs(sdk.transport_error, urllib3.exceptions.HTTPError)

    def test_kubeconfig_without_namespace_defaults_to_default(self) -> None:
        with (
            mock.patch.object(k8s_client, "has_kubeconfig", return_value=True),
            mock.patch.object(kubernetes.config, "load_kube_config"),
            mock.patch.object(
                kubernetes.config,
                "list_kube_config_contexts",
                return_value=([], {"name": "ctx-a"}),
            ),
            mock.patch.object(k8s_client.atexit, "register"),
        ):
            sdk = k8s_client._sdk()

        self.addCleanup(k8s_client._close_api_client, sdk.core.api_client)
        self.assertEqual(sdk.context, "ctx-a")
        self.assertEqual(sdk.namespace, "default")

    def test_close_api_client_clears_pool_and_closes(self) -> None:
        pool_manager = mock.Mock()
        api_client = SimpleNamespace(
            rest_client=SimpleNamespace(pool_manager=pool_manager),
            close=mock.Mock(),
        )
        k8s_client._close_api_client(api_client)
        pool_manager.clear.assert_called_once_with()
        api_client.close.assert_called_once_with()

        minimal_client = SimpleNamespace(rest_client=None, close=mock.Mock())
        k8s_client._close_api_client(minimal_client)
        minimal_client.close.assert_called_once_with()

    def test_falls_back_to_incluster_config_with_and_without_ns_file(
        self,
    ) -> None:
        with (
            mock.patch.object(k8s_client, "has_kubeconfig", return_value=False),
            mock.patch.dict(
                os.environ, {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}, clear=False
            ),
            mock.patch.object(
                kubernetes.config, "load_kube_config"
            ) as load_kube_config,
            mock.patch.object(
                kubernetes.config, "load_incluster_config"
            ) as load_incluster,
            mock.patch.object(
                k8s_client,
                "open",
                mock.mock_open(read_data="incluster-ns\n"),
                create=True,
            ),
            mock.patch.object(k8s_client.atexit, "register"),
        ):
            sdk = k8s_client._sdk()

        self.addCleanup(k8s_client._close_api_client, sdk.core.api_client)
        load_kube_config.assert_not_called()
        load_incluster.assert_called_once()
        self.assertIsNone(sdk.context)
        self.assertEqual(sdk.namespace, "incluster-ns")

        k8s_fakes.REAL_SDK.cache_clear()
        with (
            mock.patch.object(k8s_client, "has_kubeconfig", return_value=False),
            mock.patch.dict(
                os.environ, {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}, clear=False
            ),
            mock.patch.object(kubernetes.config, "load_incluster_config"),
            mock.patch.object(
                k8s_client,
                "open",
                side_effect=OSError("missing namespace file"),
                create=True,
            ),
            mock.patch.object(k8s_client.atexit, "register"),
        ):
            sdk_default = k8s_client._sdk()

        self.addCleanup(
            k8s_client._close_api_client, sdk_default.core.api_client
        )
        self.assertIsNone(sdk_default.context)
        self.assertEqual(sdk_default.namespace, "default")

    def test_raises_kube_error_when_config_cannot_be_loaded(self) -> None:
        with (
            mock.patch.object(k8s_client, "has_kubeconfig", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            with self.assertRaisesRegex(
                KubeError,
                "No kubeconfig found and not running inside a Kubernetes "
                "cluster",
            ):
                k8s_client._sdk()

        k8s_fakes.REAL_SDK.cache_clear()
        with (
            mock.patch.object(k8s_client, "has_kubeconfig", return_value=True),
            mock.patch.dict(
                os.environ, {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}, clear=True
            ),
            mock.patch.object(
                kubernetes.config,
                "load_kube_config",
                side_effect=ConfigException("broken kubeconfig"),
            ),
            mock.patch.object(
                kubernetes.config, "load_incluster_config"
            ) as load_incluster,
        ):
            with self.assertRaisesRegex(
                KubeError,
                "Kubernetes configuration could not be loaded: "
                "broken kubeconfig",
            ):
                k8s_client._sdk()
            load_incluster.assert_not_called()

        k8s_fakes.REAL_SDK.cache_clear()
        with (
            mock.patch.object(k8s_client, "has_kubeconfig", return_value=False),
            mock.patch.dict(
                os.environ, {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}, clear=True
            ),
            mock.patch.object(
                kubernetes.config,
                "load_incluster_config",
                side_effect=ConfigException("missing token"),
            ),
        ):
            with self.assertRaisesRegex(
                KubeError,
                "Kubernetes configuration could not be loaded: missing token",
            ):
                k8s_client._sdk()

    def test_proxy_parity_with_kubectl(self) -> None:
        def load_config_for(
            env: dict[str, str],
        ) -> kubernetes.client.Configuration:
            k8s_fakes.REAL_SDK.cache_clear()

            def set_host(
                client_configuration: kubernetes.client.Configuration,
                **_kwargs: Any,
            ) -> None:
                client_configuration.host = "https://api.example.com:6443"

            with (
                mock.patch.object(
                    k8s_client, "has_kubeconfig", return_value=True
                ),
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch.object(
                    kubernetes.config,
                    "load_kube_config",
                    side_effect=set_host,
                ) as load_kube_config,
                mock.patch.object(
                    kubernetes.config,
                    "list_kube_config_contexts",
                    return_value=([], {"name": "ctx"}),
                ),
                mock.patch.object(k8s_client.atexit, "register"),
            ):
                sdk = k8s_client._sdk()
            self.addCleanup(k8s_client._close_api_client, sdk.core.api_client)
            return load_kube_config.call_args.kwargs["client_configuration"]

        with self.subTest("HTTPS_PROXY used for https host"):
            cfg = load_config_for(
                {"HTTPS_PROXY": "http://https-proxy.internal:8443"}
            )
            self.assertEqual(cfg.proxy, "http://https-proxy.internal:8443")
            self.assertIsNone(cfg.no_proxy)

        with self.subTest("HTTP_PROXY alone not used for https host"):
            cfg = load_config_for(
                {"HTTP_PROXY": "http://http-proxy.internal:8080"}
            )
            self.assertIsNone(cfg.proxy)
            self.assertIsNone(cfg.no_proxy)

        with self.subTest("NO_PROXY containing the host disables proxy"):
            cfg = load_config_for(
                {
                    "HTTPS_PROXY": "http://https-proxy.internal:8443",
                    "NO_PROXY": "localhost,api.example.com",
                }
            )
            self.assertIsNone(cfg.proxy)
            self.assertIsNone(cfg.no_proxy)


class ClientFunctionsTest(unittest.TestCase):
    """Tests for k8s_client module-level functions and helpers."""

    def setUp(self) -> None:
        super().setUp()
        self.sdk = SimpleNamespace(
            context="gke_proj_zone_cluster",
            namespace="ctx-default",
            core=mock.Mock(),
            batch=mock.Mock(),
            custom=mock.Mock(),
            api_error=ApiException,
            transport_error=urllib3.exceptions.HTTPError,
        )
        sdk_patcher = mock.patch.object(
            k8s_client, "_sdk", return_value=self.sdk
        )
        sdk_patcher.start()
        self.addCleanup(sdk_patcher.stop)

        self.clock = _FakeClock()
        time_patcher = mock.patch.object(k8s_client, "time", self.clock)
        time_patcher.start()
        self.addCleanup(time_patcher.stop)

    def test_request_error_translation_and_timeout_override(self) -> None:
        with self.subTest("ApiException with JSON message"):
            method = mock.Mock(
                side_effect=_api_error(404, "Not Found", 'nodes "n1" not found')
            )
            with self.assertRaises(KubeError) as ctx:
                k8s_client._request(method, name="n1")
            self.assertEqual(str(ctx.exception), 'nodes "n1" not found')
            self.assertEqual(ctx.exception.status, 404)

        with self.subTest("ApiException with non-JSON body"):
            for raw_body in (b"<html>Bad Gateway</html>", None, "{}"):
                method = mock.Mock(
                    side_effect=_api_error(
                        502, "Bad Gateway", raw_body=raw_body
                    )
                )
                with self.assertRaises(KubeError) as ctx:
                    k8s_client._request(method)
                self.assertEqual(str(ctx.exception), "502 Bad Gateway")
                self.assertEqual(ctx.exception.status, 502)

        with self.subTest("urllib3 HTTPError"):
            method = mock.Mock(
                side_effect=urllib3.exceptions.HTTPError("connection reset")
            )
            with self.assertRaises(KubeError) as ctx:
                k8s_client._request(method)
            self.assertEqual(
                str(ctx.exception),
                "Kubernetes API request failed: connection reset",
            )
            self.assertIsNone(ctx.exception.status)

        with self.subTest("urllib3 HTTPError while reading response.data"):
            resp = _FakeResponse(
                data_error=urllib3.exceptions.HTTPError("read timed out")
            )
            method = mock.Mock(return_value=resp)
            with self.assertRaises(KubeError) as ctx:
                k8s_client._request(method)
            self.assertEqual(
                str(ctx.exception),
                "Kubernetes API request failed: read timed out",
            )
            self.assertIsNone(ctx.exception.status)
            self.assertTrue(resp.closed)
            self.assertTrue(resp.released)

        with self.subTest("_request_timeout override"):
            resp = _FakeResponse({"metadata": {"name": "n1"}})
            self.sdk.core.read_node.return_value = resp
            k8s_client.get_node(name="n1", _request_timeout=15)
            self.assertTrue(resp.closed)
            self.assertTrue(resp.released)
            self.sdk.core.read_node.assert_called_once_with(
                name="n1",
                _preload_content=False,
                _request_timeout=15,
            )

        with self.subTest("invalid JSON in _get"):
            self.sdk.core.read_node.return_value = _FakeResponse(
                b"<html>502 Bad Gateway</html>"
            )
            with self.assertRaisesRegex(
                KubeError, "Invalid JSON in Kubernetes API response:"
            ):
                k8s_client.get_node(name="n1")

        with self.subTest("non-dict and null items in _list"):
            self.sdk.core.list_pod_for_all_namespaces.return_value = (
                _FakeResponse(b"[]")
            )
            with self.assertRaisesRegex(
                KubeError,
                "Invalid Kubernetes API list response: expected dict, got list",
            ):
                k8s_client.list_pods()

            self.sdk.core.list_pod_for_all_namespaces.return_value = (
                _FakeResponse({"items": None})
            )
            self.assertEqual(k8s_client.list_pods(), [])

            self.sdk.core.list_pod_for_all_namespaces.return_value = (
                _FakeResponse({})
            )
            self.assertEqual(k8s_client.list_pods(), [])

    def test_context_and_nodes(self) -> None:
        self.assertEqual(k8s_client.current_context(), "gke_proj_zone_cluster")

        self.sdk.core.read_node.return_value = _FakeResponse(
            {"metadata": {"name": "n1"}}
        )
        self.assertEqual(
            k8s_client.get_node(name="n1", resource_version="42"),
            {"metadata": {"name": "n1"}},
        )
        self.sdk.core.read_node.assert_called_once_with(
            name="n1",
            resource_version="42",
            _preload_content=False,
            _request_timeout=120,
        )

        self.sdk.core.list_node.return_value = _FakeResponse(
            {"items": [{"metadata": {"name": "n1"}}]}
        )
        self.assertEqual(
            k8s_client.list_nodes(label_selector="pool=p1", limit=10),
            [{"metadata": {"name": "n1"}}],
        )
        self.sdk.core.list_node.assert_called_once_with(
            label_selector="pool=p1",
            limit=10,
            _preload_content=False,
            _request_timeout=120,
        )

    def test_list_pods_namespaced_and_all_namespaces(self) -> None:
        self.sdk.core.list_namespaced_pod.return_value = _FakeResponse(
            {"items": [{"metadata": {"name": "p1"}}]}
        )
        self.assertEqual(
            k8s_client.list_pods(
                namespace="chs",
                label_selector="app=x",
                field_selector="a=b",
                limit=5,
            ),
            [{"metadata": {"name": "p1"}}],
        )
        self.sdk.core.list_namespaced_pod.assert_called_once_with(
            namespace="chs",
            label_selector="app=x",
            field_selector="a=b",
            limit=5,
            _preload_content=False,
            _request_timeout=120,
        )

        self.sdk.core.list_pod_for_all_namespaces.return_value = _FakeResponse(
            {"items": [{"metadata": {"name": "p2"}}]}
        )
        self.assertEqual(
            k8s_client.list_pods(field_selector="spec.nodeName=n1"),
            [{"metadata": {"name": "p2"}}],
        )
        self.sdk.core.list_pod_for_all_namespaces.assert_called_once_with(
            field_selector="spec.nodeName=n1",
            _preload_content=False,
            _request_timeout=120,
        )

    def test_stream_pod_log(self) -> None:
        with self.subTest("uses default-container annotation when present"):
            resp = _FakeResponse(chunks=[b"line1\n", b"line2\n"])
            self.sdk.core.read_namespaced_pod_log.return_value = resp
            pod = {
                "metadata": {
                    "name": "p1",
                    "namespace": "chs",
                    "annotations": {
                        "kubectl.kubernetes.io/default-container": "worker"
                    },
                },
                "spec": {
                    "containers": [{"name": "sidecar"}, {"name": "worker"}]
                },
            }
            out = io.BytesIO()
            k8s_client.stream_pod_log(pod=pod, out=out, tail_lines=50)
            self.assertEqual(out.getvalue(), b"line1\nline2\n")
            self.assertTrue(resp.released)
            self.assertTrue(resp.closed)
            self.sdk.core.read_namespaced_pod_log.assert_called_once_with(
                name="p1",
                namespace="chs",
                container="worker",
                tail_lines=50,
                _preload_content=False,
                _request_timeout=120,
            )

        with self.subTest("explicit container and default namespace"):
            self.sdk.core.read_namespaced_pod_log.reset_mock()
            resp = _FakeResponse(chunks=[b"ok\n"])
            self.sdk.core.read_namespaced_pod_log.return_value = resp
            pod = {
                "metadata": {"name": "p1"},
                "spec": {"containers": [{"name": "main"}]},
            }
            out = io.BytesIO()
            k8s_client.stream_pod_log(pod=pod, out=out, container="sidecar")
            self.assertEqual(out.getvalue(), b"ok\n")
            self.sdk.core.read_namespaced_pod_log.assert_called_once_with(
                name="p1",
                namespace="ctx-default",
                container="sidecar",
                _preload_content=False,
                _request_timeout=120,
            )

        with self.subTest(
            "falls back to first container and releases on error"
        ):
            self.sdk.core.read_namespaced_pod_log.reset_mock()
            resp = _FakeResponse(
                chunks=[b"partial\n"],
                stream_error=urllib3.exceptions.HTTPError("stream broke"),
            )
            self.sdk.core.read_namespaced_pod_log.return_value = resp
            pod = {
                "metadata": {
                    "name": "p2",
                    "namespace": "chs",
                    "annotations": None,
                },
                "spec": {"containers": [{"name": "main"}]},
            }
            out = io.BytesIO()
            with mock.patch.object(k8s_client.time, "sleep") as mock_sleep:
                with self.assertRaisesRegex(
                    KubeError, "Failed to read log of pod p2: stream broke"
                ):
                    k8s_client.stream_pod_log(pod=pod, out=out)
                self.assertEqual(mock_sleep.call_count, 4)
                mock_sleep.assert_has_calls(
                    [
                        mock.call(1.0),
                        mock.call(2.0),
                        mock.call(4.0),
                        mock.call(8.0),
                    ]
                )
            self.assertTrue(resp.released)
            self.assertTrue(resp.closed)
            self.assertEqual(
                self.sdk.core.read_namespaced_pod_log.call_count, 5
            )

        with self.subTest("retries on transient error and succeeds"):
            self.sdk.core.read_namespaced_pod_log.reset_mock()
            fail_resp = _FakeResponse(
                chunks=[b"bad\n"],
                stream_error=urllib3.exceptions.HTTPError("RST_STREAM"),
            )
            ok_resp = _FakeResponse(chunks=[b"good log\n"])
            self.sdk.core.read_namespaced_pod_log.side_effect = [
                fail_resp,
                ok_resp,
            ]
            pod = {
                "metadata": {
                    "name": "p3",
                    "namespace": "chs",
                },
                "spec": {"containers": [{"name": "main"}]},
            }
            out = io.BytesIO()
            with mock.patch.object(k8s_client.time, "sleep") as mock_sleep:
                k8s_client.stream_pod_log(pod=pod, out=out)
                mock_sleep.assert_called_once_with(1.0)
            self.assertEqual(out.getvalue(), b"good log\n")
            self.assertEqual(
                self.sdk.core.read_namespaced_pod_log.call_count, 2
            )

    def test_create_and_delete_namespace(self) -> None:
        create_resp = _FakeResponse({})
        self.sdk.core.create_namespace.return_value = create_resp
        k8s_client.create_namespace_if_not_exists(name="chs", dry_run="All")
        self.assertTrue(create_resp.released)
        self.assertTrue(create_resp.closed)
        self.sdk.core.create_namespace.assert_called_once_with(
            body={"metadata": {"name": "chs"}},
            dry_run="All",
            _preload_content=False,
            _request_timeout=120,
        )

        self.sdk.core.create_namespace.side_effect = _api_error(
            409, "AlreadyExists", 'namespaces "chs" already exists'
        )
        k8s_client.create_namespace_if_not_exists(name="chs")

        self.sdk.core.create_namespace.side_effect = _api_error(
            403, "Forbidden", "forbidden"
        )
        with self.assertRaises(KubeError) as ctx:
            k8s_client.create_namespace_if_not_exists(name="chs")
        self.assertEqual(ctx.exception.status, 403)

        delete_resp = _FakeResponse({})
        read_resp = _FakeResponse({"metadata": {"name": "chs"}})
        self.sdk.core.delete_namespace.return_value = delete_resp
        self.sdk.core.read_namespace.side_effect = [
            read_resp,
            _api_error(404, "NotFound", 'namespaces "chs" not found'),
        ]
        k8s_client.delete_namespace(
            name="chs",
            wait_timeout_seconds=5,
            propagation_policy="Foreground",
        )
        self.assertTrue(delete_resp.released)
        self.assertTrue(delete_resp.closed)
        self.assertTrue(read_resp.released)
        self.assertTrue(read_resp.closed)
        self.sdk.core.delete_namespace.assert_called_once_with(
            name="chs",
            propagation_policy="Foreground",
            _preload_content=False,
            _request_timeout=120,
        )
        self.sdk.core.read_namespace.assert_called_with(
            name="chs",
            _preload_content=False,
            _request_timeout=120,
        )
        self.assertEqual(self.clock.sleeps, [0.5])

    def test_delete_and_wait_edge_cases(self) -> None:
        with self.subTest("empty names returns immediately"):
            delete = mock.Mock()
            read = mock.Mock()
            k8s_client._delete_and_wait(delete, read, [], 1.0)
            delete.assert_not_called()
            read.assert_not_called()
            self.assertEqual(self.clock.sleeps, [])

        with self.subTest("computes deadline before deletes"):

            def slow_delete(**_kwargs: Any) -> Any:
                self.clock.now += 2.0
                return _FakeResponse({})

            delete = mock.Mock(side_effect=slow_delete)
            read = mock.Mock(return_value=_FakeResponse({}))
            with self.assertRaisesRegex(
                KubeError, "Timed out waiting for deletion of a, b"
            ):
                k8s_client._delete_and_wait(delete, read, ["a", "b"], 1.0)
            # Because the deadline was computed before slow_delete advanced the
            # clock by 4s (> 1.0s), the first poll immediately times out
            # without sleeping.
            self.assertEqual(self.clock.sleeps, [])

        with self.subTest("does not wait for objects that returned 404"):
            delete = mock.Mock(
                side_effect=[
                    _api_error(404, "NotFound", "not found"),
                    _FakeResponse({}),
                ]
            )
            read = mock.Mock(
                side_effect=_api_error(404, "NotFound", "not found")
            )
            k8s_client._delete_and_wait(delete, read, ["a", "b"], 1.0)
            self.assertEqual(delete.call_count, 2)
            read.assert_called_once_with(
                name="b", _preload_content=False, _request_timeout=120
            )

        with self.subTest("all deletes returned 404 skips polling"):
            delete = mock.Mock(
                side_effect=_api_error(404, "NotFound", "not found")
            )
            read = mock.Mock()
            k8s_client._delete_and_wait(delete, read, ["a", "b"], 1.0)
            read.assert_not_called()
            self.assertEqual(self.clock.sleeps, [])

        with self.subTest("polls until timeout when still present"):
            self.clock.sleeps.clear()
            delete = mock.Mock(return_value=_FakeResponse({}))
            read = mock.Mock(return_value=_FakeResponse({}))
            with self.assertRaisesRegex(
                KubeError, "Timed out waiting for deletion of a"
            ):
                k8s_client._delete_and_wait(delete, read, ["a"], 1.0)
            self.assertEqual(self.clock.sleeps, [0.5, 0.5, 0.5])

        with self.subTest("raises non-404 error on delete"):
            delete = mock.Mock(
                side_effect=_api_error(403, "Forbidden", "denied")
            )
            read = mock.Mock()
            with self.assertRaises(KubeError) as ctx:
                k8s_client._delete_and_wait(delete, read, ["a"], 1.0)
            self.assertEqual(ctx.exception.status, 403)
            read.assert_not_called()

        with self.subTest("raises non-404 error on read poll"):
            delete = mock.Mock(return_value=_FakeResponse({}))
            read = mock.Mock(
                side_effect=_api_error(500, "InternalError", "boom")
            )
            with self.assertRaises(KubeError) as ctx:
                k8s_client._delete_and_wait(delete, read, ["a"], 1.0)
            self.assertEqual(ctx.exception.status, 500)

    def test_secrets(self) -> None:
        self.sdk.core.list_namespaced_secret.return_value = _FakeResponse(
            {"items": [{"metadata": {"name": "s1"}}]}
        )
        self.assertEqual(
            k8s_client.list_secrets(namespace="chs", label_selector="k=v"),
            [{"metadata": {"name": "s1"}}],
        )
        self.sdk.core.list_namespaced_secret.assert_called_once_with(
            namespace="chs",
            label_selector="k=v",
            _preload_content=False,
            _request_timeout=120,
        )

        self.sdk.core.list_secret_for_all_namespaces.return_value = (
            _FakeResponse({"items": [{"metadata": {"name": "s2"}}]})
        )
        self.assertEqual(
            k8s_client.list_secrets(field_selector="type=Opaque"),
            [{"metadata": {"name": "s2"}}],
        )
        self.sdk.core.list_secret_for_all_namespaces.assert_called_once_with(
            field_selector="type=Opaque",
            _preload_content=False,
            _request_timeout=120,
        )

        self.sdk.core.delete_namespaced_secret.return_value = _FakeResponse({})
        k8s_client.delete_secret(
            name="s1", namespace="chs", grace_period_seconds=0
        )
        self.sdk.core.delete_namespaced_secret.assert_called_once_with(
            name="s1",
            namespace="chs",
            grace_period_seconds=0,
            _preload_content=False,
            _request_timeout=120,
        )

        self.sdk.core.delete_namespaced_secret.reset_mock()
        k8s_client.delete_secret(name="s1")
        self.sdk.core.delete_namespaced_secret.assert_called_once_with(
            name="s1",
            namespace="ctx-default",
            _preload_content=False,
            _request_timeout=120,
        )

        self.sdk.core.delete_namespaced_secret.side_effect = _api_error(
            404, "NotFound", "not found"
        )
        k8s_client.delete_secret(name="s1", namespace="chs")

        self.sdk.core.delete_namespaced_secret.side_effect = _api_error(
            403, "Forbidden", "forbidden"
        )
        with self.assertRaises(KubeError) as ctx:
            k8s_client.delete_secret(name="s1", namespace="chs")
        self.assertEqual(ctx.exception.status, 403)

    def test_delete_service_ignores_404_only(self) -> None:
        self.sdk.core.delete_namespaced_service.return_value = _FakeResponse({})
        k8s_client.delete_service(name="chs-run1", namespace="chs")
        self.sdk.core.delete_namespaced_service.assert_called_once_with(
            name="chs-run1",
            namespace="chs",
            _preload_content=False,
            _request_timeout=120,
        )

        self.sdk.core.delete_namespaced_service.side_effect = _api_error(
            404, "NotFound", "not found"
        )
        k8s_client.delete_service(name="chs-run1", namespace="chs")

        self.sdk.core.delete_namespaced_service.side_effect = _api_error(
            403, "Forbidden", "forbidden"
        )
        with self.assertRaises(KubeError) as ctx:
            k8s_client.delete_service(name="chs-run1", namespace="chs")
        self.assertEqual(ctx.exception.status, 403)

    def test_jobs(self) -> None:
        self.sdk.batch.list_namespaced_job.return_value = _FakeResponse(
            {"items": [{"metadata": {"name": "j1"}}]}
        )
        self.assertEqual(
            k8s_client.list_jobs(namespace="chs", label_selector="jobset=js1"),
            [{"metadata": {"name": "j1"}}],
        )
        self.sdk.batch.list_namespaced_job.assert_called_once_with(
            namespace="chs",
            label_selector="jobset=js1",
            _preload_content=False,
            _request_timeout=120,
        )

        self.sdk.batch.list_job_for_all_namespaces.return_value = _FakeResponse(
            {"items": [{"metadata": {"name": "j2"}}]}
        )
        self.assertEqual(
            k8s_client.list_jobs(label_selector="jobset=js1"),
            [{"metadata": {"name": "j2"}}],
        )
        self.sdk.batch.list_job_for_all_namespaces.assert_called_once_with(
            label_selector="jobset=js1",
            _preload_content=False,
            _request_timeout=120,
        )

    def test_list_custom_objects(self) -> None:
        self.sdk.custom.list_namespaced_custom_object.return_value = (
            _FakeResponse({"items": [{"metadata": {"name": "cd1"}}]})
        )
        self.assertEqual(
            k8s_client.list_custom_objects(
                group="resource.nvidia.com",
                version="v1beta1",
                plural="computedomains",
                namespace="chs",
                label_selector="k=v",
            ),
            [{"metadata": {"name": "cd1"}}],
        )
        self.sdk.custom.list_namespaced_custom_object.assert_called_once_with(
            namespace="chs",
            group="resource.nvidia.com",
            version="v1beta1",
            plural="computedomains",
            label_selector="k=v",
            _preload_content=False,
            _request_timeout=120,
        )

        self.sdk.custom.list_cluster_custom_object.return_value = _FakeResponse(
            {"items": [{"metadata": {"name": "s1"}}]}
        )
        self.assertEqual(
            k8s_client.list_custom_objects(
                group="accelerator.gke.io",
                version="v1beta1",
                plural="slices",
                label_selector="app.kubernetes.io/managed-by=chs",
            ),
            [{"metadata": {"name": "s1"}}],
        )
        self.sdk.custom.list_cluster_custom_object.assert_called_once_with(
            group="accelerator.gke.io",
            version="v1beta1",
            plural="slices",
            label_selector="app.kubernetes.io/managed-by=chs",
            _preload_content=False,
            _request_timeout=120,
        )

    def test_delete_custom_objects(self) -> None:
        self.sdk.custom.delete_namespaced_custom_object.return_value = (
            _FakeResponse({})
        )
        self.sdk.custom.get_namespaced_custom_object.side_effect = _api_error(
            404, "NotFound", "not found"
        )
        k8s_client.delete_custom_objects(
            group="resource.nvidia.com",
            version="v1beta1",
            plural="computedomains",
            object_names=["cd1"],
            namespace="chs",
            wait_timeout_seconds=60,
            propagation_policy="Background",
        )
        self.sdk.custom.delete_namespaced_custom_object.assert_called_once_with(
            group="resource.nvidia.com",
            version="v1beta1",
            namespace="chs",
            plural="computedomains",
            name="cd1",
            propagation_policy="Background",
            _preload_content=False,
            _request_timeout=120,
        )

        self.sdk.custom.delete_cluster_custom_object.return_value = (
            _FakeResponse({})
        )
        self.sdk.custom.get_cluster_custom_object.side_effect = _api_error(
            404, "NotFound", "not found"
        )
        k8s_client.delete_custom_objects(
            group="accelerator.gke.io",
            version="v1beta1",
            plural="slices",
            object_names=["s1"],
        )
        self.sdk.custom.delete_cluster_custom_object.assert_called_once_with(
            group="accelerator.gke.io",
            version="v1beta1",
            plural="slices",
            name="s1",
            _preload_content=False,
            _request_timeout=120,
        )

    def test_apply_secret(self) -> None:
        body = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": "job1-ssh", "namespace": "chs"},
            "stringData": {"ssh-privatekey": "key"},
        }
        self.sdk.core.patch_namespaced_secret.return_value = _FakeResponse({})
        k8s_client.apply_secret(body=body)
        self.sdk.core.patch_namespaced_secret.assert_called_once_with(
            name="job1-ssh",
            namespace="chs",
            body=body,
            _preload_content=False,
            _request_timeout=120,
            field_manager="chs",
            force=True,
            _content_type="application/apply-patch+yaml",
        )

    def test_apply_service(self) -> None:
        body = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "chs", "namespace": "chs"},
            "spec": {"clusterIP": "None"},
        }
        self.sdk.core.patch_namespaced_service.return_value = _FakeResponse({})
        k8s_client.apply_service(body=body)
        self.sdk.core.patch_namespaced_service.assert_called_once_with(
            name="chs",
            namespace="chs",
            body=body,
            _preload_content=False,
            _request_timeout=120,
            field_manager="chs",
            force=True,
            _content_type="application/apply-patch+yaml",
        )

    def test_apply_and_patch_job(self) -> None:
        body = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": "job1", "namespace": "chs"},
            "spec": {"suspend": True},
        }
        self.sdk.batch.patch_namespaced_job.return_value = _FakeResponse({})
        k8s_client.apply_job(body=body)
        k8s_client.patch_job(
            name="job1", namespace="chs", body={"spec": {"suspend": False}}
        )
        self.assertEqual(
            self.sdk.batch.patch_namespaced_job.call_args_list,
            [
                mock.call(
                    name="job1",
                    namespace="chs",
                    body=body,
                    _preload_content=False,
                    _request_timeout=120,
                    field_manager="chs",
                    force=True,
                    _content_type="application/apply-patch+yaml",
                ),
                mock.call(
                    name="job1",
                    namespace="chs",
                    body={"spec": {"suspend": False}},
                    _preload_content=False,
                    _request_timeout=120,
                    _content_type="application/merge-patch+json",
                ),
            ],
        )

    def test_delete_jobs_uses_foreground_propagation_and_waits(self) -> None:
        self.sdk.batch.delete_namespaced_job.return_value = _FakeResponse({})
        self.sdk.batch.read_namespaced_job.side_effect = [
            _FakeResponse({}),
            _api_error(404, "NotFound", "not found"),
        ]
        k8s_client.delete_jobs(
            object_names=["job1"],
            namespace="chs",
            wait_timeout_seconds=60,
            propagation_policy="Orphan",
        )
        self.sdk.batch.delete_namespaced_job.assert_called_once_with(
            namespace="chs",
            name="job1",
            propagation_policy="Foreground",
            _preload_content=False,
            _request_timeout=120,
        )
        self.assertEqual(
            self.sdk.batch.read_namespaced_job.call_args_list,
            [
                mock.call(
                    namespace="chs",
                    name="job1",
                    _preload_content=False,
                    _request_timeout=120,
                )
            ]
            * 2,
        )

    def test_create_custom_object(self) -> None:
        cluster_body = {
            "apiVersion": "accelerator.gke.io/v1beta1",
            "kind": "Slice",
            "metadata": {"name": "s1"},
        }
        self.sdk.custom.create_cluster_custom_object.return_value = (
            _FakeResponse({})
        )
        k8s_client.create_custom_object(plural="slices", body=cluster_body)
        self.sdk.custom.create_cluster_custom_object.assert_called_once_with(
            group="accelerator.gke.io",
            version="v1beta1",
            plural="slices",
            body=cluster_body,
            _preload_content=False,
            _request_timeout=120,
        )
        self.sdk.custom.patch_cluster_custom_object.assert_not_called()

    def test_apply_custom_object(self) -> None:
        namespaced_body = {
            "apiVersion": "resource.nvidia.com/v1beta1",
            "kind": "ComputeDomain",
            "metadata": {"name": "cd1", "namespace": "chs"},
        }
        self.sdk.custom.patch_namespaced_custom_object.return_value = (
            _FakeResponse({})
        )
        k8s_client.apply_custom_object(
            plural="computedomains", body=namespaced_body
        )
        self.sdk.custom.patch_namespaced_custom_object.assert_called_once_with(
            group="resource.nvidia.com",
            version="v1beta1",
            namespace="chs",
            plural="computedomains",
            name="cd1",
            body=namespaced_body,
            _preload_content=False,
            _request_timeout=120,
            field_manager="chs",
            force=True,
            _content_type="application/apply-patch+yaml",
        )

        cluster_body = {
            "apiVersion": "accelerator.gke.io/v1beta1",
            "kind": "Slice",
            "metadata": {"name": "s1"},
        }
        self.sdk.custom.patch_cluster_custom_object.return_value = (
            _FakeResponse({})
        )
        k8s_client.apply_custom_object(
            plural="slices",
            body=cluster_body,
            field_manager="custom-mgr",
            force=False,
        )
        self.sdk.custom.patch_cluster_custom_object.assert_called_once_with(
            group="accelerator.gke.io",
            version="v1beta1",
            plural="slices",
            name="s1",
            body=cluster_body,
            _preload_content=False,
            _request_timeout=120,
            field_manager="custom-mgr",
            force=False,
            _content_type="application/apply-patch+yaml",
        )


class FakeKubeClientContractTest(unittest.TestCase):
    """Tests that FakeKubeClient mirrors k8s_client signatures and semantics."""

    def test_signatures_match_k8s_client(self) -> None:
        fake = k8s_fakes.FakeKubeClient()
        for name in k8s_fakes.RECORDED_METHODS:
            with self.subTest(method=name):
                real_params = list(
                    inspect.signature(
                        getattr(k8s_client, name)
                    ).parameters.values()
                )
                fake_params = list(
                    inspect.signature(getattr(fake, name)).parameters.values()
                )
                self.assertEqual(
                    [(p.name, p.kind, p.default) for p in fake_params],
                    [(p.name, p.kind, p.default) for p in real_params],
                )

    def test_namespaced_custom_object_without_namespace_raises_404(
        self,
    ) -> None:
        fake = k8s_fakes.FakeKubeClient()
        fake.add_object(
            "resource.nvidia.com",
            "computedomains",
            {"metadata": {"name": "cd1", "namespace": "chs"}},
        )
        with self.assertRaises(KubeError) as ctx:
            fake.delete_custom_objects(
                group="resource.nvidia.com",
                version="v1beta1",
                plural="computedomains",
                object_names=["cd1"],
            )
        self.assertEqual(ctx.exception.status, 404)

        with self.assertRaises(KubeError) as ctx:
            fake.apply_custom_object(
                plural="computedomains",
                body={
                    "apiVersion": "resource.nvidia.com/v1beta1",
                    "kind": "ComputeDomain",
                    "metadata": {"name": "cd1"},
                },
            )
        self.assertEqual(ctx.exception.status, 404)

    def test_delete_service_semantics(self) -> None:
        fake = k8s_fakes.FakeKubeClient()
        for name in ("chs-run1", "chs-run2"):
            fake.add_object(
                "", "services", {"metadata": {"name": name, "namespace": "chs"}}
            )

        fake.delete_service(name="chs-run1", namespace="chs")
        fake.delete_service(name="chs-run1", namespace="chs")

        self.assertEqual(fake.stored_names("", "services"), ["chs-run2"])


if __name__ == "__main__":
    unittest.main()
