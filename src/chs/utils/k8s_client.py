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

"""Kubernetes API access for CHS, built on the kubernetes Python SDK."""

import atexit
import functools
import json
import os

# Imported before lazy SDK load so socket outlives _sdk at shutdown.
import socket  # noqa: F401
import time
from types import SimpleNamespace
from typing import Any, BinaryIO, Callable
import urllib.parse
import urllib.request

from chs.exceptions import ChsException

_REQUEST_TIMEOUT_SECONDS = 120
_DELETE_POLL_INTERVAL_SECONDS = 0.5
_INCLUSTER_NAMESPACE_PATH = (
    "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
)


class KubeError(ChsException):
    """Raised when a Kubernetes API request fails."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def has_kubeconfig() -> bool:
    """Checks if a kubeconfig file exists at KUBECONFIG or ~/.kube/config."""
    kubeconfig = os.environ.get("KUBECONFIG")
    if kubeconfig:
        for path in kubeconfig.split(os.pathsep):
            if os.path.exists(os.path.expanduser(path.strip())):
                return True
        return False
    return os.path.exists(os.path.expanduser("~/.kube/config"))


def _close_api_client(api_client: Any) -> None:
    rest_client = getattr(api_client, "rest_client", None)
    pool_manager = getattr(rest_client, "pool_manager", None)
    if pool_manager is not None:
        pool_manager.clear()
    api_client.close()


@functools.cache
def _sdk() -> SimpleNamespace:
    # Imported on first use: the SDK takes ~1s to import
    import kubernetes
    import urllib3

    configuration = kubernetes.client.Configuration()
    context = None
    namespace = "default"
    if has_kubeconfig():
        try:
            kubernetes.config.load_kube_config(
                client_configuration=configuration,
                persist_config=False,
            )
            _, active = kubernetes.config.list_kube_config_contexts()
        except kubernetes.config.ConfigException as e:
            raise KubeError(
                f"Kubernetes configuration could not be loaded: {e}"
            ) from e
        context = active["name"]
        namespace = (active.get("context") or {}).get("namespace") or "default"
    elif "KUBERNETES_SERVICE_HOST" in os.environ:
        try:
            kubernetes.config.load_incluster_config(
                client_configuration=configuration
            )
        except kubernetes.config.ConfigException as e:
            raise KubeError(
                f"Kubernetes configuration could not be loaded: {e}"
            ) from e
        try:
            with open(_INCLUSTER_NAMESPACE_PATH, encoding="utf-8") as ns_file:
                namespace = ns_file.read().strip() or "default"
        except OSError:
            pass
    else:
        raise KubeError(
            "No kubeconfig found and not running inside a Kubernetes cluster"
        )

    # Match kubectl (Go http.ProxyFromEnvironment)
    parsed_host = urllib.parse.urlparse(configuration.host)
    configuration.proxy = (
        None
        if urllib.request.proxy_bypass_environment(parsed_host.netloc)
        else urllib.request.getproxies_environment().get(parsed_host.scheme)
    )
    configuration.no_proxy = None

    # Log fetching runs up to 32 job workers plus 32 pod-log workers.
    configuration.connection_pool_maxsize = 64
    api_client = kubernetes.client.ApiClient(configuration)
    api_client.set_default_header("Accept-Encoding", "gzip")
    atexit.register(_close_api_client, api_client)
    return SimpleNamespace(
        context=context,
        namespace=namespace,
        core=kubernetes.client.CoreV1Api(api_client),
        batch=kubernetes.client.BatchV1Api(api_client),
        custom=kubernetes.client.CustomObjectsApi(api_client),
        api_error=kubernetes.client.ApiException,
        transport_error=urllib3.exceptions.HTTPError,
    )


def _error_message(error: Any) -> str:
    try:
        return json.loads(error.body)["message"]
    except (TypeError, ValueError, KeyError):
        return f"{error.status} {error.reason}"


def _close_response(response: Any) -> None:
    response.close()
    response.release_conn()


def _open(method: Callable[..., Any], **query: Any) -> Any:
    """Calls an SDK method with _preload_content=False; caller must close."""
    sdk = _sdk()
    query.setdefault("_request_timeout", _REQUEST_TIMEOUT_SECONDS)
    try:
        return method(_preload_content=False, **query)
    except sdk.api_error as e:
        raise KubeError(_error_message(e), status=e.status) from e
    except sdk.transport_error as e:
        raise KubeError(f"Kubernetes API request failed: {e}") from e


def _request(method: Callable[..., Any], **query: Any) -> Any:
    """Calls an SDK method and reads the full response body."""
    response = _open(method, **query)
    try:
        _ = response.data
    except _sdk().transport_error as e:
        raise KubeError(f"Kubernetes API request failed: {e}") from e
    finally:
        _close_response(response)
    return response


def _get(method: Callable[..., Any], **query: Any) -> Any:
    try:
        return json.loads(_request(method, **query).data)
    except ValueError as e:
        raise KubeError(f"Invalid JSON in Kubernetes API response: {e}") from e


def _scoped(
    namespaced: Callable[..., Any],
    cluster_wide: Callable[..., Any],
    namespace: str | None = None,
    **path_params: Any,
) -> Callable[..., Any]:
    if namespace:
        return functools.partial(namespaced, namespace=namespace, **path_params)
    return functools.partial(cluster_wide, **path_params)


def _list(
    namespaced: Callable[..., Any],
    cluster_wide: Callable[..., Any],
    namespace: str | None = None,
    **query: Any,
) -> list[dict]:
    body = _get(_scoped(namespaced, cluster_wide, namespace), **query)
    if not isinstance(body, dict):
        raise KubeError(
            "Invalid Kubernetes API list response: "
            f"expected dict, got {type(body).__name__}"
        )
    return body.get("items") or []


def _delete_and_wait(
    delete: Callable[..., Any],
    read: Callable[..., Any],
    object_names: list[str],
    wait_timeout_seconds: float,
    **query: Any,
) -> None:
    if not object_names:
        return
    deadline = time.monotonic() + wait_timeout_seconds
    pending = []
    for name in object_names:
        try:
            _request(delete, name=name, **query)
            pending.append(name)
        except KubeError as e:
            if e.status != 404:
                raise

    while True:
        remaining = []
        for name in pending:
            try:
                _request(read, name=name)
                remaining.append(name)
            except KubeError as e:
                if e.status != 404:
                    raise
        pending = remaining
        if not pending:
            return
        if time.monotonic() > deadline:
            raise KubeError(
                f"Timed out waiting for deletion of {', '.join(pending)}"
            )
        time.sleep(_DELETE_POLL_INTERVAL_SECONDS)


def current_context() -> str | None:
    """Returns the active kubeconfig context, or None when in-cluster."""
    return _sdk().context


def list_nodes(**query: Any) -> list[dict]:
    return _get(_sdk().core.list_node, **query)["items"]


def get_node(*, name: str, **query: Any) -> dict:
    return _get(_sdk().core.read_node, name=name, **query)


def list_pods(*, namespace: str | None = None, **query: Any) -> list[dict]:
    """Lists pods in a namespace, or in all namespaces if none is given."""
    core = _sdk().core
    return _list(
        core.list_namespaced_pod,
        core.list_pod_for_all_namespaces,
        namespace=namespace,
        **query,
    )


_LOG_STREAM_MAX_RETRIES = 4
_LOG_STREAM_INITIAL_BACKOFF = 1.0


def _stream_once(
    sdk: SimpleNamespace, name: str, namespace: str, out: BinaryIO, query: dict
) -> None:
    response = _open(
        sdk.core.read_namespaced_pod_log,
        name=name,
        namespace=namespace,
        **query,
    )
    try:
        for chunk in response.stream():
            out.write(chunk)
    except sdk.transport_error as e:
        raise KubeError(f"Failed to read log of pod {name}: {e}") from e
    finally:
        _close_response(response)


def stream_pod_log(*, pod: dict, out: BinaryIO, **query: Any) -> None:
    """Writes the log of the pod's default (or specified) container to out."""
    metadata = pod["metadata"]
    if "container" not in query:
        query["container"] = (metadata.get("annotations") or {}).get(
            "kubectl.kubernetes.io/default-container"
        ) or pod["spec"]["containers"][0]["name"]
    sdk = _sdk()
    name = metadata["name"]
    namespace = metadata.get("namespace") or sdk.namespace
    start_pos = out.tell() if out.seekable() else None
    for attempt in range(_LOG_STREAM_MAX_RETRIES + 1):
        if attempt > 0 and start_pos is not None:
            out.seek(start_pos)
            out.truncate()
        try:
            _stream_once(sdk, name, namespace, out, query)
            return
        except KubeError as e:
            if attempt == _LOG_STREAM_MAX_RETRIES or (
                e.status is not None and e.status < 500 and e.status != 429
            ):
                raise
        time.sleep(_LOG_STREAM_INITIAL_BACKOFF * (2**attempt))


def create_namespace_if_not_exists(*, name: str, **query: Any) -> None:
    """Creates the namespace if it does not already exist."""
    try:
        _request(
            _sdk().core.create_namespace,
            body={"metadata": {"name": name}},
            **query,
        )
    except KubeError as e:
        if e.status != 409:
            raise


def delete_namespace(
    *,
    name: str,
    wait_timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS,
    **query: Any,
) -> None:
    core = _sdk().core
    _delete_and_wait(
        core.delete_namespace,
        core.read_namespace,
        [name],
        wait_timeout_seconds,
        **query,
    )


def list_secrets(*, namespace: str | None = None, **query: Any) -> list[dict]:
    core = _sdk().core
    return _list(
        core.list_namespaced_secret,
        core.list_secret_for_all_namespaces,
        namespace=namespace,
        **query,
    )


def delete_secret(
    *, name: str, namespace: str | None = None, **query: Any
) -> None:
    sdk = _sdk()
    try:
        _request(
            sdk.core.delete_namespaced_secret,
            name=name,
            namespace=namespace or sdk.namespace,
            **query,
        )
    except KubeError as e:
        if e.status != 404:
            raise


def apply_secret(*, body: dict, **query: Any) -> None:
    """Creates or updates the Secret with server-side apply."""
    metadata = body["metadata"]
    query.setdefault("field_manager", "chs")
    query.setdefault("force", True)
    query.setdefault("_content_type", "application/apply-patch+yaml")
    _request(
        _sdk().core.patch_namespaced_secret,
        name=metadata["name"],
        namespace=metadata["namespace"],
        body=body,
        **query,
    )


def apply_service(*, body: dict, **query: Any) -> None:
    """Creates or updates the Service with server-side apply."""
    metadata = body["metadata"]
    query.setdefault("field_manager", "chs")
    query.setdefault("force", True)
    query.setdefault("_content_type", "application/apply-patch+yaml")
    _request(
        _sdk().core.patch_namespaced_service,
        name=metadata["name"],
        namespace=metadata["namespace"],
        body=body,
        **query,
    )


def delete_service(*, name: str, namespace: str, **query: Any) -> None:
    """Deletes the Service; a Service that does not exist is ignored."""
    try:
        _request(
            _sdk().core.delete_namespaced_service,
            name=name,
            namespace=namespace,
            **query,
        )
    except KubeError as e:
        if e.status != 404:
            raise


def list_jobs(*, namespace: str | None = None, **query: Any) -> list[dict]:
    batch = _sdk().batch
    return _list(
        batch.list_namespaced_job,
        batch.list_job_for_all_namespaces,
        namespace=namespace,
        **query,
    )


def apply_job(*, body: dict, **query: Any) -> None:
    """Creates or updates the batch/v1 Job with server-side apply."""
    metadata = body["metadata"]
    query.setdefault("field_manager", "chs")
    query.setdefault("force", True)
    query.setdefault("_content_type", "application/apply-patch+yaml")
    _request(
        _sdk().batch.patch_namespaced_job,
        name=metadata["name"],
        namespace=metadata["namespace"],
        body=body,
        **query,
    )


def patch_job(*, name: str, namespace: str, body: dict, **query: Any) -> None:
    """Patches the batch/v1 Job with a JSON merge patch."""
    query.setdefault("_content_type", "application/merge-patch+json")
    _request(
        _sdk().batch.patch_namespaced_job,
        name=name,
        namespace=namespace,
        body=body,
        **query,
    )


def delete_jobs(
    *,
    object_names: list[str],
    namespace: str,
    wait_timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS,
    **query: Any,
) -> None:
    """Deletes the named Jobs and waits until they and their pods are gone.

    Foreground propagation keeps each Job until its pods are deleted, so
    waiting for the Job to disappear also waits for its pods. It is not
    overridable: the default for batch/v1 Jobs orphans the pods.
    """
    batch = _sdk().batch
    query["propagation_policy"] = "Foreground"
    _delete_and_wait(
        functools.partial(batch.delete_namespaced_job, namespace=namespace),
        functools.partial(batch.read_namespaced_job, namespace=namespace),
        object_names,
        wait_timeout_seconds,
        **query,
    )


def list_custom_objects(
    *,
    group: str,
    version: str,
    plural: str,
    namespace: str | None = None,
    **query: Any,
) -> list[dict]:
    """Lists objects in a namespace, or cluster-wide if none is given."""
    custom = _sdk().custom
    return _list(
        custom.list_namespaced_custom_object,
        custom.list_cluster_custom_object,
        namespace=namespace,
        group=group,
        version=version,
        plural=plural,
        **query,
    )


def delete_custom_objects(
    *,
    group: str,
    version: str,
    plural: str,
    object_names: list[str],
    namespace: str | None = None,
    wait_timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS,
    **query: Any,
) -> None:
    """Deletes the named objects and waits until they are gone."""
    custom = _sdk().custom
    coords = {"group": group, "version": version, "plural": plural}
    delete = _scoped(
        custom.delete_namespaced_custom_object,
        custom.delete_cluster_custom_object,
        namespace,
        **coords,
    )
    read = _scoped(
        custom.get_namespaced_custom_object,
        custom.get_cluster_custom_object,
        namespace,
        **coords,
    )
    _delete_and_wait(delete, read, object_names, wait_timeout_seconds, **query)


def create_custom_object(*, plural: str, body: dict, **query: Any) -> None:
    """Creates the object.

    Raises:
      KubeError: With status 409 if the object already exists.
    """
    group, version = body["apiVersion"].split("/")
    custom = _sdk().custom
    _request(
        _scoped(
            custom.create_namespaced_custom_object,
            custom.create_cluster_custom_object,
            body["metadata"].get("namespace"),
            group=group,
            version=version,
            plural=plural,
        ),
        body=body,
        **query,
    )


def apply_custom_object(*, plural: str, body: dict, **query: Any) -> None:
    """Creates or updates the object with server-side apply."""
    group, version = body["apiVersion"].split("/")
    metadata = body["metadata"]
    query.setdefault("field_manager", "chs")
    query.setdefault("force", True)
    query.setdefault("_content_type", "application/apply-patch+yaml")
    custom = _sdk().custom
    _request(
        _scoped(
            custom.patch_namespaced_custom_object,
            custom.patch_cluster_custom_object,
            metadata.get("namespace"),
            group=group,
            version=version,
            plural=plural,
        ),
        name=metadata["name"],
        body=body,
        **query,
    )
