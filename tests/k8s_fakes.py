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

"""In-memory fake for chs.utils.k8s_client."""

from collections.abc import Callable, Iterator
import contextlib
import copy
import dataclasses
import functools
import inspect
import itertools
import re
from typing import Any, BinaryIO, NamedTuple
import unittest
from unittest import mock

from chs.utils import k8s_client
from chs.utils.k8s_client import KubeError


class RealKubeClientError(BaseException):
    """Raised when a unit test attempts to create a real Kubernetes client."""


REAL_SDK = getattr(k8s_client._sdk, "_real_fn", k8s_client._sdk)


def _guard_sdk() -> Any:
    raise RealKubeClientError(
        "Unit test tried to create a real Kubernetes client; use "
        "k8s_fakes.install_fake_client(self) or k8s_fakes.patch_k8s_client()."
    )


_guard_sdk._real_fn = REAL_SDK  # type: ignore[attr-defined]
_guard_sdk.cache_clear = REAL_SDK.cache_clear  # type: ignore[attr-defined]
k8s_client._sdk = _guard_sdk  # type: ignore[assignment]
REAL_SDK.cache_clear()


@contextlib.contextmanager
def real_client_allowed() -> Iterator[None]:
    """Allows k8s_client._sdk() to construct real SDK clients in tests."""
    REAL_SDK.cache_clear()
    try:
        with mock.patch.object(k8s_client, "_sdk", REAL_SDK):
            yield
    finally:
        REAL_SDK.cache_clear()


@dataclasses.dataclass(frozen=True)
class FakeResourceType:
    """Metadata for a resource type served by FakeKubeClient."""

    group: str
    plural: str
    kind: str
    namespaced: bool
    versions: tuple[str, ...] = ("v1",)

    @property
    def api_version(self) -> str:
        return (
            f"{self.group}/{self.versions[0]}"
            if self.group
            else self.versions[0]
        )


DEFAULT_TYPES = (
    FakeResourceType("", "nodes", "Node", False),
    FakeResourceType("", "namespaces", "Namespace", False),
    FakeResourceType("", "pods", "Pod", True),
    FakeResourceType("", "configmaps", "ConfigMap", True),
    FakeResourceType("", "secrets", "Secret", True),
    FakeResourceType("apps", "deployments", "Deployment", True),
    FakeResourceType("batch", "jobs", "Job", True),
    FakeResourceType(
        "apiextensions.k8s.io",
        "customresourcedefinitions",
        "CustomResourceDefinition",
        False,
    ),
    FakeResourceType(
        "jobset.x-k8s.io", "jobsets", "JobSet", True, ("v1alpha2",)
    ),
    FakeResourceType(
        "networking.gke.io",
        "networks",
        "Network",
        False,
        ("v1", "v1beta2", "v1beta1", "v1alpha1"),
    ),
    FakeResourceType(
        "accelerator.gke.io", "slices", "Slice", False, ("v1beta1",)
    ),
    FakeResourceType(
        "resource.nvidia.com",
        "computedomains",
        "ComputeDomain",
        True,
        ("v1beta1",),
    ),
)


def _merge_patch(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, val in patch.items():
        if val is None:
            target.pop(key, None)
        elif isinstance(val, dict) and isinstance(target.get(key), dict):
            _merge_patch(target[key], val)
        else:
            target[key] = copy.deepcopy(val)


class Call(NamedTuple):
    """A recorded k8s_client function invocation."""

    method: str
    kwargs: dict[str, Any]


RECORDED_METHODS = frozenset(
    {
        "current_context",
        "get_node",
        "list_nodes",
        "list_pods",
        "stream_pod_log",
        "create_namespace_if_not_exists",
        "delete_namespace",
        "list_secrets",
        "delete_secret",
        "apply_secret",
        "list_jobs",
        "list_deployments",
        "get_deployment",
        "list_crds",
        "get_crd",
        "get_custom_object",
        "list_custom_objects",
        "delete_custom_objects",
        "apply_custom_object",
        "patch_custom_object",
    }
)


def _recorded(copy_args: bool = True) -> Callable[..., Any]:
    def decorate(method: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(method)

        @functools.wraps(method)
        def wrapper(self: "FakeKubeClient", *args: Any, **kwargs: Any) -> Any:
            sig.bind(self, *args, **kwargs)
            self._record(method.__name__, kwargs, copy_args)
            return method(self, *args, **kwargs)

        return wrapper

    return decorate


class FakeKubeClient:
    """In-memory stand-in for chs.utils.k8s_client."""

    def __init__(
        self,
        namespace: str = "default",
        types: tuple[FakeResourceType, ...] = DEFAULT_TYPES,
        context: str | None = None,
    ) -> None:
        self.objects: dict[
            tuple[str, str, str | None, str], dict[str, Any]
        ] = {}
        self.calls: list[Call] = []
        self.context = context
        self.namespace = namespace
        self._types = {(t.group, t.plural): t for t in types}
        self._next_failures: dict[str, list[BaseException | None]] = {}
        self._permanent_failures: dict[str, BaseException] = {}
        self._uids = itertools.count(1)

    def unregister_type(self, group: str, plural: str) -> None:
        """Stops serving a resource type."""
        del self._types[(group, plural)]

    def add_object(
        self,
        group: str,
        plural: str,
        obj: dict[str, Any],
        *,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Stores a copy of `obj` in the fake cluster."""
        rtype = self._types.get((group, plural))
        if rtype is None:
            raise ValueError(f"{plural!r} in group {group!r} is not served")
        stored = copy.deepcopy(obj)
        metadata = stored.setdefault("metadata", {})
        name = metadata.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("the object needs a metadata.name")
        object_namespace = None
        if rtype.namespaced:
            object_namespace = (
                metadata.get("namespace") or namespace or self.namespace
            )
            metadata["namespace"] = object_namespace
        else:
            metadata.pop("namespace", None)
        self._store((group, plural, object_namespace, name), stored, rtype)
        return copy.deepcopy(stored)

    def add_crd(self, name: str, versions: tuple[str, ...] = ("v1",)) -> None:
        """Stores a CustomResourceDefinition object."""
        plural, _, group = name.partition(".")
        self.add_object(
            "apiextensions.k8s.io",
            "customresourcedefinitions",
            {
                "metadata": {"name": name},
                "spec": {
                    "group": group,
                    "names": {"plural": plural},
                    "versions": [
                        {"name": version, "served": True, "storage": idx == 0}
                        for idx, version in enumerate(versions)
                    ],
                },
            },
        )

    def stored_names(
        self, group: str, plural: str, namespace: str | None = None
    ) -> list[str]:
        """Returns stored object names for a resource type."""
        return [
            name
            for (obj_group, obj_plural, obj_ns, name) in self.objects
            if (obj_group, obj_plural) == (group, plural)
            and (namespace is None or obj_ns == namespace)
        ]

    def fail_next(self, method: str, *errors: BaseException | None) -> None:
        """Queues exceptions (or None for pass-through) for subsequent calls."""
        self._check_method(method)
        self._next_failures.setdefault(method, []).extend(errors)

    def fail_always(self, method: str, error: BaseException) -> None:
        """Makes every call to `method` raise `error`."""
        self._check_method(method)
        self._permanent_failures[method] = error

    def calls_to(self, method: str) -> list[Call]:
        """Returns recorded calls to `method`."""
        self._check_method(method)
        return [call for call in self.calls if call.method == method]

    @_recorded()
    def current_context(self) -> str | None:
        return self.context

    @_recorded()
    def list_nodes(self, **query: Any) -> list[dict[str, Any]]:
        return self._list_many("", "nodes", **query)

    @_recorded()
    def get_node(self, *, name: str, **query: Any) -> dict[str, Any]:
        del query
        return self._get_one("", "nodes", name)

    @_recorded()
    def list_pods(
        self,
        *,
        namespace: str | None = None,
        **query: Any,
    ) -> list[dict[str, Any]]:
        return self._list_many("", "pods", namespace=namespace, **query)

    @_recorded(copy_args=False)
    def stream_pod_log(
        self, *, pod: dict[str, Any], out: BinaryIO, **query: Any
    ) -> None:
        del out, query
        pods = self._served("", "pods")
        metadata = pod["metadata"]
        name = metadata["name"]
        namespace = metadata.get("namespace") or self.namespace
        if self._key(pods, name, namespace) not in self.objects:
            raise _not_found(pods, name)

    @_recorded()
    def create_namespace_if_not_exists(
        self, *, name: str, **query: Any
    ) -> None:
        del query
        if ("", "namespaces", None, name) not in self.objects:
            self.add_object(
                "",
                "namespaces",
                {"metadata": {"name": name}, "status": {"phase": "Active"}},
            )

    @_recorded()
    def delete_namespace(
        self,
        *,
        name: str,
        wait_timeout_seconds: float = k8s_client._REQUEST_TIMEOUT_SECONDS,
        **query: Any,
    ) -> None:
        del wait_timeout_seconds, query
        rtype = self._served("", "namespaces")
        key = self._key(rtype, name, None)
        if key in self.objects:
            self._remove(key)

    @_recorded()
    def list_secrets(
        self, *, namespace: str | None = None, **query: Any
    ) -> list[dict[str, Any]]:
        return self._list_many("", "secrets", namespace=namespace, **query)

    @_recorded()
    def delete_secret(
        self, *, name: str, namespace: str | None = None, **query: Any
    ) -> None:
        del query
        rtype = self._served("", "secrets")
        key = self._key(rtype, name, namespace or self.namespace)
        if key in self.objects:
            self._remove(key)

    @_recorded()
    def apply_secret(self, *, body: dict[str, Any], **query: Any) -> None:
        del query
        rtype = self._served("", "secrets")
        stored = copy.deepcopy(body)
        key = self._key(
            rtype, stored["metadata"]["name"], stored["metadata"]["namespace"]
        )
        self._store(key, stored, rtype)

    @_recorded()
    def list_jobs(
        self, *, namespace: str | None = None, **query: Any
    ) -> list[dict[str, Any]]:
        return self._list_many("batch", "jobs", namespace=namespace, **query)

    @_recorded()
    def list_deployments(
        self, *, namespace: str | None = None, **query: Any
    ) -> list[dict[str, Any]]:
        return self._list_many(
            "apps", "deployments", namespace=namespace, **query
        )

    @_recorded()
    def get_deployment(
        self, *, name: str, namespace: str | None = None, **query: Any
    ) -> dict[str, Any]:
        del query
        return self._get_one(
            "apps",
            "deployments",
            name,
            namespace=namespace or self.namespace,
        )

    @_recorded()
    def list_crds(self, **query: Any) -> list[dict[str, Any]]:
        return self._list_many(
            "apiextensions.k8s.io", "customresourcedefinitions", **query
        )

    @_recorded()
    def get_crd(self, *, name: str, **query: Any) -> dict[str, Any]:
        del query
        return self._get_one(
            "apiextensions.k8s.io", "customresourcedefinitions", name
        )

    @_recorded()
    def list_custom_objects(
        self,
        *,
        group: str,
        version: str,
        plural: str,
        namespace: str | None = None,
        **query: Any,
    ) -> list[dict[str, Any]]:
        del version
        return self._list_many(group, plural, namespace=namespace, **query)

    @_recorded()
    def get_custom_object(
        self,
        *,
        group: str,
        version: str,
        plural: str,
        name: str,
        namespace: str | None = None,
        **query: Any,
    ) -> dict[str, Any]:
        del version, query
        return self._get_one(group, plural, name, namespace=namespace)

    @_recorded()
    def delete_custom_objects(
        self,
        *,
        group: str,
        version: str,
        plural: str,
        object_names: list[str],
        namespace: str | None = None,
        wait_timeout_seconds: float = k8s_client._REQUEST_TIMEOUT_SECONDS,
        **query: Any,
    ) -> None:
        del version, wait_timeout_seconds, query
        rtype = self._served(group, plural)
        for name in object_names:
            key = self._key(rtype, name, namespace)
            if key in self.objects:
                self._remove(key)

    @_recorded()
    def apply_custom_object(
        self, *, plural: str, body: dict[str, Any], **query: Any
    ) -> None:
        del query
        group, version = body["apiVersion"].split("/")
        rtype = self._served(group, plural)
        if version not in rtype.versions:
            raise KubeError(
                f'no matches for kind "{body.get("kind")}" in version '
                f'"{body["apiVersion"]}"',
                status=404,
            )
        stored = copy.deepcopy(body)
        object_namespace = stored["metadata"].get("namespace")
        if not rtype.namespaced:
            object_namespace = None
            stored["metadata"].pop("namespace", None)
        key = self._key(rtype, stored["metadata"]["name"], object_namespace)
        existing = self.objects.get(key)
        if existing is not None:
            if "status" in existing:
                stored["status"] = copy.deepcopy(existing["status"])
            if uid := existing.get("metadata", {}).get("uid"):
                stored["metadata"]["uid"] = uid
        self._store(key, stored, rtype)

    @_recorded()
    def patch_custom_object(
        self,
        *,
        group: str,
        version: str,
        plural: str,
        name: str,
        body: dict[str, Any],
        namespace: str | None = None,
        **query: Any,
    ) -> None:
        del query
        rtype = self._served(group, plural)
        if version not in rtype.versions:
            raise KubeError(
                f'no matches for version "{version}" in {group}',
                status=404,
            )
        key = self._key(rtype, name, namespace)
        existing = self.objects.get(key)
        if existing is None:
            raise _not_found(rtype, name)
        _merge_patch(existing, copy.deepcopy(body))

    def _get_one(
        self,
        group: str,
        plural: str,
        name: str,
        *,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        rtype = self._served(group, plural)
        key = self._key(rtype, name, namespace)
        obj = self.objects.get(key)
        if obj is None:
            raise _not_found(rtype, name)
        return copy.deepcopy(obj)

    def _list_many(
        self,
        group: str,
        plural: str,
        *,
        namespace: str | None = None,
        label_selector: str | None = None,
        field_selector: str | None = None,
        **_query: Any,
    ) -> list[dict[str, Any]]:
        rtype = self._served(group, plural)
        keys = self._select(rtype, namespace, label_selector, field_selector)
        return [copy.deepcopy(self.objects[key]) for key in keys]

    def _record(
        self,
        method: str,
        kwargs: dict[str, Any],
        copy_args: bool,
    ) -> None:
        if copy_args:
            kwargs = copy.deepcopy(kwargs)
        self.calls.append(Call(method, dict(kwargs)))
        queued = self._next_failures.get(method)
        if queued:
            err = queued.pop(0)
            if err is not None:
                raise err
        permanent = self._permanent_failures.get(method)
        if permanent is not None:
            raise permanent

    def _check_method(self, method: str) -> None:
        if method not in RECORDED_METHODS:
            raise ValueError(
                f"{method!r} is not one of {sorted(RECORDED_METHODS)}"
            )

    def _served(self, group: str, plural: str) -> FakeResourceType:
        rtype = self._types.get((group, plural))
        if rtype is None:
            raise KubeError(
                f'the server doesn\'t have a resource type "{plural}"',
                status=404,
            )
        return rtype

    def _key(
        self, rtype: FakeResourceType, name: Any, namespace: Any
    ) -> tuple[str, str, str | None, str]:
        object_namespace = None
        if rtype.namespaced:
            if not namespace:
                raise _not_found(rtype, name)
            object_namespace = namespace
        return (rtype.group, rtype.plural, object_namespace, name)

    def _store(
        self,
        key: tuple[str, str, str | None, str],
        obj: dict[str, Any],
        rtype: FakeResourceType,
    ) -> None:
        obj.setdefault("apiVersion", rtype.api_version)
        obj.setdefault("kind", rtype.kind)
        obj["metadata"].setdefault("uid", f"fake-uid-{next(self._uids)}")
        self.objects[key] = obj

    def _remove(self, key: tuple[str, str, str | None, str]) -> None:
        del self.objects[key]
        if key[:3] == ("", "namespaces", None):
            for other in list(self.objects):
                if other[2] == key[3]:
                    del self.objects[other]

    def _select(
        self,
        rtype: FakeResourceType,
        namespace: str | None,
        label_selector: str | None,
        field_selector: str | None,
    ) -> list[tuple[str, str, str | None, str]]:
        list_namespace = namespace if rtype.namespaced and namespace else None
        label_matches = (
            _parse_label_selector(label_selector) if label_selector else None
        )
        field_matches = (
            _parse_field_selector(field_selector) if field_selector else None
        )
        keys = []
        for key, obj in self.objects.items():
            group, plural, object_namespace, _ = key
            if (group, plural) != (rtype.group, rtype.plural):
                continue
            if (
                list_namespace is not None
                and object_namespace != list_namespace
            ):
                continue
            labels = obj.get("metadata", {}).get("labels") or {}
            if label_matches is not None and not label_matches(labels):
                continue
            if field_matches is not None and not field_matches(obj):
                continue
            keys.append(key)
        return keys


def _not_found(rtype: FakeResourceType, name: str) -> KubeError:
    resource = f"{rtype.plural}.{rtype.group}" if rtype.group else rtype.plural
    return KubeError(
        f'Error from server (NotFound): {resource} "{name}" not found',
        status=404,
    )


def _parse_label_selector(
    selector: str,
) -> Callable[[dict[str, Any]], bool]:
    checks: list[Callable[[dict[str, Any]], bool]] = []
    for raw_term in re.split(r",(?![^()]*\))", selector):
        term = raw_term.strip()
        if match := re.fullmatch(r"(\S+)\s+(in|notin)\s*\(([^()]*)\)", term):
            key, op, raw_vals = match.groups()
            vals = {v.strip() for v in raw_vals.split(",") if v.strip()}
            checks.append(
                lambda labels, k=key, v=vals, is_in=(op == "in"): (
                    (labels.get(k) in v) == is_in
                )
            )
        elif "!=" in term:
            key, val = (s.strip() for s in term.split("!=", 1))
            checks.append(lambda labels, k=key, v=val: labels.get(k) != v)
        elif "=" in term:
            key, val = (s.strip() for s in re.split(r"==?", term, maxsplit=1))
            checks.append(lambda labels, k=key, v=val: labels.get(k) == v)
        elif term.startswith("!"):
            checks.append(lambda labels, k=term[1:].strip(): k not in labels)
        else:
            checks.append(lambda labels, k=term: k in labels)
    return lambda labels: all(check(labels) for check in checks)


def _field_value(obj: dict[str, Any], path: str) -> str:
    val: Any = obj
    for part in path.split("."):
        val = val.get(part) if isinstance(val, dict) else None
    return "" if val is None else str(val)


def _parse_field_selector(
    selector: str,
) -> Callable[[dict[str, Any]], bool]:
    checks: list[Callable[[dict[str, Any]], bool]] = []
    for raw_term in selector.split(","):
        term = raw_term.strip()
        if "!=" in term:
            path, val = (s.strip() for s in term.split("!=", 1))
            checks.append(lambda obj, p=path, v=val: _field_value(obj, p) != v)
        else:
            path, val = (s.strip() for s in re.split(r"==?", term, maxsplit=1))
            checks.append(lambda obj, p=path, v=val: _field_value(obj, p) == v)
    return lambda obj: all(check(obj) for check in checks)


@contextlib.contextmanager
def patch_k8s_client(
    fake: FakeKubeClient | None = None,
    *,
    error: BaseException | None = None,
) -> Iterator[FakeKubeClient | None]:
    """Patches k8s_client functions to delegate to `fake` or raise `error`."""
    if fake is not None and error is not None:
        raise ValueError("pass either fake or error, not both")
    REAL_SDK.cache_clear()
    with contextlib.ExitStack() as stack:
        if error is not None:
            for method in RECORDED_METHODS:
                stack.enter_context(
                    mock.patch.object(k8s_client, method, side_effect=error)
                )
            yield None
            return
        if fake is None:
            fake = FakeKubeClient()
        for method in RECORDED_METHODS:
            stack.enter_context(
                mock.patch.object(k8s_client, method, getattr(fake, method))
            )
        yield fake


def install_fake_client(
    test_case: unittest.TestCase,
    fake: FakeKubeClient | None = None,
    *,
    error: BaseException | None = None,
) -> FakeKubeClient | None:
    """Installs a FakeKubeClient for the duration of `test_case`."""
    stack = contextlib.ExitStack()
    installed = stack.enter_context(patch_k8s_client(fake, error=error))
    test_case.addCleanup(stack.close)
    return installed
