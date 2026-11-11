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

"""Asynchronous telemetry module for Cluster Health Scanner (CHS).

Telemetry is off until enable() is called. Only the chs entry point calls it,
so tests and code that imports chs as a library never send anything. It skips
the call for users who ran `chs configure telemetry disable` or set
CHS_TELEMETRY=0 or DO_NOT_TRACK=1.
"""

import atexit
import contextlib
import dataclasses
import json
import logging
import queue
import re
import threading
import time
import uuid
from collections.abc import Collection
from typing import Any

from chs.telemetry.supported_metrics import MetricName
from chs.telemetry.telemetry_util import is_googler, post
from chs.version import get_version

logger = logging.getLogger(__name__)

_MAX_QUEUE_SIZE = 50
_WINDOW_S = 10.0
_HTTP_TIMEOUT_S = 5.0
_MAX_ATTEMPTS = 3
_RETRY_DELAY_S = 1.0
_EXIT_TIMEOUT_S = 2.0

_CLIENT_TYPE = "CLUSTER_HEALTH_SCANNER"


@dataclasses.dataclass
class _Metric:
    name: str
    value: str
    time_ms: int


@dataclasses.dataclass
class _State:
    enabled: bool = False
    worker_thread: threading.Thread | None = None
    run_id: str | None = None
    project_id: str | None = None


_state = _State()
_queue: queue.Queue[_Metric | object] = queue.Queue(maxsize=_MAX_QUEUE_SIZE)
_lock = threading.Lock()

_exit_requested = threading.Event()
_exit_done = threading.Event()

_SESSION_ID = str(uuid.uuid4())
_WORKSPACE_NAME_RE = re.compile(r"(\d{8}-\d{6})_.+-([a-z0-9]{4})")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _worker() -> None:
    logger.debug("Telemetry background thread started.")
    try:
        _process_queue()
    except Exception as e:
        logger.debug("Telemetry stopped: %s", e)
    finally:
        _state.enabled = False
        _exit_done.set()


def _process_queue() -> None:
    batch: dict[str, _Metric] = {}
    due_at: float | None = None
    while True:
        timeout = None
        if due_at is not None:
            timeout = max(0.0, due_at - time.monotonic())
        try:
            item = _queue.get(timeout=timeout)
        except queue.Empty:
            item = None
        if not _state.enabled:
            return
        if isinstance(item, _Metric):
            batch[item.name] = item
            if due_at is None:
                due_at = time.monotonic() + _WINDOW_S
        if _exit_requested.is_set():
            _exit_flush(batch)
            return
        if due_at is not None and time.monotonic() >= due_at:
            _send(batch)
            due_at = None


def _send(batch: dict[str, _Metric]) -> None:
    for attempt in range(_MAX_ATTEMPTS):
        if attempt and _exit_requested.wait(
            _RETRY_DELAY_S * 2 ** (attempt - 1)
        ):
            return
        retry = post(_build_request(batch.values()), _HTTP_TIMEOUT_S)
        if not retry:
            break
    batch.clear()


def _build_request(metrics: Collection[_Metric]) -> bytes:
    values = {metric.name: metric.value for metric in metrics}
    if _state.run_id:
        values["RUN_ID"] = _state.run_id
    concord_event = {
        "console_type": _CLIENT_TYPE,
        "event_type": "chsCLI",
        "release_version": get_version(),
        "client_session_id": _SESSION_ID,
        "is_googler": is_googler(),
        "event_metadata": [
            {"key": f"{_CLIENT_TYPE}_{name}", "value": value}
            for name, value in values.items()
        ],
    }
    if _state.project_id:
        concord_event["project_id"] = _state.project_id
    log_request = {
        "client_info": {"client_type": _CLIENT_TYPE},
        "log_source_name": "CONCORD",
        "request_time_ms": _now_ms(),
        "log_event": [
            {
                "event_time_ms": min(metric.time_ms for metric in metrics),
                "source_extension_json": json.dumps(concord_event),
            }
        ],
    }
    return json.dumps(log_request).encode()


def _exit_flush(batch: dict[str, _Metric]) -> None:
    with contextlib.suppress(queue.Empty):
        while True:
            item = _queue.get_nowait()
            if isinstance(item, _Metric):
                batch[item.name] = item
    if batch:
        post(_build_request(batch.values()), _EXIT_TIMEOUT_S)


def _on_exit() -> None:
    if _exit_done.is_set() or not _state.enabled:
        return
    _exit_requested.set()
    with contextlib.suppress(queue.Full):
        _queue.put_nowait(object())
    with contextlib.suppress(KeyboardInterrupt):
        _exit_done.wait(_EXIT_TIMEOUT_S)


def _start_worker() -> bool:
    if _state.worker_thread is not None:
        return True

    with _lock:
        if _state.worker_thread is None:
            thread = threading.Thread(
                target=_worker,
                name="chs-telemetry",
                daemon=True,
            )
            try:
                thread.start()
            except RuntimeError as e:
                logger.debug("Failed to start telemetry thread: %s", e)
                return False
            _state.worker_thread = thread
            atexit.register(_on_exit)
    return True


def enable() -> None:
    _state.enabled = True


def send_metric(name: MetricName, value: Any) -> None:
    try:
        if not _state.enabled or not _start_worker():
            return
        name = name.value if isinstance(name, MetricName) else str(name)
        if isinstance(value, (dict, list)):
            value = json.dumps(value, separators=(",", ":"))
        _queue.put_nowait(_Metric(name, str(value), _now_ms()))
    except queue.Full:
        logger.debug("Telemetry queue is full; dropping metric '%s'.", name)
    except Exception as e:
        logger.debug("Failed to enqueue telemetry metric '%s': %s", name, e)


def set_telemetry_run_id(workspace_name: str | None) -> None:
    if not isinstance(workspace_name, str):
        return
    match = _WORKSPACE_NAME_RE.fullmatch(workspace_name)
    if not match:
        return
    run_id = f"{match.group(1)}-{match.group(2)}"
    _state.run_id = run_id


def set_telemetry_project_id(project_id: str | None) -> None:
    if isinstance(project_id, str) and project_id:
        _state.project_id = project_id
