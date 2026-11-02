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

"""Unit tests for the CHS telemetry module.

The worker and exit tests run the real background thread against a fake
Clearcut server with a 50 ms batch window, so each takes well under a second.
"""

import contextlib
import dataclasses
import http.client
import json
import os
import queue
import threading
import time
import unittest
import urllib.error
import uuid
from typing import Any
from unittest import mock

from chs import telemetry
from chs.telemetry import telemetry_util
from chs.telemetry.supported_metrics import MetricName

# Upper bound for anything that should happen right away.
_TIMEOUT_S = 5.0
_WINDOW_S = 0.05
# Slack for float rounding when comparing time.monotonic() values.
_EPSILON_S = 0.001
# A reply from https://play.googleapis.com/staging/log?format=json_proto for
# CLUSTER_HEALTH_SCANNER, with qos_tier_configuration cut to one entry.
_REAL_REPLY = (
    b'{"next_request_wait_millis":"900000",'
    b'"qos_tier":{"qos_tier_configuration":'
    b'[{"log_source_name":"ANDROID_BACKUP","qos_tier":0}],'
    b'"qos_tier_fingerprint":"-3334737594024971225"},'
    b'"log_source_batching_denylist":{"log_source_name":[]},'
    b'"log_response_details":[],"product_id_zwieback_tokens":[],'
    b'"session_zwiebacks":[]}'
)


@dataclasses.dataclass
class _Request:
    """A request the fake Clearcut server received."""

    sent_at: float
    url: str
    headers: dict[str, str]
    timeout: float
    body: dict[str, Any]

    @property
    def metadata(self) -> dict[str, str]:
        """The metrics in the request's single event, by name."""
        (log_event,) = self.body["log_event"]
        event = json.loads(log_event["source_extension_json"])
        prefix = f"{telemetry._CLIENT_TYPE}_"
        return {
            entry["key"].removeprefix(prefix): entry["value"]
            for entry in event["event_metadata"]
        }


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        pass


class _FakeClearcut:
    """Stands in for urlopen: records requests and replies from a script.

    Each reply is a LogResponse dict, an exception to raise, or a callable that
    returns one of those, for acting while the request is in flight. Once the
    script runs out, every request gets an empty LogResponse.
    """

    def __init__(self):
        self.replies: list[Any] = []
        self.requests: queue.Queue[_Request] = queue.Queue()

    def __call__(self, request, timeout):
        self.requests.put(
            _Request(
                sent_at=time.monotonic(),
                url=request.full_url,
                headers=dict(request.header_items()),
                timeout=timeout,
                body=json.loads(request.data),
            )
        )
        reply = self.replies.pop(0) if self.replies else {}
        if callable(reply):
            reply = reply()
        if isinstance(reply, Exception):
            raise reply
        return _FakeResponse(json.dumps(reply).encode())


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        telemetry_util._DEFAULT_ENDPOINT,
        code,
        http.HTTPStatus(code).phrase,
        None,
        None,
    )


class _TelemetryTestCase(unittest.TestCase):
    """Gives each test fresh telemetry state and a fake Clearcut server."""

    def setUp(self):
        super().setUp()
        fresh_state = {
            # A new _State per test, so tests can set its fields directly.
            "_state": telemetry._State(enabled=True),
            "_queue": queue.Queue(maxsize=telemetry._MAX_QUEUE_SIZE),
            "_exit_requested": threading.Event(),
            "_exit_done": threading.Event(),
            "_WINDOW_S": _WINDOW_S,
            "_EXIT_TIMEOUT_S": 1.0,
            "_RETRY_DELAY_S": 0.0,
        }
        for name, value in fresh_state.items():
            self._patch(name, value)
        self.clearcut = _FakeClearcut()
        self.enterContext(
            mock.patch.object(
                telemetry_util.urllib.request, "urlopen", self.clearcut
            )
        )
        self.mock_atexit_register = self.enterContext(
            mock.patch.object(telemetry.atexit, "register")
        )
        # Keeps results independent of the credentials on the test machine.
        self.enterContext(
            mock.patch.object(telemetry, "is_googler", return_value=False)
        )
        # Cleanups run in reverse, so this runs before the patches are undone.
        self.addCleanup(self._stop_worker)

    def _patch(self, name: str, value: Any) -> None:
        """Sets a telemetry module global for the rest of the test."""
        self.enterContext(mock.patch.object(telemetry, name, value))

    def _stop_worker(self) -> None:
        worker = telemetry._state.worker_thread
        if worker is None:
            return
        telemetry._state.enabled = False
        with contextlib.suppress(queue.Full):
            telemetry._queue.put_nowait(object())
        worker.join(_TIMEOUT_S)
        self.assertFalse(worker.is_alive(), "The telemetry worker didn't stop.")

    def _hold_worker(self):
        """Keeps send_metric from starting the worker; metrics stay queued."""
        return mock.patch.object(telemetry, "_start_worker", return_value=True)

    def _start_worker_with(self, *metrics: tuple[str, Any]) -> None:
        """Queues metrics, then starts the worker, so they share one window."""
        with self._hold_worker():
            for name, value in metrics:
                telemetry.send_metric(name, value)
        self.assertTrue(telemetry._start_worker())

    def _next_request(self) -> _Request:
        try:
            return self.clearcut.requests.get(timeout=_TIMEOUT_S)
        except queue.Empty:
            self.fail("No telemetry request was sent.")

    def _assert_no_request(self) -> None:
        with self.assertRaises(queue.Empty, msg="Unexpected request."):
            self.clearcut.requests.get(timeout=3 * _WINDOW_S)


class SendMetricTest(_TelemetryTestCase):
    """Tests for enable() and send_metric()."""

    def test_does_nothing_until_enabled(self):
        telemetry._state.enabled = False

        telemetry.send_metric("VERSION", "1.2.3")

        self.assertIsNone(telemetry._state.worker_thread)
        self.assertTrue(telemetry._queue.empty())
        self.mock_atexit_register.assert_not_called()

    def test_enable_turns_telemetry_on(self):
        telemetry._state.enabled = False

        telemetry.enable()

        self.assertTrue(telemetry._state.enabled)

    def test_first_call_starts_worker_and_registers_exit_flush(self):
        telemetry.send_metric("VERSION", "1.2.3")

        worker = telemetry._state.worker_thread
        self.assertIsNotNone(worker)
        self.assertEqual(worker.name, "chs-telemetry")
        self.assertTrue(worker.daemon)
        self.assertTrue(worker.is_alive())
        self.mock_atexit_register.assert_called_once_with(telemetry._on_exit)

    def test_concurrent_first_calls_start_one_worker(self):
        num_callers = 20
        barrier = threading.Barrier(num_callers)

        def call_send_metric():
            barrier.wait()
            telemetry.send_metric("VERSION", "1.2.3")

        # Create the callers before patching so only worker threads count.
        callers = [
            threading.Thread(target=call_send_metric)
            for _ in range(num_callers)
        ]
        with mock.patch.object(
            threading, "Thread", wraps=threading.Thread
        ) as mock_thread:
            for caller in callers:
                caller.start()
            for caller in callers:
                caller.join()

        mock_thread.assert_called_once()
        self.mock_atexit_register.assert_called_once()

    def test_failed_worker_start_is_retried_by_the_next_metric(self):
        """Each failed start is logged and drops its metric."""
        error = RuntimeError("can't start new thread")
        # Not assertLogs: other test modules leave logging.disable() set.
        with mock.patch.object(telemetry.logger, "debug") as mock_debug:
            with mock.patch.object(
                threading.Thread, "start", autospec=True, side_effect=error
            ) as mock_start:
                telemetry.send_metric("VERSION", "1.2.3")
                telemetry.send_metric("VERSION", "1.2.3")

        self.assertEqual(mock_start.call_count, 2)
        self.assertEqual(
            mock_debug.call_args_list,
            [mock.call("Failed to start telemetry thread: %s", error)] * 2,
        )
        self.assertIsNone(telemetry._state.worker_thread)
        self.assertTrue(telemetry._queue.empty())
        self.mock_atexit_register.assert_not_called()

        telemetry.send_metric("COMMAND", "run")

        self.assertTrue(telemetry._state.worker_thread.is_alive())
        self.mock_atexit_register.assert_called_once_with(telemetry._on_exit)
        self.assertEqual(self._next_request().metadata, {"COMMAND": "run"})

    def test_values_are_sent_as_strings(self):
        with self._hold_worker():
            for value in ({"passed": 3}, ["a", "b"], 5, "text"):
                telemetry.send_metric("RESULT", value)

        values = [telemetry._queue.get_nowait().value for _ in range(4)]
        self.assertEqual(values, ['{"passed":3}', '["a","b"]', "5", "text"])

    def test_values_that_cannot_be_converted_are_dropped(self):
        class Unprintable:
            def __str__(self):
                raise RuntimeError("boom")

        with self._hold_worker():
            telemetry.send_metric("RESULT", {"not_json": object()})
            telemetry.send_metric("RESULT", Unprintable())

        self.assertTrue(telemetry._queue.empty())

    def test_full_queue_drops_metrics(self):
        with self._hold_worker():
            for i in range(telemetry._MAX_QUEUE_SIZE + 5):
                telemetry.send_metric("VERSION", i)

        self.assertEqual(telemetry._queue.qsize(), telemetry._MAX_QUEUE_SIZE)

    def test_enqueue_errors_are_swallowed(self):
        """Telemetry must never break the CLI."""
        with mock.patch.object(
            telemetry._queue, "put_nowait", side_effect=ValueError("boom")
        ):
            telemetry.send_metric("VERSION", "1.2.3")


class BuildRequestTest(_TelemetryTestCase):
    """Tests for the Clearcut LogRequest built from a batch."""

    def test_batch_becomes_one_concord_event(self):
        body = json.loads(
            telemetry._build_request(
                [
                    telemetry._Metric("VERSION", "1.2.3", time_ms=2000),
                    telemetry._Metric("COMMAND", "run", time_ms=1000),
                ]
            )
        )

        self.assertEqual(
            body["client_info"], {"client_type": "CLUSTER_HEALTH_SCANNER"}
        )
        self.assertEqual(body["log_source_name"], "CONCORD")
        self.assertIsInstance(body["request_time_ms"], int)
        (log_event,) = body["log_event"]
        # The event takes the time of the batch's earliest metric.
        self.assertEqual(log_event["event_time_ms"], 1000)
        self.assertEqual(
            json.loads(log_event["source_extension_json"]),
            {
                "console_type": "CLUSTER_HEALTH_SCANNER",
                "event_type": "chsCLI",
                "release_version": telemetry.get_version(),
                "client_session_id": telemetry._SESSION_ID,
                "is_googler": False,
                "event_metadata": [
                    {"key": "CLUSTER_HEALTH_SCANNER_VERSION", "value": "1.2.3"},
                    {"key": "CLUSTER_HEALTH_SCANNER_COMMAND", "value": "run"},
                ],
            },
        )

    def test_run_id_is_added_once_set(self):
        telemetry._state.run_id = "20260926-155531-a3f9"

        body = json.loads(
            telemetry._build_request([telemetry._Metric("VERSION", "1.2.3", 0)])
        )

        event = json.loads(body["log_event"][0]["source_extension_json"])
        self.assertEqual(
            event["event_metadata"][-1],
            {
                "key": "CLUSTER_HEALTH_SCANNER_RUN_ID",
                "value": "20260926-155531-a3f9",
            },
        )

    def test_project_id_is_a_top_level_field_once_set(self):
        telemetry._state.project_id = "my-project"

        body = json.loads(
            telemetry._build_request([telemetry._Metric("VERSION", "1.2.3", 0)])
        )

        event = json.loads(body["log_event"][0]["source_extension_json"])
        self.assertEqual(event["project_id"], "my-project")


class PostTest(_TelemetryTestCase):
    """Tests for sending one request and classifying the result."""

    def test_success(self):
        retry = telemetry_util.post(b'{"log_event": []}', 2.5)

        self.assertFalse(retry)
        request = self._next_request()
        self.assertEqual(request.url, telemetry_util._DEFAULT_ENDPOINT)
        self.assertEqual(request.headers["Content-type"], "application/json")
        self.assertEqual(request.body, {"log_event": []})
        self.assertEqual(request.timeout, 2.5)

    def test_endpoint_can_be_overridden(self):
        staging = "https://play.googleapis.com/staging/log?format=json_proto"

        with mock.patch.dict(os.environ, {"CHS_TELEMETRY_ENDPOINT": staging}):
            telemetry_util.post(b"{}", 1.0)

        self.assertEqual(self._next_request().url, staging)

    def test_http_errors_are_retried_or_dropped(self):
        for error, retry in (
            (_http_error(500), True),
            (_http_error(503), True),
            (_http_error(400), False),
            (_http_error(429), False),
        ):
            with self.subTest(error=error):
                self.clearcut.replies = [error]

                self.assertIs(telemetry_util.post(b"{}", 1.0), retry)

    def test_network_errors_are_raised(self):
        for error in (
            urllib.error.URLError("no DNS"),
            TimeoutError("timed out"),
            http.client.RemoteDisconnected("closed"),
        ):
            with self.subTest(error=error):
                self.clearcut.replies = [error]

                with self.assertRaises(type(error)):
                    telemetry_util.post(b"{}", 1.0)


class WorkerTest(_TelemetryTestCase):
    """Tests for batching and retries in the worker thread."""

    def test_metrics_in_one_window_go_out_in_one_request(self):
        started_at = time.monotonic()
        self._start_worker_with(
            ("VERSION", "1.2.3"), ("COMMAND", "run"), ("VERSION", "1.2.4")
        )

        request = self._next_request()

        # One value per name; the newer one wins.
        self.assertEqual(
            request.metadata, {"VERSION": "1.2.4", "COMMAND": "run"}
        )
        self.assertGreaterEqual(
            request.sent_at - started_at, _WINDOW_S - _EPSILON_S
        )
        self.assertEqual(request.timeout, telemetry._HTTP_TIMEOUT_S)
        self._assert_no_request()

    def test_metric_after_a_send_opens_a_new_window(self):
        self._start_worker_with(("VERSION", "1.2.3"))
        first = self._next_request()

        telemetry.send_metric("COMMAND", "run")
        second = self._next_request()

        self.assertEqual(first.metadata, {"VERSION": "1.2.3"})
        self.assertEqual(second.metadata, {"COMMAND": "run"})
        self.assertGreaterEqual(
            second.sent_at - first.sent_at, _WINDOW_S - _EPSILON_S
        )

    def test_ignores_next_request_wait_millis(self):
        # Clearcut asks for 15 minutes; CHS waits only for the next window.
        self.clearcut.replies = [json.loads(_REAL_REPLY)]
        self._start_worker_with(("VERSION", "1.2.3"))
        self._next_request()

        telemetry.send_metric("COMMAND", "run")

        self.assertEqual(self._next_request().metadata, {"COMMAND": "run"})

    def test_server_errors_are_retried(self):
        self.clearcut.replies = [_http_error(503)]
        batch = {"VERSION": telemetry._Metric("VERSION", "1.2.3", 0)}

        telemetry._send(batch)

        # The retry happens within the call, and it succeeds, so there's no
        # third attempt.
        attempts = [self._next_request().metadata for _ in range(2)]
        self.assertEqual(attempts, [{"VERSION": "1.2.3"}] * 2)
        self.assertTrue(self.clearcut.requests.empty())
        self.assertEqual(batch, {})

    def test_retries_back_off_exponentially(self):
        self._patch("_RETRY_DELAY_S", 1.0)
        exit_requested = mock.Mock(spec=threading.Event)
        exit_requested.wait.return_value = False
        self._patch("_exit_requested", exit_requested)
        self.clearcut.replies = [_http_error(503)] * telemetry._MAX_ATTEMPTS
        batch = {"VERSION": telemetry._Metric("VERSION", "1.2.3", 0)}

        telemetry._send(batch)

        self.assertEqual(
            exit_requested.wait.call_args_list, [mock.call(1.0), mock.call(2.0)]
        )
        self.assertEqual(batch, {})

    def test_exit_during_backoff_leaves_the_batch_for_the_exit_flush(self):
        self._patch("_RETRY_DELAY_S", 60.0)
        self.clearcut.replies = [_http_error(503)]
        telemetry._exit_requested.set()
        batch = {"VERSION": telemetry._Metric("VERSION", "1.2.3", 0)}

        telemetry._send(batch)

        self._next_request()
        self.assertTrue(self.clearcut.requests.empty())
        self.assertEqual(list(batch), ["VERSION"])

    def test_client_errors_drop_the_batch(self):
        self.clearcut.replies = [_http_error(429)]
        self._start_worker_with(("VERSION", "1.2.3"))
        self._next_request()

        telemetry.send_metric("COMMAND", "run")

        self.assertEqual(self._next_request().metadata, {"COMMAND": "run"})

    def test_batch_is_dropped_after_max_attempts(self):
        self.clearcut.replies = [_http_error(503)] * telemetry._MAX_ATTEMPTS
        self._start_worker_with(("VERSION", "1.2.3"))
        for _ in range(telemetry._MAX_ATTEMPTS):
            self._next_request()

        telemetry.send_metric("COMMAND", "run")

        # No more attempts, and telemetry keeps going.
        self.assertEqual(self._next_request().metadata, {"COMMAND": "run"})

    def test_malformed_endpoint_is_not_retried(self):
        self.enterContext(
            mock.patch.dict(os.environ, {"CHS_TELEMETRY_ENDPOINT": "nope"})
        )
        post = self.enterContext(
            mock.patch.object(telemetry, "post", wraps=telemetry.post)
        )
        self._start_worker_with(("VERSION", "1.2.3"))

        telemetry._state.worker_thread.join(_TIMEOUT_S)

        # Retrying can't fix the URL, so the first attempt stops telemetry.
        self.assertFalse(telemetry._state.enabled)
        post.assert_called_once()

    def test_network_error_stops_telemetry(self):
        self.clearcut.replies = [urllib.error.URLError("timed out")]
        self._start_worker_with(("VERSION", "1.2.3"))
        self._next_request()

        telemetry._state.worker_thread.join(_TIMEOUT_S)
        telemetry.send_metric("COMMAND", "run")
        telemetry._on_exit()

        self.assertFalse(telemetry._state.enabled)
        self.assertTrue(telemetry._exit_done.is_set())
        self._assert_no_request()

    def test_worker_crash_stops_telemetry(self):
        self._patch(
            "_build_request", mock.Mock(side_effect=RuntimeError("bug"))
        )
        self._start_worker_with(("VERSION", "1.2.3"))

        telemetry._state.worker_thread.join(_TIMEOUT_S)

        self.assertFalse(telemetry._state.worker_thread.is_alive())
        self.assertFalse(telemetry._state.enabled)
        self.assertTrue(telemetry._exit_done.is_set())
        self.assertTrue(self.clearcut.requests.empty())


class ExitTest(_TelemetryTestCase):
    """Tests for the final flush at exit."""

    def test_idle_worker_exits_at_once_without_sending(self):
        telemetry.send_metric("VERSION", "1.2.3")
        self._next_request()

        started_at = time.monotonic()
        telemetry._on_exit()

        # Nothing was pending, so the worker exits as soon as it wakes.
        self.assertTrue(telemetry._exit_done.is_set())
        self.assertLess(
            time.monotonic() - started_at, telemetry._EXIT_TIMEOUT_S
        )
        self._assert_no_request()

    def test_sends_pending_metrics_without_waiting_for_the_window(self):
        self._patch("_WINDOW_S", 60.0)
        telemetry.send_metric("VERSION", "1.2.3")
        telemetry.send_metric("COMMAND", "run")

        telemetry._on_exit()

        self.assertTrue(telemetry._exit_done.is_set())
        request = self._next_request()
        self.assertEqual(
            request.metadata, {"VERSION": "1.2.3", "COMMAND": "run"}
        )
        self.assertEqual(request.timeout, telemetry._EXIT_TIMEOUT_S)

    def test_waits_for_a_request_already_in_flight(self):
        exit_requested = telemetry._exit_requested
        # Unblocks the fake server if the test fails before exit starts.
        self.addCleanup(exit_requested.set)
        replied = threading.Event()

        def reply_slowly_after_exit_starts():
            exit_requested.wait(_TIMEOUT_S)
            time.sleep(0.1)
            replied.set()
            return {}

        self.clearcut.replies = [reply_slowly_after_exit_starts]
        telemetry.send_metric("VERSION", "1.2.3")
        self._next_request()

        telemetry._on_exit()

        self.assertTrue(replied.is_set())
        # The batch already went out, so exit sends nothing more.
        self.assertTrue(self.clearcut.requests.empty())

    def test_sends_metrics_queued_behind_a_request_in_flight(self):
        exit_requested = telemetry._exit_requested
        # Unblocks the fake server if the test fails before exit starts.
        self.addCleanup(exit_requested.set)

        def reply_after_exit_starts():
            exit_requested.wait(_TIMEOUT_S)
            return {}

        self.clearcut.replies = [reply_after_exit_starts]
        telemetry.send_metric("VERSION", "1.2.3")
        self._next_request()
        telemetry.send_metric("COMMAND", "run")
        telemetry.send_metric("RESULT", "passed")

        telemetry._on_exit()

        # The worker sees the exit request as soon as it takes COMMAND, so
        # RESULT goes out only because the exit flush drains the queue.
        self.assertTrue(telemetry._exit_done.is_set())
        self.assertEqual(
            self._next_request().metadata,
            {"COMMAND": "run", "RESULT": "passed"},
        )

    def test_gives_up_on_a_hanging_server_within_the_timeout(self):
        self._patch("_WINDOW_S", 60.0)
        self._patch("_EXIT_TIMEOUT_S", 0.3)
        release = threading.Event()
        self.addCleanup(release.set)

        def hang():
            release.wait(_TIMEOUT_S)
            return TimeoutError("timed out")

        self.clearcut.replies = [hang]
        telemetry.send_metric("VERSION", "1.2.3")

        started_at = time.monotonic()
        telemetry._on_exit()

        self.assertLess(time.monotonic() - started_at, 0.3 + 1.0)
        # The worker is still stuck in the request.
        self.assertFalse(telemetry._exit_done.is_set())

    def test_skipped_once_the_worker_has_stopped(self):
        telemetry._exit_done.set()

        telemetry._on_exit()

        self.assertFalse(telemetry._exit_requested.is_set())

    def test_second_ctrl_c_skips_the_wait(self):
        exit_done = mock.Mock(spec=threading.Event)
        exit_done.is_set.return_value = False
        exit_done.wait.side_effect = KeyboardInterrupt
        self._patch("_exit_done", exit_done)

        telemetry._on_exit()  # Doesn't raise.

        exit_done.wait.assert_called_once_with(telemetry._EXIT_TIMEOUT_S)


class GroupingIdTest(_TelemetryTestCase):
    """Tests for the session and run IDs attached to telemetry events."""

    def test_session_id_is_random_uuid(self):
        self.assertEqual(uuid.UUID(telemetry._SESSION_ID).version, 4)

    def test_set_telemetry_run_id_keeps_only_timestamp_and_suffix(self):
        for workspace_name, expected in (
            ("20260926-155531_mycluster-lite-a3f9", "20260926-155531-a3f9"),
            # Custom --run-id values can contain '_' and '-'.
            ("20260926-155531_acme_prod-run-k2m7", "20260926-155531-k2m7"),
        ):
            with self.subTest(workspace_name=workspace_name):
                telemetry.set_telemetry_run_id(workspace_name)

                self.assertEqual(telemetry._state.run_id, expected)

    def test_set_telemetry_run_id_ignores_other_values(self):
        """Unknown formats are dropped, never sent as-is."""
        for workspace_name in (
            "mycluster-lite-a3f9",  # No timestamp.
            "20260926-155531_mycluster",  # No random suffix.
            "2026-09-26_mycluster-lite-a3f9",  # Unknown timestamp format.
            "",
            None,
            mock.MagicMock(),
        ):
            with self.subTest(workspace_name=workspace_name):
                telemetry.set_telemetry_run_id(workspace_name)

                self.assertIsNone(telemetry._state.run_id)

    def test_set_telemetry_project_id(self):
        for project_id in (None, "", mock.MagicMock()):
            with self.subTest(project_id=project_id):
                telemetry.set_telemetry_project_id(project_id)

                self.assertIsNone(telemetry._state.project_id)

        telemetry.set_telemetry_project_id("my-project")

        self.assertEqual(telemetry._state.project_id, "my-project")


class MetricNameTest(unittest.TestCase):
    def test_formats_as_value(self):
        for name in MetricName:
            with self.subTest(name=name.name):
                self.assertEqual(str(name), name.value)
                self.assertEqual(f"{name}", name.value)
                self.assertEqual(json.dumps(name), f'"{name.value}"')


if __name__ == "__main__":
    unittest.main()
