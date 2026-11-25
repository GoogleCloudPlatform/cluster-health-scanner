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

import os
import re
import sys
import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np
import yaml

_TEST_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "../src/chs/tests/tpu-compute-stress"
    )
)

sys.path.insert(0, _TEST_DIR)

sys.modules["jax"] = MagicMock()
sys.modules["jax.numpy"] = MagicMock()

import tpu_compute_stress

from chs.tests.tpu_compute_stress import TpuComputeStressPlugin


_MATCHING = [[1.0, 2.0], [1.0, 2.0]]
_MISMATCHING = [[1.0, 2.0], [1.0, 99.0]]
_LOW_PRECISION_DISPATCHES = 9
_VECTOR_DISPATCHES = 7


def _tpu_device(device_kind):
    device = MagicMock()
    device.device_kind = device_kind
    return device


def _logged_lines(mock_info, mock_error):
    calls = mock_info.call_args_list + mock_error.call_args_list
    return [f"2026-01-01 00:00:00,000 - INFO - {c.args[0]}" for c in calls]


def _run_main(
    node_name,
    matmul_results=_MATCHING,
    low_precision_results=_MATCHING,
    vector_results=_MATCHING,
    device_kind="TPU7x",
    low_precision_phase=None,
    vector_phase=None,
    experimental=True,
):
    low_precision_phase = low_precision_phase or MagicMock(
        return_value=(low_precision_results, _LOW_PRECISION_DISPATCHES)
    )
    vector_phase = vector_phase or MagicMock(
        return_value=(vector_results, _VECTOR_DISPATCHES)
    )
    environment = {
        "MY_NODE_NAME": node_name,
        "CHS_EXPERIMENTAL": "true" if experimental else "false",
    }
    with (
        patch.dict(os.environ, environment),
        patch("tpu_compute_stress.logging.info") as mock_info,
        patch("tpu_compute_stress.logging.error") as mock_error,
        patch(
            "tpu_compute_stress.time.time",
            side_effect=[10.0, 15.0, 20.0, 50.0, 60.0, 80.0],
        ),
        patch("tpu_compute_stress.jax.device_get", side_effect=lambda x: x),
        patch(
            "tpu_compute_stress.jax.local_devices",
            return_value=[_tpu_device(device_kind), _tpu_device(device_kind)],
        ),
        patch(
            "tpu_compute_stress.init_payloads",
            return_value=("A", "W1", "W2"),
        ),
        patch(
            "tpu_compute_stress.hardware_health_stress",
            return_value=matmul_results,
        ),
        patch(
            "tpu_compute_stress.hardware_low_precision_stress",
            low_precision_phase,
        ),
        patch("tpu_compute_stress.hardware_vector_stress", vector_phase),
    ):
        tpu_compute_stress.main()
    return mock_info, mock_error, low_precision_phase


def _assert_sdc_failure(test, mock_info, mock_error, node_name):
    mock_error.assert_any_call(
        f"NODE:{node_name} EVENT:STRESS_FAIL. SDC (Bit flips) Detected"
    )
    logged = [c.args[0] for c in mock_info.call_args_list]
    test.assertFalse(any("EVENT:MEASURE_END" in line for line in logged))
    test.assertNotIn(
        call(f"NODE:{node_name} EVENT:STRESS_PASS"), mock_info.call_args_list
    )


class TpuComputeStressMainTest(unittest.TestCase):
    def test_success_logs_pass(self):
        mock_info, mock_error, _ = _run_main("test-node")

        mock_info.assert_any_call("NODE:test-node EVENT:STRESS_PASS")
        mock_error.assert_not_called()

    def test_measure_end_reports_the_timed_loop(self):
        mock_info, _, _ = _run_main("test-node")

        flops = (
            2.0
            * tpu_compute_stress.MATRIX_DIM**3
            * 8
            * tpu_compute_stress.NUM_ITERATIONS_MEASURE
        )
        mock_info.assert_any_call(
            "NODE:test-node EVENT:MEASURE_END DURATION:5.00s "
            f"TFLOPs: {flops / 5.0 / 1e12:.2f}"
        )

    def test_every_phase_logs_where_it_starts_and_ends(self):
        mock_info, _, _ = _run_main("test-node")
        events = [
            event
            for c in mock_info.call_args_list
            for event in re.findall(r"EVENT:(\w+)", c.args[0])
        ]
        self.assertEqual(
            [
                "INIT_START",
                "WARMUP_START",
                "MEASURE_START",
                "LOW_PRECISION_START",
                "LOW_PRECISION_END",
                "VECTOR_START",
                "VECTOR_END",
                "MEASURE_END",
                "STRESS_PASS",
            ],
            events,
        )

    def test_matmul_mismatch_fails(self):
        mock_info, mock_error, _ = _run_main(
            "test-node-sdc", matmul_results=_MISMATCHING
        )

        _assert_sdc_failure(self, mock_info, mock_error, "test-node-sdc")

    def test_low_precision_mismatch_alone_fails(self):
        mock_info, mock_error, _ = _run_main(
            "test-node-low", low_precision_results=_MISMATCHING
        )

        _assert_sdc_failure(self, mock_info, mock_error, "test-node-low")

    def test_vector_mismatch_alone_fails(self):
        mock_info, mock_error, _ = _run_main(
            "test-node-vector", vector_results=_MISMATCHING
        )

        _assert_sdc_failure(self, mock_info, mock_error, "test-node-vector")

    def test_nan_on_every_device_fails(self):
        all_nan = [[np.nan, 1.0], [np.nan, 1.0]]
        mock_info, mock_error, _ = _run_main(
            "test-node-nan", vector_results=all_nan
        )

        _assert_sdc_failure(self, mock_info, mock_error, "test-node-nan")
        mock_error.assert_any_call(
            "NODE:test-node-nan EVENT:NON_FINITE_RESULTS PHASE:vector DEVICES:0,1"
        )

    def test_failure_names_every_failing_phase_and_device(self):
        _, mock_error, _ = _run_main(
            "test-node",
            low_precision_results=_MISMATCHING,
            vector_results=_MISMATCHING,
        )

        self.assertEqual(
            [
                "NODE:test-node EVENT:RESULTS_DIFFER_FROM_DEVICE_0 "
                "PHASE:low_precision DEVICES:1",
                "NODE:test-node EVENT:RESULTS_DIFFER_FROM_DEVICE_0 "
                "PHASE:vector DEVICES:1",
                "NODE:test-node EVENT:STRESS_FAIL. SDC (Bit flips) Detected",
            ],
            [c.args[0] for c in mock_error.call_args_list],
        )

    def test_low_precision_phase_reports_dispatches_and_duration(self):
        mock_info, _, _ = _run_main("test-node")

        mock_info.assert_any_call(
            "NODE:test-node EVENT:LOW_PRECISION_END "
            f"DISPATCHES:{_LOW_PRECISION_DISPATCHES} "
            "STEPS_PER_DISPATCH:"
            f"{tpu_compute_stress.LOW_PRECISION_STEPS_PER_DISPATCH} "
            "DURATION_WITH_COMPILE:30.00s"
        )

    def test_vector_phase_reports_dispatches_and_duration(self):
        mock_info, _, _ = _run_main("test-node")

        mock_info.assert_any_call(
            f"NODE:test-node EVENT:VECTOR_END DISPATCHES:{_VECTOR_DISPATCHES} "
            f"STEPS_PER_DISPATCH:{tpu_compute_stress.VECTOR_STEPS_PER_DISPATCH} "
            "DURATION_WITH_COMPILE:20.00s"
        )

    def test_vector_phase_runs_for_vector_seconds(self):
        vector_phase = MagicMock(return_value=(_MATCHING, 1))

        _run_main("test-node", vector_phase=vector_phase)

        vector_phase.assert_called_once_with(tpu_compute_stress.VECTOR_SECONDS)

    def test_low_precision_phase_uses_fp8_on_tpu7x(self):
        _, _, low_precision_phase = _run_main("test-node", device_kind="TPU7x")

        low_precision_phase.assert_called_once_with(
            "A",
            "W1",
            tpu_compute_stress.jnp.float8_e4m3fn,
            tpu_compute_stress.LOW_PRECISION_SECONDS,
        )

    def test_low_precision_phase_uses_int8_on_v6e(self):
        _, _, low_precision_phase = _run_main(
            "test-node", device_kind="TPU v6 lite"
        )

        low_precision_phase.assert_called_once_with(
            "A",
            "W1",
            tpu_compute_stress.jnp.int8,
            tpu_compute_stress.LOW_PRECISION_SECONDS,
        )


def _run_main_without_experimental(node_name, matmul_results=_MATCHING):
    low_precision_phase = MagicMock()
    vector_phase = MagicMock()
    mock_info, mock_error, _ = _run_main(
        node_name,
        matmul_results=matmul_results,
        low_precision_phase=low_precision_phase,
        vector_phase=vector_phase,
        experimental=False,
    )
    return mock_info, mock_error, low_precision_phase, vector_phase


def _logged_events(mock_info, mock_error):
    return [
        event
        for c in mock_info.call_args_list + mock_error.call_args_list
        for event in re.findall(r"EVENT:(\w+)", c.args[0])
    ]


class TpuComputeStressWithoutExperimentalTest(unittest.TestCase):
    def test_success_logs_measure_end_and_pass(self):
        mock_info, mock_error, _, _ = _run_main_without_experimental(
            "test-node"
        )

        self.assertEqual(
            [
                "INIT_START",
                "WARMUP_START",
                "MEASURE_START",
                "MEASURE_END",
                "STRESS_PASS",
            ],
            _logged_events(mock_info, mock_error),
        )
        mock_error.assert_not_called()

    def test_skips_low_precision_and_vector_phases(self):
        _, _, low_precision_phase, vector_phase = (
            _run_main_without_experimental("test-node")
        )

        low_precision_phase.assert_not_called()
        vector_phase.assert_not_called()

    def test_matmul_mismatch_fails_without_measure_end(self):
        mock_info, mock_error, _, _ = _run_main_without_experimental(
            "test-node-sdc", matmul_results=_MISMATCHING
        )

        mock_error.assert_called_once_with(
            "NODE:test-node-sdc EVENT:STRESS_FAIL. SDC (Bit flips) Detected"
        )
        self.assertEqual(
            ["INIT_START", "WARMUP_START", "MEASURE_START", "STRESS_FAIL"],
            _logged_events(mock_info, mock_error),
        )


class RunUntilDeadlineTest(unittest.TestCase):
    def _run(self, clock, seconds):
        dispatch = MagicMock(side_effect=lambda x: x + 1)
        with (
            patch("tpu_compute_stress.time.monotonic", side_effect=clock),
            patch(
                "tpu_compute_stress.jax.block_until_ready",
                side_effect=lambda x: x,
            ),
        ):
            result = tpu_compute_stress._run_until_deadline(
                seconds, dispatch, 0
            )
        return result, dispatch

    def test_dispatches_until_the_deadline_and_chains_the_state(self):
        (x, dispatches), dispatch = self._run([0.0, 0.0, 0.6, 1.2], 1)

        self.assertEqual((2, 2), (x, dispatches))
        self.assertEqual([call(0), call(1)], dispatch.call_args_list)

    def test_runs_one_dispatch_even_if_it_overruns_the_deadline(self):
        (x, dispatches), _ = self._run([0.0, 0.0, 5.0], 1)

        self.assertEqual((1, 1), (x, dispatches))


def _identity_device_get():
    return patch("tpu_compute_stress.jax.device_get", side_effect=lambda x: x)


class DevicesDifferingFromDevice0Test(unittest.TestCase):
    def _differing(self, per_device_results):
        with _identity_device_get():
            return tpu_compute_stress.devices_differing_from_device_0(
                per_device_results
            )

    def test_identical_results_differ_nowhere(self):
        self.assertEqual([], self._differing([np.array([1.5, 2.0])] * 3))

    def test_reports_every_device_that_differs(self):
        self.assertEqual(
            [1, 3],
            self._differing(
                [
                    np.array([1.5]),
                    np.array([2.5]),
                    np.array([1.5]),
                    np.array([9.0]),
                ]
            ),
        )

    def test_a_sign_flip_on_zero_differs(self):
        self.assertEqual(
            [1], self._differing([np.array([0.0]), np.array([-0.0])])
        )

    def test_bit_identical_nans_do_not_differ(self):
        self.assertEqual([], self._differing([np.array([np.nan])] * 2))


class DevicesWithNonFiniteValuesTest(unittest.TestCase):
    def _non_finite(self, per_device_results):
        with _identity_device_get():
            return tpu_compute_stress.devices_with_non_finite_values(
                per_device_results
            )

    def test_finite_floats_and_integers_pass(self):
        self.assertEqual(
            [],
            self._non_finite(
                [np.array([1.5, -2.0]), np.array([3, -4], np.int8)]
            ),
        )

    def test_reports_devices_with_nan_or_inf(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad):
                self.assertEqual(
                    [1], self._non_finite([np.array([1.0]), np.array([bad])])
                )


class MatmulDispatchTest(unittest.TestCase):
    def test_measured_loop_is_split_into_fixed_size_dispatches(self):
        with (
            patch("tpu_compute_stress._stress_device_loop") as loop,
            patch("tpu_compute_stress._per_device_results"),
        ):
            tpu_compute_stress.hardware_health_stress("A", "W1", "W2", 12)

        steps = tpu_compute_stress.MATMUL_STEPS_PER_DISPATCH
        self.assertEqual(
            [steps, steps, 12 - 2 * steps],
            [c.args[3] for c in loop.call_args_list],
        )

    def test_warmup_is_one_dispatch(self):
        self.assertEqual(
            tpu_compute_stress.MATMUL_STEPS_PER_DISPATCH,
            tpu_compute_stress.NUM_ITERATIONS_WARMUP,
        )


class LowPrecisionDtypeTest(unittest.TestCase):
    def test_fp8_on_tpu7x_device_kinds(self):
        for device_kind in ("TPU7x", "TPU v7x", "tpu7x"):
            with self.subTest(device_kind=device_kind):
                self.assertIs(
                    tpu_compute_stress.jnp.float8_e4m3fn,
                    tpu_compute_stress.low_precision_dtype(device_kind),
                )

    def test_int8_on_other_device_kinds(self):
        for device_kind in ("TPU v6 lite", "TPU v6e", "TPU v5 lite"):
            with self.subTest(device_kind=device_kind):
                self.assertIs(
                    tpu_compute_stress.jnp.int8,
                    tpu_compute_stress.low_precision_dtype(device_kind),
                )

    def test_int8_matmul_cannot_overflow_int32(self):
        int8_max = 127
        largest_sum = tpu_compute_stress.MATRIX_DIM * int8_max**2
        self.assertLess(largest_sum, 2**31 - 1)


def _load_application_directives():
    foms = {}
    contexts = {}
    executables = {}

    def figure_of_merit(name, **kwargs):
        foms[name] = kwargs

    def figure_of_merit_context(name, **kwargs):
        contexts[name] = kwargs

    def executable(name, **kwargs):
        executables[name] = kwargs

    namespace = {
        "ExecutableApplication": object,
        "figure_of_merit": figure_of_merit,
        "figure_of_merit_context": figure_of_merit_context,
        "executable": executable,
    }
    for directive in ("tags", "variable", "workload"):
        namespace[directive] = MagicMock()

    app_path = os.path.join(_TEST_DIR, "application.py")
    with open(app_path) as f:
        source = f.read().replace("from ramble.appkit import *", "")
    exec(compile(source, app_path, "exec"), namespace)
    return foms, contexts, executables


class TpuComputeStressApplicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.foms, cls.contexts, cls.executables = _load_application_directives()

    def test_script_runs_at_the_configured_max_p_state(self):
        (run_script,) = [
            line
            for line in self.executables["execute"]["template"]
            if "tpu_compute_stress.py" in line and "python" in line
        ]
        self.assertIn("--xla_tpu_dvfs_p_state={max_p_state}", run_script)
        self.assertTrue(run_script.startswith("LIBTPU_INIT_ARGS="))

    def test_every_platform_sets_its_max_p_state(self):
        with open(os.path.join(_TEST_DIR, "config.yaml")) as f:
            config = yaml.safe_load(f)
        self.assertEqual(
            {"tpu7x-standard-4t": 7, "ct6e-standard-4t": 3},
            {
                machine: config["machines"][machine]["variables"]["max_p_state"]
                for machine in config["platforms"]
            },
        )

    def _finished_run_lines(self, results_valid):
        low_precision_results = _MATCHING if results_valid else _MISMATCHING
        mock_info, mock_error, _ = _run_main(
            "node-1", low_precision_results=low_precision_results
        )
        return _logged_lines(mock_info, mock_error)

    def _matching(self, regex, lines):
        return [m for m in (re.match(regex, line) for line in lines) if m]

    def test_node_context_matches_the_measure_end_line(self):
        lines = self._finished_run_lines(results_valid=True)
        (match,) = self._matching(self.contexts["node"]["regex"], lines)
        self.assertIn("EVENT:MEASURE_END", match.string)
        self.assertEqual("node-1", match.group("node"))

    def test_performance_fom(self):
        fom = self.foms["Performance"]
        lines = self._finished_run_lines(results_valid=True)
        (match,) = self._matching(fom["fom_regex"], lines)
        self.assertGreater(float(match.group(fom["group_name"])), 0.0)
        self.assertEqual(["node"], fom["contexts"])

    def test_sdc_run_reports_no_performance(self):
        fom = self.foms["Performance"]
        lines = self._finished_run_lines(results_valid=False)
        self.assertEqual([], self._matching(fom["fom_regex"], lines))

    def test_performance_is_the_only_fom(self):
        self.assertEqual({"Performance"}, set(self.foms))

    def test_performance_floor_per_platform(self):
        with open(os.path.join(_TEST_DIR, "thresholds.yaml")) as f:
            thresholds = yaml.safe_load(f)
        thresholds.pop("type")
        self.assertEqual(
            {
                "tpu7x-standard-4t": {"performance": 1050.0},
                "ct6e-standard-4t": {"performance": 685.0},
            },
            thresholds,
        )

    def test_threshold_keys_resolve_to_foms(self):
        aliases = TpuComputeStressPlugin().get_fom_aliases()
        with open(os.path.join(_TEST_DIR, "thresholds.yaml")) as f:
            thresholds = yaml.safe_load(f)
        thresholds.pop("type")
        for machine, metrics in thresholds.items():
            for key in metrics:
                with self.subTest(machine=machine, key=key):
                    self.assertIn(aliases.get(key, key), self.foms)


if __name__ == "__main__":
    unittest.main()
