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

"""Checks tpu-host-device Ramble regexes against real benchmark output."""

import os
import re
import unittest
from unittest.mock import MagicMock

import yaml

from chs.tests.tpu_host_device import TpuHostDevicePlugin

_TEST_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src/chs/tests/tpu-host-device")
)

_SAMPLE_LOG = "\n".join(
    [
        "HOST node-1: standalone_host_init: set TPU_HOST_BOUNDS=1,1,1",
        "HOST node-1: JAX devices: 8 (e.g. [TpuDevice(id=0, ...)]...)",
        "HOST node-1: Running [1/1] host_to_device "
        "(tpu_sync_devall_size_4096mib)...",
        "HOST node-1: Success [1/1] host_to_device. Metrics: "
        "{'num_devices': 8, 'wall_clock_p50_ms': 167.4, "
        "'wall_clock_bandwidth_per_device_gb_s': 25.6351543, "
        "'xprof_bandwidth_per_host_gb_s': 205.0812345}",
        "HOST node-1: Running [1/1] device_to_host "
        "(tpu_sync_devall_size_4096mib)...",
        "HOST node-1: Success [1/1] device_to_host. Metrics: "
        "{'num_devices': 8, 'wall_clock_p50_ms': 629.1, "
        "'wall_clock_bandwidth_per_device_gb_s': np.float64(6.82), "
        "'xprof_bandwidth_per_host_gb_s': np.float64(5.457e+01)}",
        "HOST node-1: Summary saved to: /tmp/out/summary.csv",
    ]
)


def _load_application_directives():
    """Executes application.py with stubbed Ramble directives."""
    foms = {}
    contexts = {}

    def figure_of_merit(name, **kwargs):
        foms[name] = kwargs

    def figure_of_merit_context(name, **kwargs):
        contexts[name] = kwargs

    namespace = {
        "ExecutableApplication": object,
        "FomType": MagicMock(),
        "figure_of_merit": figure_of_merit,
        "figure_of_merit_context": figure_of_merit_context,
    }
    for directive in (
        "maintainers",
        "tags",
        "variable",
        "executable",
        "workload",
    ):
        namespace[directive] = MagicMock()

    with open(os.path.join(_TEST_DIR, "application.py")) as f:
        source = f.read().replace("from ramble.appkit import *", "")
    exec(compile(source, "application.py", "exec"), namespace)
    return foms, contexts


class TestTpuHostDeviceApplication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.foms, cls.contexts = _load_application_directives()

    def _match(self, regex):
        matches = [
            m
            for m in (
                re.match(regex, line) for line in _SAMPLE_LOG.splitlines()
            )
            if m
        ]
        self.assertEqual(1, len(matches), f"Expected one match for {regex}")
        return matches[0]

    def test_context_regex(self):
        ctx = self.contexts["params"]
        matches = [
            m
            for m in (
                re.match(ctx["regex"], line)
                for line in _SAMPLE_LOG.splitlines()
            )
            if m
        ]
        self.assertEqual(2, len(matches))
        for m in matches:
            self.assertEqual(
                "node-1,4096MiB", ctx["output_format"].format(**m.groupdict())
            )

    def test_fom_regexes(self):
        expected = {
            "H2D Bandwidth": "205.0812345",
            "D2H Bandwidth": "5.457e+01",
        }
        for name, value in expected.items():
            with self.subTest(fom=name):
                fom = self.foms[name]
                m = self._match(fom["fom_regex"])
                self.assertEqual(value, m.group(fom["group_name"]))
                self.assertEqual("GB/s", fom["units"])
        self.assertEqual(set(expected), set(self.foms))

    def test_threshold_keys_resolve_to_foms(self):
        aliases = TpuHostDevicePlugin().get_fom_aliases()
        with open(os.path.join(_TEST_DIR, "thresholds.yaml")) as f:
            thresholds = yaml.safe_load(f)
        thresholds.pop("type")
        for machine, metrics in thresholds.items():
            for key in metrics:
                with self.subTest(machine=machine, key=key):
                    self.assertIn(aliases[key], self.foms)

    def test_benchmark_configs_use_all_local_devices(self):
        tpu_sync_params = {
            "warmup_tries": 2,
            "num_runs": 20,
            "dtype": "float32",
            "transfer_backend": "tpu_sync",
        }
        for rel_path, expected_name, expected_params, expected_sweep in (
            (
                "host_to_device.yaml",
                "host_to_device",
                {
                    "warmup_tries": 2,
                    "num_runs": 20,
                    "dtype": "float32",
                    "data_size_mib": 4096,
                },
                None,
            ),
            (
                "tpu7x/host_to_device.yaml",
                "host_to_device",
                {**tpu_sync_params, "data_size_mib": 16384},
                None,
            ),
            (
                "tpuv6e/host_to_device.yaml",
                "host_to_device",
                {**tpu_sync_params, "data_size_mib": 8192},
                None,
            ),
            (
                "device_to_host.yaml",
                "device_to_host",
                {
                    "warmup_tries": 2,
                    "num_runs": 20,
                    "dtype": "float32",
                    "data_size_mib": 4096,
                },
                None,
            ),
            (
                "tpu7x/device_to_host.yaml",
                "device_to_host",
                {**tpu_sync_params, "data_size_mib": 16384},
                None,
            ),
            (
                "tpuv6e/device_to_host.yaml",
                "device_to_host",
                {**tpu_sync_params, "data_size_mib": 8192},
                None,
            ),
        ):
            with self.subTest(config=rel_path):
                with open(os.path.join(_TEST_DIR, "benchmarks", rel_path)) as f:
                    cfg = yaml.safe_load(f)["benchmark"]
                self.assertEqual(expected_name, cfg["name"])
                self.assertTrue(cfg["standalone_host_init"])
                self.assertEqual(expected_params, cfg["params"])
                if expected_sweep is None:
                    self.assertNotIn("sweep", cfg)
                else:
                    self.assertEqual(
                        {"data_size_mib": expected_sweep}, cfg["sweep"]
                    )


if __name__ == "__main__":
    unittest.main()
