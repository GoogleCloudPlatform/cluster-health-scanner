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
        "HOST node-1: ===Starting benchmark 'host_device'===",
        "HOST node-1: Running host_device with params: {"
        "'transfer_type': 'pinned_memory', 'num_devices': 8, "
        "'data_size_mib': 4096, 'input_type': 'jax', "
        "'dtype': <class 'jax.numpy.float32'>, 'num_runs': 20, "
        "'trace_dir': '../microbenchmarks/host_device/trace/benchmark_0'}",
        "HOST node-1: Benchmarking Transfer with Data Size: 4096 MB "
        "for 20 iterations with transfer_type='pinned_memory'",
        "HOST node-1:   Allocating Pinned Host Data...",
        "HOST node-1:   H2D_bw (GiB/s) median: 205.0812345, P95: 206.23",
        "HOST node-1:   D2H_bw (GiB/s) median: 5.457e+01, P95: 54.93",
        "HOST node-1: Metrics written to CSV at "
        "../microbenchmarks/host_device/t_host_device_ABC.tsv.",
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
        m = self._match(ctx["regex"])
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
                self.assertEqual("GiB/s", fom["units"])
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
        for rel_path, num_devices in (
            ("host_device.yaml", 8),
            ("tpu7x/host_device.yaml", 8),
            ("tpuv6e/host_device.yaml", 4),
        ):
            with self.subTest(config=rel_path):
                with open(os.path.join(_TEST_DIR, "benchmarks", rel_path)) as f:
                    cfg = yaml.safe_load(f)["benchmarks"][0]
                self.assertEqual("host_device", cfg["benchmark_name"])
                self.assertTrue(cfg["single_host_tpu_init"])
                self.assertEqual(20, cfg["num_runs"])
                self.assertEqual(
                    [
                        {
                            "transfer_type": "pinned_memory",
                            "num_devices_list": [num_devices],
                            "data_size_mib_list": [4096],
                            "input_type_list": ["jax"],
                            "dtype_list": ["float32"],
                        }
                    ],
                    cfg["benchmark_sweep_params"],
                )


if __name__ == "__main__":
    unittest.main()
