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

"""Unit tests for the tpu-gemm-pulse test package."""

import json
import os
import re
import tempfile
import unittest
from unittest.mock import MagicMock

from rich.console import Console
import yaml

from chs.boh_generator import BOHGenerator
from chs.protos import bill_of_health_pb2 as pb
from chs.tests.tpu_gemm_pulse import TpuGemmPulsePlugin
from chs.tests.tpu_gemm_pulse.extractor import TpuGemmPulseExtractor
from chs.tests.tpu_gemm_pulse.renderer import TpuGemmPulseRenderer

os.environ["TERM"] = "xterm-256color"

_TEST_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src/chs/tests/tpu-gemm-pulse")
)
_HBM_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src/chs/tests/tpu-hbm")
)

_SAMPLE_LOG = "\n".join(
    [
        "HOST node-1: standalone_host_init: set TPU_HOST_BOUNDS=1,1,1",
        "HOST node-1: JAX devices: 8 (e.g. [TpuDevice(id=0, ...)]...)",
        "HOST node-1: Running [1/1] gemm_pulse "
        "(m_16384_k_16384_bfloat16_to_bfloat16_chain_100_cycles_100_"
        "idle_0.02_dur_0.0_checksum_True)...",
        "HOST node-1: PulseRunExecutor wall-clock ms/iter: "
        "pre_soaking_window=874.2400, soak_p50=874.0000 (98 windows), "
        "post_soaking_window=874.2400",
        "HOST node-1: Success [1/1] gemm_pulse. Metrics: "
        "{'total_flops': 879609302220800.0, "
        "'xprof_p50_ms': np.float64(874.24), "
        "'xprof_tflops_per_device': np.float64(1006.18), "
        "'xprof_tflops_per_chip': 2012.36, "
        "'xprof_slowdown_ratio': np.float64(1.0002), "
        "'xprof_compute_roofline_efficiency_pct': 87.23, "
        "'checksum': '0x00a56eadd35cc7fe', "
        "'all_devices_valid_checksum': True, "
        "'intra_host_checksum_match': True, "
        "'global_checksum_match': True}",
        "HOST node-1: Summary saved to: /tmp/out/summary.csv",
    ]
)


def _load_application_directives(test_dir: str):
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

    with open(os.path.join(test_dir, "application.py")) as f:
        source = f.read().replace("from ramble.appkit import *", "")
    exec(compile(source, "application.py", "exec"), namespace)
    return foms, contexts


class TestTpuGemmPulseApplication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.foms, cls.contexts = _load_application_directives(_TEST_DIR)

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
            "node-1,16384x16384x16384,bfloat16,chain=100,cycles=100",
            ctx["output_format"].format(**m.groupdict()),
        )

    def test_fom_regexes(self):
        expected = {
            "Throughput": ("1006.18", "TFLOP/s/device"),
            "Valid Checksum": ("True", ""),
            "Checksum Match": ("True", ""),
            "Throughput per chip": ("2012.36", "TFLOP/s/chip"),
            "Slowdown Ratio": ("1.0002", ""),
            "MFU": ("87.23", "%"),
            "Step Time": ("874.24", "ms"),
        }
        for name, (value, units) in expected.items():
            with self.subTest(fom=name):
                fom = self.foms[name]
                m = self._match(fom["fom_regex"])
                self.assertEqual(value, m.group(fom["group_name"]))
                self.assertEqual(units, fom["units"])
        self.assertEqual(set(expected), set(self.foms))

    def test_threshold_keys_resolve_to_foms(self):
        aliases = TpuGemmPulsePlugin().get_fom_aliases()
        with open(os.path.join(_TEST_DIR, "thresholds.yaml")) as f:
            thresholds = yaml.safe_load(f)
        thresholds.pop("type")
        for machine, metrics in thresholds.items():
            for key in metrics:
                with self.subTest(machine=machine, key=key):
                    resolved = aliases.get(key, key)
                    self.assertIn(resolved, self.foms)

    def test_benchmark_configs(self):
        base_params = {
            "warmup_tries": 2,
            "num_cycles": 100,
            "idle_s": 0.02,
            "in_dtype": "bfloat16",
            "out_dtype": "bfloat16",
            "chain_length": 100,
            "enable_checksum": True,
            "seed": 0,
        }
        expected_shapes = {
            "gemm_pulse_16384_bf16.yaml": (16384, 16384),
            "tpu7x/gemm_pulse_16384_bf16.yaml": (16384, 16384),
            "tpuv6e/gemm_pulse_32768x4096x4096_bf16.yaml": (32768, 4096),
        }
        for rel_path, (m_dim, k_dim) in expected_shapes.items():
            with self.subTest(config=rel_path):
                with open(os.path.join(_TEST_DIR, "benchmarks", rel_path)) as f:
                    cfg = yaml.safe_load(f)["benchmark"]
                self.assertEqual("gemm_pulse", cfg["name"])
                self.assertTrue(cfg["standalone_host_init"])
                self.assertTrue(cfg["xprof_timing"])
                self.assertEqual(
                    "local_slowest_device",
                    cfg["xprof_device_timing_aggregation"],
                )
                expected_params = {**base_params, "m": m_dim, "k": k_dim}
                self.assertEqual(expected_params, cfg["params"])

    def test_hbm_context_regex_supports_scope_and_legacy_dev_ids(self):
        _, hbm_contexts = _load_application_directives(_HBM_DIR)
        ctx = hbm_contexts["node_params"]
        for run_id in (
            "copy_dim_2147483648_scope_device_idx_0",
            "copy_dim_2147483648_dev_0",
        ):
            line = f"HOST node-1 (bfloat16): Running [12/13] hbm ({run_id})..."
            m = re.match(ctx["regex"], line)
            self.assertIsNotNone(m, f"Did not match {line}")
            self.assertEqual(
                "node-1,2147483648,bfloat16",
                ctx["output_format"].format(**m.groupdict()),
            )


class TestTpuGemmPulsePluginAndBoH(unittest.TestCase):
    def test_extractor_parses_context_metadata(self):
        meta = TpuGemmPulseExtractor.parse_context_metadata(
            "node-1,32768x4096x4096,bfloat16,chain=100,cycles=100"
        )
        self.assertEqual("node-1", meta["node"])
        self.assertEqual("32768x4096x4096", meta["matrix_shape"])
        self.assertEqual("bfloat16", meta["dtype"])
        self.assertEqual("chain=100", meta["chain_length"])
        self.assertEqual("cycles=100", meta["num_cycles"])
        self.assertEqual(
            "32768x4096x4096,bfloat16,chain=100,cycles=100", meta["sweep_param"]
        )

    def test_renderer_detail_output(self):
        renderer = TpuGemmPulseRenderer()
        self.assertIsNone(renderer.get_detail_data([]))

        res = pb.TestResult(test_name="tpu-gemm-pulse", test_level="check")
        res.target_resources.add(node_name="node-1")
        ctx = "node-1,32768x4096x4096,bfloat16,chain=100,cycles=100"

        for key, num_val in (
            ("Throughput", 1006.18),
            ("Throughput per chip", 2012.36),
            ("Slowdown Ratio", 1.0),
            ("MFU", 87.23),
            ("Step Time", 109.28),
        ):
            m = res.metrics.add(key=key)
            m.value.number_value = num_val
            m.metadata.fields["context"].string_value = ctx

        for key in ("Valid Checksum", "Checksum Match"):
            m = res.metrics.add(key=key)
            m.value.bool_value = True
            m.metadata.fields["context"].string_value = ctx

        console = Console(width=220)
        with console.capture() as capture:
            console.print(renderer.render_detail([res]))

        output = capture.get()
        self.assertIn("TPU GEMM Pulse Summary", output)
        self.assertIn("32768x4096x4096,bfloat16,chain=100,cycles=100", output)
        self.assertIn("node-1", output)
        self.assertIn("1006.18", output)
        self.assertIn("2012.36", output)
        self.assertIn("1.00", output)
        self.assertIn("87.2%", output)
        self.assertIn("109.28", output)
        self.assertIn("True", output)

    def test_renderer_skips_metrics_without_pulse_context(self):
        renderer = TpuGemmPulseRenderer()
        res = pb.TestResult(test_name="tpu-gemm-pulse", test_level="check")
        ctx = "node-1,32768x4096x4096,bfloat16,chain=100,cycles=100"

        m = res.metrics.add(key="Throughput")
        m.value.number_value = 1006.18
        m.metadata.fields["context"].string_value = ctx

        # Error-detector style findings with no / unparseable context.
        m = res.metrics.add(key="TPU Error")
        m.value.string_value = "CHECKSUM MISMATCH"
        m = res.metrics.add(key="TPU Error")
        m.value.string_value = "Degenerate"
        m.metadata.fields["context"].string_value = "null"

        report = renderer.get_detail_data([res])
        rows = report.tables[0].rows
        self.assertEqual(1, len(rows))
        self.assertEqual(
            "32768x4096x4096,bfloat16,chain=100,cycles=100",
            rows[0]["sweep_param"],
        )
        self.assertEqual("node-1", rows[0]["nodes"])
        self.assertNotIn("unknown", [r["sweep_param"] for r in rows])

    def _generate_boh(
        self,
        throughput: float = 1006.18,
        valid_checksum: str = "True",
        checksum_match: str = "True",
        machine_type: str = "tpu7x-standard-4t",
    ) -> pb.TestResult:
        ramble_results = {
            "workspace_name": "test-pulse-run",
            "experiments": [
                {
                    "experiment_name": "chs-tpu-gemm-pulse",
                    "application_name": "tpu-gemm-pulse",
                    "workload_name": "check",
                    "RAMBLE_VARIABLES": {
                        "chs_test_name": "tpu-gemm-pulse",
                        "machine_type": machine_type,
                        "node": "node-1",
                    },
                    "RAMBLE_RAW_VARIABLES": {},
                    "CONTEXTS": [
                        {
                            "name": (
                                "node-1,16384x16384x16384,bfloat16,"
                                "chain=100,cycles=100"
                            ),
                            "foms": [
                                {
                                    "name": "Throughput",
                                    "value": throughput,
                                    "units": "TFLOP/s/device",
                                },
                                {
                                    "name": "Valid Checksum",
                                    "value": valid_checksum,
                                    "units": "",
                                },
                                {
                                    "name": "Checksum Match",
                                    "value": checksum_match,
                                    "units": "",
                                },
                            ],
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = os.path.join(tmpdir, "results.json")
            with open(results_path, "w") as f:
                json.dump(ramble_results, f)
            boh = BOHGenerator(
                results_path, cluster_type="kubernetes"
            ).generate()
        self.assertEqual(1, len(boh.test_results))
        return boh.test_results[0]

    def test_boh_thresholds_pass_on_healthy_pulse_metrics(self):
        result = self._generate_boh()
        self.assertEqual(pb.RESULT_STATUS_SUCCESS, result.result_status)

        by_key = {m.key: m for m in result.metrics}
        self.assertEqual(
            pb.Comparator.COMPARATOR_GTE,
            by_key["Throughput"].validation_criteria.comparator,
        )
        self.assertEqual(
            round(1004.0 * 0.95),
            by_key[
                "Throughput"
            ].validation_criteria.expected_value.number_value,
        )
        for bool_key in ("Valid Checksum", "Checksum Match"):
            vc = by_key[bool_key].validation_criteria
            self.assertEqual(pb.Comparator.COMPARATOR_EQ, vc.comparator)
            self.assertTrue(vc.expected_value.HasField("bool_value"))
            self.assertTrue(vc.expected_value.bool_value)
            self.assertEqual(pb.RESULT_STATUS_SUCCESS, vc.result_status)

    def test_boh_thresholds_fail_on_checksum_mismatch(self):
        result = self._generate_boh(checksum_match="False")
        self.assertEqual(pb.RESULT_STATUS_FAILED, result.result_status)
        by_key = {m.key: m for m in result.metrics}
        self.assertEqual(
            pb.RESULT_STATUS_FAILED,
            by_key["Checksum Match"].validation_criteria.result_status,
        )

    def test_boh_thresholds_fail_on_invalid_or_none_checksum(self):
        for bad_valid, bad_match in (("False", "True"), ("True", "None")):
            with self.subTest(valid=bad_valid, match=bad_match):
                result = self._generate_boh(
                    valid_checksum=bad_valid, checksum_match=bad_match
                )
                self.assertEqual(pb.RESULT_STATUS_FAILED, result.result_status)

    def test_boh_thresholds_fail_on_low_throughput(self):
        result = self._generate_boh(throughput=500.0)
        self.assertEqual(pb.RESULT_STATUS_FAILED, result.result_status)


if __name__ == "__main__":
    unittest.main()
