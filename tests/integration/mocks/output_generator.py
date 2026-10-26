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

from __future__ import annotations
import yaml
from tests.integration.create_kind_cluster import resolve_config_path
from tests.integration.mocks.mock_utils import resolve_node_mapping

TPU_HBM_TEMPLATE = (
    "HOST {node}: Running benchmark with params: 'num_elements': {elements}, "
    "'dtype': <class 'jax.numpy.bfloat16'>\n"
    "HOST {node}: bandwidth (median): {bandwidth}\n"
    "HOST {node}: time taken (median): 0.5\n"
    "HOST {node}: Tensor size: 128 MB\n"
)

TPU_INFO_TEMPLATE = """\
HOST {node}: JAX Devices: {chips}
HOST {node}: Host Info: Linux x86_64
HOST {node}: Libtpu version: 0.1.dev
HOST {node}: Accelerator type: v6e
{stderr}\n\
"""

TPU_GEMM_TEMPLATE = (
    "HOST {node}: Running benchmark with params: 'm': 16384, 'k': 16384, "
    "'n': 16384, 'dtype': <class 'jax.numpy.bfloat16'>\n"
    "HOST {node}: Throughput (median): {tflops} TFLOP / second / device\n"
    "HOST {node}: TotalThroughput (median): 250.0 TFLOP / second\n"
    "HOST {node}: MFU: 45.0%\n"
    "HOST {node}: Step Time (median): 3.5\n"
    "HOST {node}: Total floating-point ops: 8796093022208\n"
)

TPU_COLLECTIVES_TEMPLATE = (
    "HOST {node}: Running benchmark: psum with params: "
    "{'matrix_dim': 16384, 'dtype': <class 'jax.numpy.bfloat16'>}\n"
    "HOST {node}: Result metrics: {'achieved_bw (GB/s)_p50': "
    "np.float64({bandwidth}), 'step_time_ms_p50': np.float64({time})}\n"
)

TPU_COMPUTE_STRESS_TEMPLATE = (
    "HOST {node}: NODE:{node} EVENT:MEASURE_END DURATION: 10s "
    "TFLOPs: {performance}\n"
)

TPU_HBM_STRESS_TEMPLATE = (
    "HOST {node}: NODE:{node} EVENT:MEASURE_END DURATION: 10s "
    "BANDWIDTH_GBPS:{bandwidth}\n"
)

TPU_HOST_DEVICE_TEMPLATE = (
    "HOST {node}: Running host_device with params: {'transfer_type': "
    "'pinned_memory', 'num_devices': 8, 'data_size_mib': 4096, "
    "'input_type': 'jax', 'dtype': <class 'jax.numpy.float32'>, "
    "'num_runs': 20}\n"
    "HOST {node}:   H2D_bw (GiB/s) median: {bandwidth}, P95: {bandwidth}\n"
    "HOST {node}:   D2H_bw (GiB/s) median: {bandwidth}, P95: {bandwidth}\n"
)

TPU_NETWORK_STRESS_TEMPLATE = (
    "HOST {node}: Host: {node} | Axis: X | {bandwidth} GB/s\n"
)

GPU_SMI_TEMPLATE = (
    "0, NVIDIA A4X, 535.104, P0, 00000000:00:04.0, 0000000000000, "
    "00000000-0000-0000-0000-000000000000, 100.0, 200.0, 1000, 1000\n"
    "GPU Count: 4\n"
    "CUDA Version: 12.2\n"
)

GPU_DCGM_TEMPLATE = """\
| Diagnostic Test | Pass |
| | GPU0: Pass |
| DCGM Version | 3.1.7 |
| Driver Version Detected | 535.104 |
"""

GPU_STREAM_TEMPLATE = """\
0: Device 0: ... = {peak_bw} GB/s PEAK
0: Copy: {rate} 0.1 0.1 0.1
0: Scale: {rate} 0.1 0.1 0.1
0: Add: {rate} 0.1 0.1 0.1
0: Triad: {rate} 0.1 0.1 0.1
"""

GPU_NCCL_TEMPLATE = (
    "# out of place allreduce\n"
    "#       size         count    type   redop    root     time   algbw   "
    "busbw #wrong     time   algbw   busbw #wrong\n"
    "        1024           256   float     sum      -1      1.0     1.0   "
    "{bandwidth}      0      1.0     1.0   {bandwidth}      0\n"
    "# Out of bounds values : 0\n"
    "Avg bus bandwidth    : {bandwidth}\n"
)

GPU_GEMM_TEMPLATE = """\
All GEMM tests passed
MEDIAN = {tflops}
MEAN = {tflops}
MIN = {tflops}
MAX = {tflops}
"""


class RambleOutputGenerator:
    HEALTHY_HBM = 5000.0
    HEALTHY_HBM_ELEMENTS = 2147483648
    HEALTHY_CHIPS = 4
    HEALTHY_TFLOPS = 5000.0
    HEALTHY_GPU_BW = 1000.0
    HEALTHY_NCCL_BW = 5500.0
    HEALTHY_GPU_TFLOPS = 45000.0

    HEALTHY_GEMM_TFLOPS_BY_FAMILY = (
        ("a4x", 45000.0),
        ("a4-highgpu", 80000.0),
        ("a3-ultra", 60000.0),
        ("a3-mega", 60000.0),
    )

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.scenarios = self._build_scenario_map()

    def _load_config(self) -> dict:
        resolved_path = resolve_config_path(self.config_path)
        with open(resolved_path, "r") as f:
            return yaml.safe_load(f) or {}

    def _unpack_pairwise_links(
        self, scenario: dict, test: str, scenarios: dict
    ) -> None:
        """Unpacks cluster-level degraded links into per-node scenarios."""
        for link in scenario.get("degraded_links", []):
            # Unpack source
            src = link.get("source", {})
            src_np = src.get("nodepool", "")
            src_idx = src.get("node_index", 0)
            scenarios[(src_np, src_idx, test)] = {
                "outcome": scenario.get("outcome", ""),
                "metrics": link.get("metrics", {}),
            }

            # Unpack target
            tgt = link.get("target", {})
            tgt_np = tgt.get("nodepool", "")
            tgt_idx = tgt.get("node_index", 0)
            scenarios[(tgt_np, tgt_idx, test)] = {
                "outcome": scenario.get("outcome", ""),
                "metrics": link.get("metrics", {}),
            }

    def _build_scenario_map(self) -> dict[tuple[str, int, str], dict]:
        scenarios = {}
        for scenario in self.config.get("test_scenarios", []):
            test = scenario.get("test", "")
            if "degraded_links" in scenario:
                self._unpack_pairwise_links(scenario, test, scenarios)
            else:
                nodepool = scenario.get("nodepool", "")
                index = scenario.get("node_index", 0)
                scenarios[(nodepool, index, test)] = scenario
        return scenarios

    def get_output_dict(self, test_name: str) -> dict:
        """Returns a dict with per-pool mappings and a default log."""
        # Find all scenarios for this test.
        # test_name from Ramble might be "nvidia-stream" but scenario config
        # expects "stream". So we strip "nvidia-" if present.
        config_test_name = test_name.replace("nvidia-", "")
        # For nccl tests, we want to match whatever was put in the yaml
        # e.g., if Ramble passes "nccl-tests", but config says "nccl-cluster"
        if "nccl" in test_name:
            test_specific = {
                k: v for k, v in self.scenarios.items() if "nccl" in k[2]
            }
        else:
            test_specific = {
                k: v
                for k, v in self.scenarios.items()
                if k[2] in (test_name, config_test_name)
            }

        node_mapping = None
        try:
            node_mapping = resolve_node_mapping()
        except Exception:
            pass

        output_map = {}
        # We fill 'default' first
        default_log = self._render_template(test_name, None)

        for (nodepool, index, t), scenario in test_specific.items():
            if node_mapping and nodepool in node_mapping:
                pool_nodes = node_mapping[nodepool]
                if index < len(pool_nodes):
                    node_name = pool_nodes[index]
                    output_map[node_name] = self._render_template(
                        test_name, scenario
                    )

        return {"nodes": output_map, "default": default_log}

    def _render_template(self, test_name: str, scenario: dict | None) -> str:
        if test_name == "tpu-info":
            return self.populate_tpu_info_logs(scenario)
        elif test_name == "tpu-hbm":
            return self.populate_hbm_logs(scenario)
        elif test_name == "tpu-gemm":
            return self.populate_tpu_gemm_logs(scenario)
        elif (
            test_name == "tpu-collectives"
            or test_name == "tpu-collectives-single-node"
        ):
            return self.populate_tpu_collectives_logs(scenario)
        elif test_name == "tpu-compute-stress":
            return self.populate_tpu_compute_stress_logs(scenario)
        elif test_name == "tpu-hbm-stress":
            return self.populate_tpu_hbm_stress_logs(scenario)
        elif test_name == "tpu-host-device":
            return self.populate_tpu_host_device_logs(scenario)
        elif test_name == "tpu-network-stress":
            return self.populate_tpu_network_stress_logs(scenario)
        elif test_name in ("nvidia-smi", "smi"):
            return self.populate_nvidia_smi_logs(scenario)
        elif test_name in ("nvidia-dcgm", "dcgm"):
            return self.populate_dcgm_logs(scenario)
        elif test_name in ("nvidia-stream", "stream"):
            return self.populate_stream_logs(scenario)
        elif test_name.startswith("nccl"):
            return self.populate_nccl_logs(scenario)
        elif test_name in ("nvidia-gemm", "gemm"):
            return self.populate_gemm_logs(scenario)
        else:
            if scenario and scenario.get("outcome") == "FAILED":
                return (
                    "HOST {node}: "
                    + str(scenario.get("stderr", "Crash"))
                    + "\n"
                )
            return "HOST {node}: Default mock output\n"

    def populate_tpu_network_stress_logs(self, scenario: dict | None) -> str:
        if scenario and scenario.get("outcome") == "FAILED":
            return (
                "HOST {node}: "
                + str(scenario.get("stderr", "Network stress crash"))
                + "\n"
            )

        bandwidth = 300.0  # Default healthy BW for network stress
        if scenario and scenario.get("outcome") == "DEGRADED":
            metrics = scenario.get("metrics", {})
            if "Bandwidth" in metrics or "bandwidth" in metrics:
                bandwidth = metrics.get(
                    "Bandwidth", metrics.get("bandwidth", bandwidth)
                )

        return TPU_NETWORK_STRESS_TEMPLATE.replace(
            "{bandwidth}", str(bandwidth)
        )

    def populate_tpu_gemm_logs(self, scenario: dict | None) -> str:
        if scenario and scenario.get("outcome") == "FAILED":
            return (
                "HOST {node}: "
                + str(scenario.get("stderr", "GEMM crash"))
                + "\n"
            )

        tflops = self.HEALTHY_TFLOPS
        if scenario and scenario.get("outcome") == "DEGRADED":
            metrics = scenario.get("metrics", {})
            tflops = metrics.get("Throughput", tflops)

        return TPU_GEMM_TEMPLATE.replace("{tflops}", str(tflops))

    def populate_hbm_logs(self, scenario: dict | None) -> str:
        if scenario and scenario.get("outcome") == "FAILED":
            return (
                "HOST {node}: "
                + str(scenario.get("stderr", "HBM crash"))
                + "\n"
            )

        bandwidth = self.HEALTHY_HBM
        if scenario and scenario.get("outcome") == "DEGRADED":
            metrics = scenario.get("metrics", {})
            if "Bandwidth (median)" in metrics:
                bandwidth = metrics["Bandwidth (median)"]

        return TPU_HBM_TEMPLATE.replace("{bandwidth}", str(bandwidth)).replace(
            "{elements}", str(self.HEALTHY_HBM_ELEMENTS)
        )

    def populate_tpu_info_logs(self, scenario: dict | None) -> str:
        chips = 8 if "tpu7x" in self.config_path.lower() else self.HEALTHY_CHIPS
        stderr = ""
        if scenario and scenario.get("outcome") == "FAILED":
            stderr = "HOST {node}: " + str(scenario.get("stderr", "Info crash"))
            chips = 0

        return TPU_INFO_TEMPLATE.replace("{chips}", str(chips)).replace(
            "{stderr}", stderr
        )

    def populate_tpu_collectives_logs(self, scenario: dict | None) -> str:
        if scenario and scenario.get("outcome") == "FAILED":
            return (
                "HOST {node}: "
                + str(scenario.get("stderr", "Collectives crash"))
                + "\n"
            )

        # Default healthy bandwidth for mock (exceeds TPU 7x threshold)
        bandwidth = 600.0
        time = 2.0  # Default healthy time for mock

        if scenario and scenario.get("outcome") == "DEGRADED":
            metrics = scenario.get("metrics", {})
            if "Bandwidth (median)" in metrics or "achieved_bw" in metrics:
                bandwidth = metrics.get(
                    "Bandwidth (median)", metrics.get("achieved_bw", bandwidth)
                )
            if (
                "Time Taken (median)" in metrics
                or "step_time_ms_p50" in metrics
            ):
                time = metrics.get(
                    "Time Taken (median)", metrics.get("step_time_ms_p50", time)
                )

        return TPU_COLLECTIVES_TEMPLATE.replace(
            "{bandwidth}", str(bandwidth)
        ).replace("{time}", str(time))

    def populate_tpu_compute_stress_logs(self, scenario: dict | None) -> str:
        if scenario and scenario.get("outcome") == "FAILED":
            return (
                "HOST {node}: "
                + str(scenario.get("stderr", "Compute stress crash"))
                + "\n"
            )

        tflops = self.HEALTHY_TFLOPS
        if scenario and scenario.get("outcome") == "DEGRADED":
            metrics = scenario.get("metrics", {})
            if "performance" in metrics or "Throughput" in metrics:
                tflops = metrics.get(
                    "performance", metrics.get("Throughput", tflops)
                )

        return TPU_COMPUTE_STRESS_TEMPLATE.replace("{performance}", str(tflops))

    def populate_tpu_hbm_stress_logs(self, scenario: dict | None) -> str:
        if scenario and scenario.get("outcome") == "FAILED":
            return (
                "HOST {node}: "
                + str(scenario.get("stderr", "HBM stress crash"))
                + "\n"
            )

        bandwidth = (
            7000.0 if "tpu7x" in self.config_path.lower() else self.HEALTHY_HBM
        )
        if scenario and scenario.get("outcome") == "DEGRADED":
            metrics = scenario.get("metrics", {})
            if "bandwidth" in metrics or "Bandwidth" in metrics:
                bandwidth = metrics.get(
                    "bandwidth", metrics.get("Bandwidth", bandwidth)
                )

        return TPU_HBM_STRESS_TEMPLATE.replace("{bandwidth}", str(bandwidth))

    def populate_tpu_host_device_logs(self, scenario: dict | None) -> str:
        if scenario and scenario.get("outcome") == "FAILED":
            return (
                "HOST {node}: "
                + str(scenario.get("stderr", "Host-device test crash"))
                + "\n"
            )

        bandwidth = 210.0
        if scenario and scenario.get("outcome") == "DEGRADED":
            metrics = scenario.get("metrics", {})
            if "Bandwidth" in metrics or "h2d_bandwidth" in metrics:
                bandwidth = metrics.get(
                    "Bandwidth", metrics.get("h2d_bandwidth", bandwidth)
                )

        return TPU_HOST_DEVICE_TEMPLATE.replace("{bandwidth}", str(bandwidth))

    def _get_machine_type(self, scenario: dict | None = None) -> str:
        if scenario and "nodepool" in scenario:
            pool_name = scenario["nodepool"]
            for pool in self.config.get("nodepools", []):
                if isinstance(pool, dict) and pool.get("name") == pool_name:
                    return pool.get("machine_type", "")
        for pool in self.config.get("nodepools", []):
            if isinstance(pool, dict) and pool.get("machine_type"):
                return pool["machine_type"]
        return ""

    def _get_gpu_info(self, scenario: dict | None = None) -> tuple[int, str]:
        machine_type = self._get_machine_type(scenario).lower()
        if "a3-ultra" in machine_type or machine_type == "a3-ultragpu-8g":
            return 8, "NVIDIA H200"
        elif "a3-mega" in machine_type or machine_type == "a3-megagpu-8g":
            return 8, "NVIDIA H100"
        elif "a4-highgpu" in machine_type or machine_type == "a4-highgpu-8g":
            return 8, "NVIDIA B200"
        elif "a4x" in machine_type:
            return 4, "NVIDIA A4X"
        elif "8g" in machine_type:
            return 8, "NVIDIA GPU"
        return 4, "NVIDIA A4X"

    def populate_nvidia_smi_logs(self, scenario: dict | None) -> str:
        if scenario and scenario.get("outcome") == "FAILED":
            return "HOST {node}: " + str(scenario.get("stderr", "Crash")) + "\n"

        gpu_count, gpu_name = self._get_gpu_info(scenario)
        if scenario and scenario.get("outcome") == "DEGRADED":
            metrics = scenario.get("metrics", {})
            gpu_count = int(
                metrics.get(
                    "gpus",
                    metrics.get(
                        "gpu_count", metrics.get("GPU Count", gpu_count)
                    ),
                )
            )

        lines = []
        for i in range(gpu_count):
            lines.append(
                f"{i}, {gpu_name}, 535.104, P0, 00000000:00:{i + 4:02x}.0, "
                f"000000000000{i}, "
                f"00000000-0000-0000-0000-00000000000{i}, "
                "100.0, 200.0, 1000, 1000"
            )
        lines.append(f"GPU Count: {gpu_count}")
        lines.append("CUDA Version: 12.2\n")
        return "\n".join(lines)

    def populate_dcgm_logs(self, scenario: dict | None) -> str:
        if scenario and scenario.get("outcome") == "FAILED":
            return "HOST {node}: " + str(scenario.get("stderr", "Crash")) + "\n"
        return GPU_DCGM_TEMPLATE

    def populate_stream_logs(self, scenario: dict | None) -> str:
        if scenario and scenario.get("outcome") == "FAILED":
            return "HOST {node}: " + str(scenario.get("stderr", "Crash")) + "\n"

        peak_bw_gb_s = self.HEALTHY_GPU_BW
        achieved_gb_s = peak_bw_gb_s * 0.90

        if scenario and scenario.get("outcome") == "DEGRADED":
            metrics = scenario.get("metrics", {})
            achieved_gb_s = float(
                metrics.get(
                    "Bandwidth", metrics.get("bandwidth", peak_bw_gb_s * 0.50)
                )
            )

        return GPU_STREAM_TEMPLATE.replace(
            "{peak_bw}", f"{peak_bw_gb_s:.1f}"
        ).replace("{rate}", f"{achieved_gb_s * 1000.0:.1f}")

    def populate_nccl_logs(self, scenario: dict | None) -> str:
        if scenario and scenario.get("outcome") == "FAILED":
            return "HOST {node}: " + str(scenario.get("stderr", "Crash")) + "\n"
        bandwidth = self.HEALTHY_NCCL_BW
        if scenario and scenario.get("outcome") == "DEGRADED":
            metrics = scenario.get("metrics", {})
            bandwidth = metrics.get(
                "Bandwidth", metrics.get("bandwidth", bandwidth)
            )
        return GPU_NCCL_TEMPLATE.replace(
            "{bandwidth}", f"{float(bandwidth):.1f}"
        )

    def populate_gemm_logs(self, scenario: dict | None) -> str:
        if scenario and scenario.get("outcome") == "FAILED":
            return "HOST {node}: " + str(scenario.get("stderr", "Crash")) + "\n"
        machine_type = self._get_machine_type(scenario).lower()
        tflops = self.HEALTHY_GPU_TFLOPS
        for family, healthy_tflops in self.HEALTHY_GEMM_TFLOPS_BY_FAMILY:
            if family in machine_type:
                tflops = healthy_tflops
                break

        if scenario and scenario.get("outcome") == "DEGRADED":
            metrics = scenario.get("metrics", {})
            tflops = metrics.get(
                "Throughput",
                metrics.get(
                    "performance",
                    metrics.get("Value", tflops),
                ),
            )

        return GPU_GEMM_TEMPLATE.replace("{tflops}", f"{float(tflops):.1f}")
