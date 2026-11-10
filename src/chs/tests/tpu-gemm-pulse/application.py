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

from ramble.appkit import *


class TpuGemmPulse(ExecutableApplication):
    """TPU JAX GEMM pulse benchmark for cluster health scanner.

    This application runs multi-cycle idle-to-load chained GEMM pulse
    benchmarks with per-window 64-bit checksum verification utilizing the JAX
    accelerator-microbenchmarks suite.
    """

    name = "tpu-gemm-pulse"
    maintainers("simrankaurb")
    tags("diagnostics", "stress", "tpu", "gemm")

    variable(
        "benchmarks_yaml_file",
        default="gemm_pulse_16384_bf16.yaml",
        description="Filename of the benchmarks config in the benchmarks/ directory",
    )
    variable(
        "benchmarks_yaml_escaped",
        default="",
        description="YAML content for benchmarks configuration (escaped for printf)",
    )
    variable(
        "config_file",
        default="{experiment_run_dir}/chs_tpu_gemm_pulse.yaml",
        description="Path to config file",
    )
    variable(
        "driver_escape_overrides",
        default="",
        description="Optional Driver Escape overrides injected before execution",
    )
    variable(
        "tpu_run_wrapper",
        default="",
        description="Wrapper command for running TPU benchmarks (e.g. srun on Slurm)",
    )

    executable(
        "execute",
        template=[
            "printf '%b' '{benchmarks_yaml_escaped}' | tee {config_file}",
            'set -o pipefail; {driver_escape_overrides} {tpu_run_wrapper} env PYTHONUNBUFFERED=1 tpums benchmark run-config {config_file} 2>&1 | sed "s/^/HOST $\\{MY_NODE_NAME:-$(hostname)\\}: /"',
        ],
        use_mpi=False,
    )
    workload("check", executables=["execute"])

    figure_of_merit_context(
        "params",
        regex=r"HOST (?P<node>\S+): Running \[\d+/\d+\] gemm_pulse \(m_(?P<m>\d+)_k_(?P<k>\d+)_(?P<dtype>\w+?)_to_\w+_chain_(?P<chain_length>\d+)_cycles_(?P<num_cycles>\d+)_idle_(?P<idle_s>[0-9.]+)",
        output_format="{node},{m}x{k}x{k},{dtype},chain={chain_length},cycles={num_cycles}",
    )

    figure_of_merit(
        "Throughput",
        fom_regex=r".*Success \[\d+/\d+\] gemm_pulse\. Metrics: .*'xprof_tflops_per_device': (?:np\.float64\()?(?P<tflops>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?",
        group_name="tflops",
        units="TFLOP/s/device",
        contexts=["params"],
    )

    figure_of_merit(
        "Valid Checksum",
        fom_regex=r".*Success \[\d+/\d+\] gemm_pulse\. Metrics: .*'all_devices_valid_checksum': (?P<valid_checksum>True|False|None)",
        group_name="valid_checksum",
        units="",
        contexts=["params"],
    )

    figure_of_merit(
        "Checksum Match",
        fom_regex=r".*Success \[\d+/\d+\] gemm_pulse\. Metrics: .*'intra_host_checksum_match': (?P<checksum_match>True|False|None)",
        group_name="checksum_match",
        units="",
        contexts=["params"],
    )

    figure_of_merit(
        "Throughput per chip",
        fom_regex=r".*Success \[\d+/\d+\] gemm_pulse\. Metrics: .*'xprof_tflops_per_chip': (?:np\.float64\()?(?P<chip_tflops>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?",
        group_name="chip_tflops",
        units="TFLOP/s/chip",
        fom_type=FomType.INFO,
        contexts=["params"],
    )

    figure_of_merit(
        "Slowdown Ratio",
        fom_regex=r".*Success \[\d+/\d+\] gemm_pulse\. Metrics: .*'xprof_slowdown_ratio': (?:np\.float64\()?(?P<slowdown_ratio>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?",
        group_name="slowdown_ratio",
        units="",
        fom_type=FomType.INFO,
        contexts=["params"],
    )

    figure_of_merit(
        "MFU",
        fom_regex=r".*Success \[\d+/\d+\] gemm_pulse\. Metrics: .*'xprof_compute_roofline_efficiency_pct': (?:np\.float64\()?(?P<mfu>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?",
        group_name="mfu",
        units="%",
        fom_type=FomType.INFO,
        contexts=["params"],
    )

    figure_of_merit(
        "Step Time",
        fom_regex=r".*Success \[\d+/\d+\] gemm_pulse\. Metrics: .*'xprof_p50_ms': (?:np\.float64\()?(?P<step_time>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?",
        group_name="step_time",
        units="ms",
        fom_type=FomType.INFO,
        contexts=["params"],
    )
