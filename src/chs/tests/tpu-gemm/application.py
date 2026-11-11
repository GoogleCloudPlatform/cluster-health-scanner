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


class TpuGemm(ExecutableApplication):
    """TPU JAX GEMM benchmark for cluster health scanner.

    This application runs TPU GEMM benchmarks utilizing the JAX
    accelerator-microbenchmarks suite.
    """

    name = "tpu-gemm"
    maintainers("simrankaurb")
    tags("diagnostics", "tpu", "gemm")

    variable(
        "benchmarks_yaml_file",
        default="gemm_16384_bf16.yaml",
        description="Filename of the benchmarks config in the benchmarks/ directory",
    )
    variable(
        "benchmarks_yaml_escaped",
        default="",
        description="YAML content for benchmarks configuration (escaped for printf)",
    )
    variable(
        "config_file",
        default="{experiment_run_dir}/chs_tpu_gemm.yaml",
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
        regex=r"HOST (?P<node>\S+): Running \[\d+/\d+\] gemm \(m_(?P<m>\d+)_k_(?P<k>\d+)_n_(?P<n>\d+)_(?P<dtype>\w+?)_to_",
        output_format="{node},{m}x{n}x{k},{dtype}",
    )

    figure_of_merit(
        "Throughput",
        fom_regex=r".*Success \[\d+/\d+\] gemm\. Metrics: .*'xprof_tflops_per_device': (?:np\.float64\()?(?P<tflops>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?",
        group_name="tflops",
        units="TFLOP/s/device",
        contexts=["params"],
    )

    figure_of_merit(
        "Throughput per chip",
        fom_regex=r".*Success \[\d+/\d+\] gemm\. Metrics: .*'xprof_tflops_per_chip': (?:np\.float64\()?(?P<chip_tflops>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?",
        group_name="chip_tflops",
        units="TFLOP/s/chip",
        contexts=["params"],
    )

    figure_of_merit(
        "MFU",
        fom_regex=r".*Success \[\d+/\d+\] gemm\. Metrics: .*'xprof_compute_roofline_efficiency_pct': (?:np\.float64\()?(?P<mfu>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?",
        group_name="mfu",
        units="%",
        contexts=["params"],
    )

    figure_of_merit(
        "Step Time",
        fom_regex=r".*Success \[\d+/\d+\] gemm\. Metrics: .*'xprof_p50_ms': (?:np\.float64\()?(?P<step_time>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?",
        group_name="step_time",
        units="ms",
        fom_type=FomType.INFO,
        contexts=["params"],
    )

    figure_of_merit(
        "Total FLOPs",
        fom_regex=r".*Success \[\d+/\d+\] gemm\. Metrics: .*'total_flops': (?P<flops>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        group_name="flops",
        units="FLOP",
        fom_type=FomType.INFO,
        contexts=["params"],
    )
