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
        "container_app_dir",
        default="/app/accelerator-microbenchmarks",
        description="Directory of accelerator-microbenchmarks inside the container",
    )
    variable(
        "run_script",
        default="Ironwood/src/run_benchmark.py",
        description="Path to run_benchmark.py relative to container_app_dir",
    )
    variable(
        "benchmarks_yaml_file",
        default="gemm_16384_bf16.yaml",
        description="Filename of the benchmarks config in configs/benchmarks/ directory",
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
            '{driver_escape_overrides} {tpu_run_wrapper} python3 -u {container_app_dir}/{run_script} --config {config_file} 2>&1 | sed "s/^/HOST $\\{MY_NODE_NAME:-$(hostname)\\}: /"',
        ],
        use_mpi=False,
    )
    workload("check", executables=["execute"])

    figure_of_merit_context(
        "params",
        regex=r"HOST (?P<node>\S+): Running .* with params:.*'m': (?P<m>\d+), 'k': (?P<k>\d+), 'n': (?P<n>\d+),.*'dtype': <class 'jax.numpy.(?P<dtype>\w+)'>",
        output_format="{node},{m}x{n}x{k},{dtype}",
    )

    figure_of_merit(
        "Throughput",
        fom_regex=r".*Throughput \(median\): (?P<tflops>\d+\.\d+) TFLOP / second / device",
        group_name="tflops",
        units="TFLOP/s/device",
        contexts=["params"],
    )

    figure_of_merit(
        "Total Throughput",
        fom_regex=r".*TotalThroughput \(median\): (?P<total_tflops>\d+\.\d+) TFLOP / second",
        group_name="total_tflops",
        units="TFLOPs/s",
        contexts=["params"],
    )

    figure_of_merit(
        "MFU",
        fom_regex=r".*MFU: (?P<mfu>\d+\.\d+)%",
        group_name="mfu",
        units="%",
        contexts=["params"],
    )

    figure_of_merit(
        "Step Time",
        fom_regex=r".*Step Time \(median\): (?P<step_time>\d+\.\d+)",
        group_name="step_time",
        units="s",
        fom_type=FomType.INFO,
        contexts=["params"],
    )

    figure_of_merit(
        "Total FLOPs",
        fom_regex=r".*Total floating-point ops: (?P<flops>\d+)",
        group_name="flops",
        units="ops",
        fom_type=FomType.INFO,
        contexts=["params"],
    )
