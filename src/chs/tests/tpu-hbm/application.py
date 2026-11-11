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


class TpuHbm(ExecutableApplication):
    """TPU JAX HBM benchmark for cluster health scanner.

    This application runs TPU HBM benchmarks utilizing the JAX
    accelerator-microbenchmarks suite.
    """

    name = "tpu-hbm"
    maintainers("simrankaurb")
    tags("diagnostics", "tpu", "hbm")

    variable(
        "benchmarks_yaml_file",
        default="hbm_copy_bf16.yaml",
        description="Filename of the benchmarks config in the benchmarks/ directory",
    )
    variable(
        "benchmarks_yaml_escaped",
        default="",
        description="YAML content for benchmarks configuration (escaped for printf)",
    )
    variable(
        "config_file",
        default="{experiment_run_dir}/chs_tpu_hbm.yaml",
        description="Path to config file",
    )
    variable(
        "dtype",
        default="bfloat16",
        description="Data type for HBM benchmark",
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
            'set -o pipefail; {driver_escape_overrides} {tpu_run_wrapper} env PYTHONUNBUFFERED=1 tpums benchmark run-config {config_file} 2>&1 | sed "s/^/HOST $\\{MY_NODE_NAME:-$(hostname)\\} ({dtype}): /"',
        ],
        use_mpi=False,
    )
    workload("check", executables=["execute"])

    figure_of_merit_context(
        "node_params",
        regex=r"HOST (?P<node>\S+) \((?P<dtype>\w+)\): Running \[\d+/\d+\] hbm \(\w+_dim_(?P<num_elements>\d+)_dev_\d+\)",
        output_format="{node},{num_elements},{dtype}",
    )

    figure_of_merit(
        "Bandwidth (median)",
        fom_regex=r".*Success \[\d+/\d+\] hbm\. Metrics: .*'xprof_bandwidth_per_device_gb_s': (?:np\.float64\()?(?P<bandwidth>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?",
        group_name="bandwidth",
        units="GB/s",
        contexts=["node_params"],
    )

    figure_of_merit(
        "Time Taken (median)",
        fom_regex=r".*Success \[\d+/\d+\] hbm\. Metrics: .*'xprof_p50_ms': (?:np\.float64\()?(?P<time>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?",
        group_name="time",
        units="ms",
        fom_type=FomType.INFO,
        contexts=["node_params"],
    )

    figure_of_merit(
        "Tensor Size",
        fom_regex=r".*Success \[\d+/\d+\] hbm\. Metrics: .*'data_size_mib': (?P<tensor_size>\d+(?:\.\d+)?)",
        group_name="tensor_size",
        units="MiB",
        fom_type=FomType.INFO,
        contexts=["node_params"],
    )
