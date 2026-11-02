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


class TpuCollectives(ExecutableApplication):
    """TPU JAX Collectives benchmark for cluster health scanner.

    This application runs TPU Collectives benchmarks utilizing the JAX
    accelerator-microbenchmarks suite.
    """

    name = "tpu-collectives"
    maintainers("simrankaurb")
    tags("diagnostics", "tpu", "collectives")

    variable(
        "benchmarks_yaml_file",
        default="collectives_all_gather.yaml",
        description="Filename of the benchmarks config in the benchmarks/ directory",
    )
    variable(
        "benchmarks_yaml_escaped",
        default="",
        description="YAML content for benchmarks configuration (escaped for printf)",
    )
    variable(
        "config_file",
        default="{experiment_run_dir}/chs_tpu_collectives.yaml",
        description="Path to config file",
    )
    variable(
        "dtype",
        default="float32",
        description="Data type for collectives benchmark",
    )

    variable(
        "driver_escape_overrides",
        default="",
        description="Optional Driver Escape overrides injected before execution",
    )

    executable(
        "execute",
        template=[
            "printf '%b' '{benchmarks_yaml_escaped}' | tee {config_file}",
            'set -o pipefail; {driver_escape_overrides} env PYTHONUNBUFFERED=1 tpums benchmark run-config {config_file} 2>&1 | sed "s/^/HOST $\\{MY_NODE_NAME:-$(hostname)\\} ({dtype}): /"',
        ],
        use_mpi=False,
    )

    workload("all_gather", executables=["execute"])
    workload("psum", executables=["execute"])
    workload("all_to_all", executables=["execute"])

    figure_of_merit_context(
        "params",
        regex=r"HOST (?P<node>\S+) \((?P<dtype>\w+)\): Running \[\d+/\d+\] (?P<op_type>all_gather|all_reduce|all_to_all) \(dim_(?P<matrix_dim>\d+)\S*\)",
        output_format="{node},{matrix_dim},{dtype}",
    )

    figure_of_merit(
        "Bandwidth (median)",
        fom_regex=r".*Success \[\d+/\d+\] (?:all_gather|all_reduce|all_to_all)\. Metrics: .*'xprof_bandwidth_per_chip_gb_s': (?:np\.float64\()?(?P<bandwidth>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?",
        group_name="bandwidth",
        units="GB/s",
        contexts=["params"],
    )

    figure_of_merit(
        "Time Taken (median)",
        fom_regex=r".*Success \[\d+/\d+\] (?:all_gather|all_reduce|all_to_all)\. Metrics: .*'xprof_p50_ms': (?:np\.float64\()?(?P<time>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?",
        group_name="time",
        units="ms",
        fom_type=FomType.INFO,
        contexts=["params"],
    )
