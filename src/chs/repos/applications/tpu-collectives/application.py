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
        "run_dir",
        default="/app/accelerator-microbenchmarks",
        description="Directory of accelerator-microbenchmarks",
    )
    variable(
        "run_script",
        default="Ironwood/src/run_benchmark.py",
        description="Path to run_benchmark.py relative to run_dir",
    )
    variable(
        "benchmarks_yaml_file",
        default="collectives_all_gather.yaml",
        description="Filename of the benchmarks config in configs/benchmarks/ directory",
    )
    variable(
        "benchmarks_yaml_escaped",
        default="",
        description="YAML content for benchmarks configuration (escaped for printf)",
    )
    variable(
        "config_file",
        default="Ironwood/configs/chs_tpu_collectives.yaml",
        description="Path to config file",
    )

    variable(
        "driver_escape_overrides",
        default="",
        description="Optional Driver Escape overrides injected before execution",
    )

    executable(
        "execute",
        template=(
            '{driver_escape_overrides} cd {run_dir} && printf "%b" "{benchmarks_yaml_escaped}" > {config_file} && python3 {run_script} --config {config_file} 2>&1 | sed "s/^/HOST $\\{MY_NODE_NAME:-$(hostname)\\}: /"'
        ),
        use_mpi=False,
    )

    workload("all_gather", executables=["execute"])
    workload("psum", executables=["execute"])
    workload("all_to_all", executables=["execute"])

    figure_of_merit_context(
        "params",
        regex=r"HOST (?P<node>\S+): Running\s+(?:benchmark:\s+)?(?P<op_type>\w+)\s+with params:.*?['\"]matrix_dim['\"]:\s*(?P<matrix_dim>\d+).*?['\"]dtype['\"]:\s*<class\s+['\"]jax\.numpy\.(?P<dtype>\w+)['\"]>",
        output_format="{node},{matrix_dim},{dtype}",
    )

    figure_of_merit(
        "Bandwidth (median)",
        fom_regex=r".*metrics:\s*.*?['\"]achieved_bw \(GB/s\)_p50['\"]:\s*(?:np\.float64\()?(?P<bandwidth>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?",
        group_name="bandwidth",
        units="GB/s",
        contexts=["params"],
    )

    figure_of_merit(
        "Time Taken (median)",
        fom_regex=r".*metrics:\s*.*?['\"]step_time_ms_p50['\"]:\s*(?:np\.float64\()?(?P<time>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?",
        group_name="time",
        units="ms",
        fom_type=FomType.INFO,
        contexts=["params"],
    )
