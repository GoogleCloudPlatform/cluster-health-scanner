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

_BW_VALUE = r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_H2D_MEDIAN_REGEX = (
    r".*H2D_bw \(GiB/s\) median:\s*(?P<h2d_bandwidth>" + _BW_VALUE + r")"
)
_D2H_MEDIAN_REGEX = (
    r".*D2H_bw \(GiB/s\) median:\s*(?P<d2h_bandwidth>" + _BW_VALUE + r")"
)


class TpuHostDevice(ExecutableApplication):
    """TPU JAX Host-Device I/O benchmark for cluster health scanner.

    This application runs TPU Host-Device I/O benchmarks utilizing the JAX
    accelerator-microbenchmarks suite.
    """

    name = "tpu-host-device"
    maintainers("simrankaurb")
    tags("diagnostics", "tpu", "host-device")

    variable(
        "run_script",
        default="Ironwood/src/run_benchmark.py",
        description="Path to run_benchmark.py relative to $CHS_AMB_DIR",
    )
    variable(
        "benchmarks_yaml_file",
        default="host_device.yaml",
        description="Filename of the benchmarks config in the benchmarks/ directory",
    )
    variable(
        "benchmarks_yaml_escaped",
        default="",
        description="YAML content for benchmarks configuration (escaped for printf)",
    )
    variable(
        "config_file",
        default="{experiment_run_dir}/chs_tpu_host_device.yaml",
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
            'set -o pipefail; {driver_escape_overrides} {tpu_run_wrapper} python3 -u $\\{CHS_AMB_DIR:-/app/accelerator-microbenchmarks\\}/{run_script} --config {config_file} 2>&1 | sed "s/^/HOST $\\{MY_NODE_NAME:-$(hostname)\\}: /"',
        ],
        use_mpi=False,
    )
    workload("check", executables=["execute"])

    figure_of_merit_context(
        "params",
        regex=r"HOST (?P<node>\S+): Running host_device with params:.*?'data_size_mib':\s*(?P<size>\d+)",
        output_format="{node},{size}MiB",
    )

    figure_of_merit(
        "H2D Bandwidth",
        fom_regex=_H2D_MEDIAN_REGEX,
        group_name="h2d_bandwidth",
        units="GiB/s",
        contexts=["params"],
    )

    figure_of_merit(
        "D2H Bandwidth",
        fom_regex=_D2H_MEDIAN_REGEX,
        group_name="d2h_bandwidth",
        units="GiB/s",
        contexts=["params"],
    )
