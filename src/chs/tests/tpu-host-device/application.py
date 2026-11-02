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


class TpuHostDevice(ExecutableApplication):
    """TPU JAX Host-Device I/O benchmark for cluster health scanner.

    This application runs TPU Host-Device I/O benchmarks utilizing the JAX
    accelerator-microbenchmarks suite.
    """

    name = "tpu-host-device"
    maintainers("simrankaurb")
    tags("diagnostics", "tpu", "host-device")

    variable(
        "benchmarks_yaml_file",
        default="host_to_device.yaml",
        description="Filename of the host_to_device config in the benchmarks/ directory",
    )
    variable(
        "benchmarks_yaml_escaped",
        default="",
        description="YAML content for host_to_device configuration (escaped for printf)",
    )
    variable(
        "benchmarks_yaml_d2h_file",
        default="device_to_host.yaml",
        description="Filename of the device_to_host config in the benchmarks/ directory",
    )
    variable(
        "benchmarks_yaml_d2h_escaped",
        default="",
        description="YAML content for device_to_host configuration (escaped for printf)",
    )
    variable(
        "config_file",
        default="{experiment_run_dir}/chs_tpu_host_to_device.yaml",
        description="Path to the host_to_device config file",
    )
    variable(
        "config_file_d2h",
        default="{experiment_run_dir}/chs_tpu_device_to_host.yaml",
        description="Path to the device_to_host config file",
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
            "printf '%b' '{benchmarks_yaml_d2h_escaped}' | tee {config_file_d2h}",
            'set -o pipefail; {driver_escape_overrides} {tpu_run_wrapper} env PYTHONUNBUFFERED=1 tpums benchmark run-config {config_file} 2>&1 | sed "s/^/HOST $\\{MY_NODE_NAME:-$(hostname)\\}: /"',
            'set -o pipefail; {driver_escape_overrides} {tpu_run_wrapper} env PYTHONUNBUFFERED=1 tpums benchmark run-config {config_file_d2h} 2>&1 | sed "s/^/HOST $\\{MY_NODE_NAME:-$(hostname)\\}: /"',
        ],
        use_mpi=False,
    )
    workload("check", executables=["execute"])

    figure_of_merit_context(
        "params",
        regex=r"HOST (?P<node>\S+): Running \[\d+/\d+\] (?:host_to_device|device_to_host) \(\w+_dev\w+_size_(?P<size_mib>\d+)mib\)",
        output_format="{node},{size_mib}MiB",
    )

    figure_of_merit(
        "H2D Bandwidth",
        fom_regex=r".*Success \[\d+/\d+\] host_to_device\. Metrics: .*'xprof_bandwidth_per_host_gb_s': (?:np\.float64\()?(?P<h2d_bandwidth>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?",
        group_name="h2d_bandwidth",
        units="GB/s",
        contexts=["params"],
    )

    figure_of_merit(
        "D2H Bandwidth",
        fom_regex=r".*Success \[\d+/\d+\] device_to_host\. Metrics: .*'xprof_bandwidth_per_host_gb_s': (?:np\.float64\()?(?P<d2h_bandwidth>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?",
        group_name="d2h_bandwidth",
        units="GB/s",
        contexts=["params"],
    )
