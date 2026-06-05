# Copyright 2022-2026 The Ramble Authors
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

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
        default="host_device.yaml",
        description="Filename of the benchmarks config in configs/benchmarks/ directory",
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
            "{driver_escape_overrides} {tpu_run_wrapper} python3 -u {container_app_dir}/{run_script} --config {config_file} 2>&1",
        ],
        use_mpi=False,
    )
    workload("check", executables=["execute"])

    figure_of_merit_context(
        "params",
        regex=r"Running .* with params:.*'data_size_mib': (?P<size>\d+)",
        output_format="{size}MiB",
    )

    figure_of_merit(
        "H2D Bandwidth",
        fom_regex=r".*(?:Host-to-Device|H2D).*?:\s*(?P<h2d_bandwidth>\d+\.\d+)",
        group_name="h2d_bandwidth",
        units="GB/s",
        contexts=["params"],
    )

    figure_of_merit(
        "D2H Bandwidth",
        fom_regex=r".*(?:Device-to-Host|D2H).*?:\s*(?P<d2h_bandwidth>\d+\.\d+)",
        group_name="d2h_bandwidth",
        units="GB/s",
        contexts=["params"],
    )
