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
        default="Ironwood/configs/chs_tpu_host_device.yaml",
        description="Path to config file",
    )

    executable(
        "execute",
        template=[
            "cd {run_dir} && \\",
            'printf "%b" "{benchmarks_yaml_escaped}" > {config_file} && \\',
            'python3 {run_script} --config {config_file} 2>&1 | sed "s/^/[$MY_NODE_NAME] /"',
        ],
        use_mpi=False,
    )
    workload("check", executables=["execute"])

    figure_of_merit_context(
        "params",
        regex=r"^\[(?P<node>[a-zA-Z0-9-]+)\]\s*Running .* with params:.*'data_size_mib': (?P<size>\d+)",
        output_format="{node},{size}MiB",
    )

    figure_of_merit(
        "H2D Bandwidth",
        fom_regex=r"^\[(?P<node>[a-zA-Z0-9-]+)\]\s*.*(?:Host-to-Device|H2D).*?:\s*(?P<h2d_bandwidth>\d+\.\d+)",
        group_name="h2d_bandwidth",
        units="GB/s",
        contexts=["params"],
    )

    figure_of_merit(
        "D2H Bandwidth",
        fom_regex=r"^\[(?P<node>[a-zA-Z0-9-]+)\]\s*.*(?:Device-to-Host|D2H).*?:\s*(?P<d2h_bandwidth>\d+\.\d+)",
        group_name="d2h_bandwidth",
        units="GB/s",
        contexts=["params"],
    )
