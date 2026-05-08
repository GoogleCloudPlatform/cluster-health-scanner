# Copyright 2022-2026 The Ramble Authors
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

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
        default="hbm_copy_bf16.yaml",
        description="Filename of the benchmarks config in configs/benchmarks/ directory",
    )
    variable(
        "benchmarks_yaml_escaped",
        default="",
        description="YAML content for benchmarks configuration (escaped for printf)",
    )
    variable(
        "config_file",
        default="Ironwood/configs/chs_tpu_hbm.yaml",
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
        "node_params",
        regex=r"^\[(?P<node>[a-zA-Z0-9-]+)\]\s*Running .* with params:.*'num_elements': (?P<num_elements>\d+),.*'dtype': <class 'jax.numpy.(?P<dtype>\w+)'>",
        output_format="{node}-{num_elements}-{dtype}",
    )

    figure_of_merit(
        "Bandwidth (median)",
        fom_regex=r"^\[(?P<node>[a-zA-Z0-9-]+)\]\s*.*bandwidth.*?:\s*(?P<bandwidth>\d+\.\d+)",
        group_name="bandwidth",
        units="GB/s",
        contexts=["node_params"],
    )

    figure_of_merit(
        "Time Taken (median)",
        fom_regex=r"^\[(?P<node>[a-zA-Z0-9-]+)\]\s*.*?time taken.*?:\s*(?P<time>\d+\.\d+)",
        group_name="time",
        units="ms",
        fom_type=FomType.INFO,
        contexts=["node_params"],
    )

    figure_of_merit(
        "Tensor Size",
        fom_regex=r"^\[(?P<node>[a-zA-Z0-9-]+)\]\s*Tensor size:\s*(?P<tensor_size>\d+\.\d+)\s*MB",
        group_name="tensor_size",
        units="MB",
        fom_type=FomType.INFO,
        contexts=["node_params"],
    )
