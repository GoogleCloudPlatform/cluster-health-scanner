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

    executable(
        "execute",
        template=(
            'cd {run_dir} && printf "%b" "{benchmarks_yaml_escaped}" > {config_file} && python3 {run_script} --config {config_file} 2>&1'
        ),
        use_mpi=False,
    )
    workload("all_gather", executables=["execute"])
    workload("psum", executables=["execute"])
    workload("all_to_all", executables=["execute"])

    figure_of_merit_context(
        "params",
        regex=r"Running\s+(?:benchmark:\s+)?(?P<op_type>\w+)\s+with params:.*?['\"]matrix_dim['\"]:\s*(?P<matrix_dim>\d+).*?['\"]dtype['\"]:\s*<class\s+['\"]jax\.numpy\.(?P<dtype>\w+)['\"]>",
        output_format="{matrix_dim},{dtype}",
    )

    figure_of_merit(
        "Bandwidth (median)",
        fom_regex=r"metrics:\s*.*?['\"]achieved_bw \(GB/s\)_p50['\"]:\s*(?:np\.float64\()?(?P<bandwidth>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?",
        group_name="bandwidth",
        units="GB/s",
        contexts=["params"],
    )

    figure_of_merit(
        "Time Taken (median)",
        fom_regex=r"metrics:\s*.*?['\"]step_time_ms_p50['\"]:\s*(?:np\.float64\()?(?P<time>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?",
        group_name="time",
        units="ms",
        fom_type=FomType.INFO,
        contexts=["params"],
    )
