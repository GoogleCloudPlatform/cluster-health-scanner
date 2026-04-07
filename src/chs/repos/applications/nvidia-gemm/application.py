# Copyright 2022-2026 The Ramble Authors
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

from ramble.appkit import *


class NvidiaGemm(ExecutableApplication):
    """NVIDIA GEMM Benchmark application

    This application runs the NVIDIA GEMM benchmark using gemm_tests.sh
    """

    name = "nvidia-gemm"
    maintainers("bytetwin")

    tags("micro-benchmark", "gpu")

    workload_variable(
        "gpu_affinity",
        default="",
        description="GPU affinity mapping",
        workloads=["check", "perf"],
    )
    workload_variable(
        "cpu_affinity",
        default="",
        description="CPU affinity mapping",
        workloads=["check", "perf"],
    )
    workload_variable(
        "mem_affinity",
        default="",
        description="Memory affinity mapping",
        workloads=["check", "perf"],
    )
    workload_variable(
        "type",
        default="fp64",
        description="Set type of GEMM (fp64 / fp64emu)",
        workloads=["check", "perf"],
    )
    workload_variable(
        "m",
        default="65536",
        description="Set m dimension",
        workloads=["check", "perf"],
    )
    workload_variable(
        "n",
        default="16384",
        description="Set n dimension",
        workloads=["check", "perf"],
    )
    workload_variable(
        "k",
        default="1024",
        description="Set k dimension",
        workloads=["check", "perf"],
    )
    workload_variable(
        "iter",
        default="32",
        description="Number of iterations",
        workloads=["check", "perf"],
    )
    workload_variable(
        "warmup",
        default="32",
        description="Warmup iterations",
        workloads=["check", "perf"],
    )

    executable(
        "execute_perf",
        template=[
            "./gemm_tests.sh "
            "--gpu-affinity {gpu_affinity} "
            "--cpu-affinity {cpu_affinity} "
            "--mem-affinity {mem_affinity} "
            '--params "--mode=perf --type={type} --m={m} --n={n} --k={k} --iter={iter} --warmup={warmup}" | tee {experiment_run_dir}/{experiment_name}_local.out',
            'sh -c \'RANK="${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-0}}"; if [ "$RANK" -eq 0 ]; then ./gemm_parse_results.sh --list --stats {experiment_run_dir}/{experiment_name}_local.out; fi\'',
        ],
        use_mpi=True,
    )

    executable(
        "execute_check",
        template=[
            "./gemm_tests.sh "
            "--gpu-affinity {gpu_affinity} "
            "--cpu-affinity {cpu_affinity} "
            "--mem-affinity {mem_affinity} "
            '--params "--mode=check --type={type} --m={m} --n={n} --k={k} --iter={iter}" | tee {experiment_run_dir}/{experiment_name}_local.out',
            'sh -c \'RANK="${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-0}}"; if [ "$RANK" -eq 0 ]; then ./gemm_parse_results.sh --correctness --list --stats {experiment_run_dir}/{experiment_name}_local.out; fi\'',
        ],
        use_mpi=True,
    )

    workload("perf", executables=["execute_perf"])
    workload("check", executables=["execute_check"])

    # Figure of merits parsing
    stats_regex = (
        r"(?P<stat_type>MEDIAN|MEAN|MIN|MAX)\s+=\s+(?P<value>[0-9]+\.[0-9]+)"
    )

    figure_of_merit_context(
        "stats", regex=stats_regex, output_format="{stat_type}"
    )

    figure_of_merit(
        "Value",
        fom_regex=stats_regex,
        group_name="value",
        units="",
        contexts=["stats"],
    )

    correctness_regex = r"^(?P<status>All GEMM tests passed)"

    figure_of_merit(
        "Correctness",
        fom_regex=correctness_regex,
        group_name="status",
        units="",
    )
