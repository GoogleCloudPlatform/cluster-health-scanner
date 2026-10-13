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


class Gemm(ExecutableApplication):
    """NVIDIA GEMM Benchmark application

    This application runs the NVIDIA GEMM benchmark using gemm_tests.sh
    """

    name = "gemm"
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
