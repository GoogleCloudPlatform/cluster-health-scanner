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

from ramble.modkit import *


class GenericErrorDetector(BasicModifier):
    """Modifier to detect common generic errors in logs."""

    name = "generic-error-detector"

    tags("gpu", "generic")

    maintainers("sarthakag")

    mode("standard", description="Standard error detection mode")
    default_mode("standard")

    figure_of_merit(
        "CUDA Error Detected",
        fom_regex=r"(?i)(?P<cuda_error>.*(cudaError|CUDA error|Cuda failure).*)",
        group_name="cuda_error",
        units="",
        fom_type=FomType.INFO,
    )

    figure_of_merit(
        "Hardware Error Detected",
        fom_regex=r"(?i)(?P<hardware_error>.*(Xid|Uncorrectable\s+ECC|ECC\s+fail|ECC_ERROR|fallen off the bus).*)",
        group_name="hardware_error",
        units="",
        fom_type=FomType.INFO,
    )

    figure_of_merit(
        "System Error Detected",
        fom_regex=r"(?i)(?P<system_error>.*(Segmentation fault|Segfault|core dumped|Bus error).*)",
        group_name="system_error",
        units="",
        fom_type=FomType.INFO,
    )

    figure_of_merit(
        "Host OOM Error Detected",
        fom_regex=r"(?i)(?P<host_oom>.*(Out of memory|Killed).*)",
        group_name="host_oom",
        units="",
        fom_type=FomType.INFO,
    )

    figure_of_merit(
        "MPI Error Detected",
        fom_regex=r"(?i)(?P<mpi_error>.*(MPI_Abort|Fatal error in|primary job terminated).*)",
        group_name="mpi_error",
        units="",
        fom_type=FomType.INFO,
    )

    figure_of_merit(
        "Timeout Error Detected",
        fom_regex=r"(?i)(?P<timeout>.*(Time limit exceeded|watchdog timeout|timed out).*)",
        group_name="timeout",
        units="",
        fom_type=FomType.INFO,
    )
