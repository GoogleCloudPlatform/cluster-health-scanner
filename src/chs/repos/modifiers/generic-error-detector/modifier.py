# Copyright 2022-2026 The Ramble Authors
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

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
