# Copyright 2022-2026 The Ramble Authors
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

from ramble.modkit import *


class NcclErrorDetector(BasicModifier):
    """Modifier to detect common NCCL errors in logs."""

    name = "nccl-error-detector"

    tags("gpu", "nccl")

    maintainers("sarthakag")

    mode("standard", description="Standard error detection mode")
    default_mode("standard")

    # Detect hard failures
    figure_of_merit(
        "Network Error Detected",
        fom_regex=r"(?i)(?P<network_error>.*(network error|remote error|Connection refused|Connection timed out).*)",
        group_name="network_error",
        units="",
        fom_type=FomType.INFO,
    )

    figure_of_merit(
        "CUDA Error Detected",
        fom_regex=r"(?i)(?P<cuda_error>.*(cudaError|unhandled cuda error).*)",
        group_name="cuda_error",
        units="",
        fom_type=FomType.INFO,
    )

    figure_of_merit(
        "Internal Error Detected",
        fom_regex=r"(?i)(?P<internal_error>.*(system error|internal error|invalid argument|invalid usage).*)",
        group_name="internal_error",
        units="",
        fom_type=FomType.INFO,
    )

    figure_of_merit(
        "Topology Warning Detected",
        fom_regex=r"(?i)(?P<topo_warn>.*(Topology detection failed|via PXB|via PHB|via SYS).*)",
        group_name="topo_warn",
        units="",
        fom_type=FomType.INFO,
    )

    figure_of_merit(
        "Other NCCL Error Detected",
        fom_regex=r"(?P<other_error>.*NCCL.*(ERROR|FATAL).*)",
        group_name="other_error",
        units="",
        fom_type=FomType.INFO,
    )
