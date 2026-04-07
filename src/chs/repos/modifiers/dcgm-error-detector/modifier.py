# Copyright 2022-2026 The Ramble Authors
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

from ramble.modkit import *


class DcgmErrorDetector(BasicModifier):
    """Modifier to detect common DCGM errors in logs.

    Example logs:
    - NVML: Unable to retrieve NVLink information as all links are inActive
    - DCGM_ST_IN_USE
    """

    name = "dcgm-error-detector"

    tags("gpu", "dcgm")

    maintainers("sarthakag")

    mode("standard", description="Standard error detection mode")
    default_mode("standard")

    # Detect resource contention
    figure_of_merit(
        "DCGM Resource Contention Error Detected",
        fom_regex=r"(?i)(?P<dcgm_contention>.*(DCGM_ST_IN_USE|DCGM_ST_BAD_PARAMETER).*)",
        group_name="dcgm_contention",
        units="",
        fom_type=FomType.INFO,
    )

    # Detect NVML specific errors reported by DCGM
    figure_of_merit(
        "NVML Error Detected",
        fom_regex=r"(?i)(?P<nvml_error>.*(Failed to initialize NVML|NVML: Unable to retrieve|NVML error|NVML failure).*)",
        group_name="nvml_error",
        units="",
        fom_type=FomType.INFO,
    )

    # Detect page retirement errors
    figure_of_merit(
        "DCGM Page Retirement Error Detected",
        fom_regex=r"(?i)(?P<page_retirement>.*maximum number of retired pages.*)",
        group_name="page_retirement",
        units="",
        fom_type=FomType.INFO,
    )

    # Detect ECC errors
    figure_of_merit(
        "DCGM ECC Error Detected",
        fom_regex=r"(?i)(?P<ecc_error>.*(ECC error|Double-bit error|uncorrectable error).*)",
        group_name="ecc_error",
        units="",
        fom_type=FomType.INFO,
    )

    # Detect PCIe integrity errors
    figure_of_merit(
        "DCGM PCIe Error Detected",
        fom_regex=r"(?i)(?P<pcie_error>.*(PCIe link width.*downgraded|PCIe throughput.*lower than expected).*)",
        group_name="pcie_error",
        units="",
        fom_type=FomType.INFO,
    )

    # Detect thermal errors
    figure_of_merit(
        "DCGM Thermal Error Detected",
        fom_regex=r"(?i)(?P<thermal_error>.*(thermal limit|thermal throttling).*)",
        group_name="thermal_error",
        units="",
        fom_type=FomType.INFO,
    )

    # Detect power errors
    figure_of_merit(
        "DCGM Power Error Detected",
        fom_regex=r"(?i)(?P<power_error>.*(cannot maintain.*power draw|power limit reached).*)",
        group_name="power_error",
        units="",
        fom_type=FomType.INFO,
    )

    # Detect NVLink specific errors (CRC/Link down)
    figure_of_merit(
        "DCGM NVLink Error Detected",
        fom_regex=r"(?i)(?P<nvlink_error>.*(CRC Error|link down).*)",
        group_name="nvlink_error",
        units="",
        fom_type=FomType.INFO,
    )
