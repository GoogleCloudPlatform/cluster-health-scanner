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


class DcgmFom(BasicModifier):
    """Modifier to extract GPU-specific Figure of Merits for DCGM."""

    name = "dcgm-fom"

    tags("gpu", "dcgm")

    maintainers("simrankaurb")

    mode("standard", description="Standard extraction mode")
    default_mode("standard")

    figure_of_merit_context(
        "test_name",
        regex=r"^\|\s*(?P<diag_test>[\w\s\/]+?)\s*\|\s*(?P<result>Pass|Fail)\b",
        output_format="{diag_test}",
    )
    figure_of_merit(
        "Result",
        fom_regex=r"^\|\s*(?P<diag_test>[\w\s\/]+?)\s*\|\s*(?P<result>Pass|Fail)\b",
        group_name="result",
        units="",
        contexts=["test_name"],
    )
    figure_of_merit(
        "GPU{gpu_id}",
        fom_regex=r"^\|\s*\|\s*GPU(?P<gpu_id>\d+):\s*(?P<gpu_result>Pass|Fail)\b",
        group_name="gpu_result",
        units="",
        contexts=["test_name"],
    )

    figure_of_merit(
        "DCGM Version",
        fom_regex=r"^\|\s*DCGM Version\s*\|\s*(?P<dcgm_version>[0-9.]+)\b.*",
        group_name="dcgm_version",
        units="",
        fom_type=FomType.INFO,
    )
    figure_of_merit(
        "Driver Version Detected",
        fom_regex=r"^\|\s*Driver Version Detected\s*\|\s*(?P<driver_version>[0-9.]+)\b.*",
        group_name="driver_version",
        units="",
        fom_type=FomType.INFO,
    )
