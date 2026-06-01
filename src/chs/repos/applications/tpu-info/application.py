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


class TpuInfo(ExecutableApplication):
    """TPU Info application for cluster health scanner"""

    name = "tpu-info"
    maintainers("simrankaurb")
    tags("diagnostics", "tpu")

    executable(
        "execute",
        template=[
            "TPU_OUT=$(tpu-info)",
            'echo "$TPU_OUT"',
            'echo "TPU Count: $(echo "$TPU_OUT" | grep -c "chip")"',
            'echo "Host Info: $(uname -a)"',
        ],
        use_mpi=False,
    )
    workload("local", executables=["execute"])

    figure_of_merit(
        "TPU Chip Count",
        fom_regex=r"TPU Count: (?P<tpu_count>\d+)",
        group_name="tpu_count",
        units="",
    )

    figure_of_merit(
        "Host Info",
        fom_regex=r"Host Info: (?P<host_info>[^;\n]+)",
        group_name="host_info",
        units="",
        fom_type=FomType.INFO,
    )

    figure_of_merit(
        "Libtpu version",
        fom_regex=r"Libtpu version: (?P<libtpu_version>[^\n]+)",
        group_name="libtpu_version",
        units="",
        fom_type=FomType.INFO,
    )

    figure_of_merit(
        "Accelerator type",
        fom_regex=r"Accelerator type: (?P<accelerator_type>[^\n]+)",
        group_name="accelerator_type",
        units="",
        fom_type=FomType.INFO,
    )

    figure_of_merit_context(
        "tpu_chip",
        regex=r"\|\s+(?P<chip_path>/dev/vfio/\d+|/dev/accel\d+)\s+\|\s*(?P<chip_type>[^|]+?)\s*\|",
        output_format="Chip {chip_path}",
    )

    figure_of_merit(
        "Chip Type",
        fom_regex=r"\|\s+(?:/dev/vfio/\d+|/dev/accel\d+)\s+\|\s*(?P<chip_type>[^|]+?)\s*\|",
        group_name="chip_type",
        units="",
        fom_type=FomType.INFO,
        contexts=["tpu_chip"],
    )

    figure_of_merit(
        "Chip Devices",
        fom_regex=r"\|\s+(?:/dev/vfio/\d+|/dev/accel\d+)\s+\|\s*[^|]+\s*\|\s*(?P<chip_devices>\d+)\s*\|",
        group_name="chip_devices",
        units="",
        fom_type=FomType.INFO,
        contexts=["tpu_chip"],
    )
