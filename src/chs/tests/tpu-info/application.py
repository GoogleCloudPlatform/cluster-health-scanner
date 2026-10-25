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

import os
import inspect
import base64
from ramble.appkit import *

_APP_DIR = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)

with open(os.path.join(_APP_DIR, "tpu_inventory.py"), "r") as f:
    _TPU_INVENTORY_SCRIPT = f.read().strip()

_B64_SCRIPT = base64.b64encode(_TPU_INVENTORY_SCRIPT.encode("utf-8")).decode(
    "utf-8"
)


class TpuInfo(ExecutableApplication):
    """An application to gather TPU inventory and diagnostics."""

    name = "tpu-info"
    maintainers("simrankaurb")
    tags("diagnostics", "tpu")

    executable(
        "execute",
        template=[
            f"echo '{_B64_SCRIPT}' | base64 -d > tpu_inventory.py",
            "python3 tpu_inventory.py",
        ],
        use_mpi=True,
    )
    workload("local", executables=["execute"])

    figure_of_merit_context(
        "node",
        regex=r"(?i)HOST (?P<node>\S+):",
        output_format="Node {node}",
    )

    figure_of_merit(
        "JAX Devices",
        fom_regex=r"(?i)HOST [\w.-]+: JAX Devices: (?P<tpu_count>\d+)",
        group_name="tpu_count",
        units="",
        contexts=["node"],
    )

    figure_of_merit(
        "Host Info",
        fom_regex=r"(?i)HOST [\w.-]+: Host Info: (?P<host_info>[^;\n]+)",
        group_name="host_info",
        units="",
        fom_type=FomType.INFO,
        contexts=["node"],
    )

    figure_of_merit(
        "Libtpu version",
        fom_regex=r"(?i)HOST [\w.-]+: Libtpu version: (?P<libtpu_version>[^\n]+)",
        group_name="libtpu_version",
        units="",
        fom_type=FomType.INFO,
        contexts=["node"],
    )

    figure_of_merit(
        "Accelerator type",
        fom_regex=r"(?i)HOST [\w.-]+: Accelerator type: (?P<accelerator_type>[^\n]+)",
        group_name="accelerator_type",
        units="",
        fom_type=FomType.INFO,
        contexts=["node"],
    )
