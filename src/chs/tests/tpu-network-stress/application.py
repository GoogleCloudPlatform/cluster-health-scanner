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

with open(os.path.join(_APP_DIR, "tpu_network_stress.py"), "r") as f:
    _NETWORK_STRESS_SCRIPT = f.read().strip()

# Base64 encode the script to bypass Ramble's template parser which would otherwise
# try to evaluate Python's `{}` dicts/f-strings as Ramble variables. It also avoids
# shell character escaping issues during injection.
_B64_SCRIPT = base64.b64encode(_NETWORK_STRESS_SCRIPT.encode("utf-8")).decode(
    "utf-8"
)


class TpuNetworkStress(ExecutableApplication):
    """An application to run the check tpu ici network using device to device data transfer."""

    name = "tpu-network-stress"
    maintainers("bytetwin")
    tags("stress", "tpu", "network")

    variable(
        "payload_size_mb",
        default="1024",
        description="Payload size in MB for the network stress test.",
    )
    variable(
        "num_iters",
        default="1000",
        description="Number of iterations for the ppermute data swap.",
    )

    executable(
        "execute",
        template=[
            f"echo '{_B64_SCRIPT}' | base64 -d > tpu_network_stress.py",
            "PAYLOAD_SIZE_MB={payload_size_mb} NUM_ITERATIONS={num_iters} python3 tpu_network_stress.py",
        ],
        use_mpi=False,
    )

    workload("stress", executables=["execute"])

    figure_of_merit_context(
        "host_axis",
        regex=r".*Host: (?P<host>[a-zA-Z0-9_-]+) \| Axis: (?P<axis>[XYZ]) \|.*",
        output_format="{host} Axis-{axis}",
    )

    figure_of_merit(
        "bandwidth",
        fom_regex=r".*Host: .* \| Axis: [XYZ] \| (?P<bandwidth>[0-9.]+) GB/s.*",
        group_name="bandwidth",
        units="GB/s",
        contexts=["host_axis"],
    )
