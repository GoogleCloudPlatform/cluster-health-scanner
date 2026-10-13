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

with open(os.path.join(_APP_DIR, "tpu_hbm_stress.py"), "r") as f:
    _HBM_STRESS_SCRIPT = f.read().strip()

_B64_SCRIPT = base64.b64encode(_HBM_STRESS_SCRIPT.encode("utf-8")).decode(
    "utf-8"
)


class TpuHbmStress(ExecutableApplication):
    """An application to run the check tpu hbm using memory operations."""

    name = "tpu-hbm-stress"
    tags("stress", "tpu", "hbm")

    executable(
        "execute",
        template=[
            "echo '" + _B64_SCRIPT + "' | base64 -d > tpu_hbm_stress.py",
            "ACCELERATOR_TYPE={machine_type} python3 tpu_hbm_stress.py",
        ],
        use_mpi=False,
    )

    workload("stress", executables=["execute"])

    figure_of_merit_context(
        "node",
        regex=r".*NODE:(?P<node>[a-zA-Z0-9_-]+) EVENT:MEASURE_END.*",
        output_format="{node}",
    )

    figure_of_merit(
        "bandwidth",
        fom_regex=r".*NODE:.* EVENT:MEASURE_END DURATION:.*s BANDWIDTH_GBPS:(?P<bandwidth>[0-9.]+)",
        group_name="bandwidth",
        units="GB/s",
        contexts=["node"],
    )
