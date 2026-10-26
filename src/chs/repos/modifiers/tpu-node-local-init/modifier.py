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


class TpuNodeLocalInit(BasicModifier):
    """Modifier to configure node-local TPU initialization for single-host tests."""

    name = "tpu-node-local-init"

    tags("tpu")

    maintainers("simrankaurb")

    mode("standard", description="Standard node-local TPU initialization mode")
    default_mode("standard")

    register_builtin("node_local_tpu_env", injection_method="prepend")

    def node_local_tpu_env(self):
        return [
            "export TPU_HOST_BOUNDS=1,1,1",
            "export TPU_SKIP_MDS_QUERY=true",
            "export TPU_WORKER_ID=0",
            "export TPU_WORKER_HOSTNAMES=localhost",
            (
                'export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-}'
                " --deepsea_hal_test_skip_slicebuilder=true"
                " --deepsea_hal_test_allow_multichip_skip_slicebuilder=true"
                " --xla_tpu_use_enhanced_launch_barrier=false"
                ' --megascale_error_reporter_abort_on_hang=true"'
            ),
        ]
