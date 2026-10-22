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

import jax
import importlib.metadata as meta
import os
import socket
import warnings

for k in list(os.environ.keys()):
    if "MPI" in k or "RANK" in k or "WORLD" in k or "JAX_COORDINATOR" in k:
        os.environ.pop(k)

warnings.filterwarnings("ignore")

node_name = os.environ.get("MY_NODE_NAME", socket.gethostname())
local_devs = jax.local_devices()

print("HOST %s: JAX Devices: %d" % (node_name, len(local_devs)))
print(
    "HOST %s: Host Info: %s %s"
    % (node_name, os.uname().sysname, os.uname().release)
)

dists = set([d.metadata.get("Name", "") for d in meta.distributions()])
libtpu_ver = (
    meta.version("libtpu-nightly")
    if "libtpu-nightly" in dists
    else (
        meta.version("libtpu")
        if "libtpu" in dists
        else getattr(
            jax.lib.xla_bridge.get_backend(), "platform_version", "N/A"
        )
    )
)

print("HOST %s: Libtpu version: %s" % (node_name, libtpu_ver))
if local_devs:
    print(
        "HOST %s: Accelerator type: %s" % (node_name, local_devs[0].device_kind)
    )
