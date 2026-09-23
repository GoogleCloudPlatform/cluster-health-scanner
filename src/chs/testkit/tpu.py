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

"""TPU mesh arithmetic shared by the collectives tests.

This lives in the toolkit rather than in a test package because two tests need
it: the multi-node collectives test and its single-node variant. Duplicating it
would put the same mesh arithmetic in two files that must agree.
"""

import logging

from chs.constants import TOPOLOGY_TO_MESH

logger = logging.getLogger(__name__)


def apply_collective_mesh_parameters(variables: dict) -> None:
    topology = variables.get("execution_topology") or variables.get(
        "tpu_topology"
    )
    if not topology:
        raise ValueError(
            "Neither execution_topology nor tpu_topology found in variables"
        )
    if "cores_per_chip" not in variables:
        raise ValueError("cores_per_chip not found in variables")
    cores_per_chip = int(variables["cores_per_chip"])

    dimensions = [int(d) for d in topology.split("x") if d.isdigit()]
    if not dimensions:
        raise ValueError(f"Invalid topology detected {topology}")

    interconnect_width = cores_per_chip
    for dimension in dimensions:
        interconnect_width *= dimension

    mesh_shape = TOPOLOGY_TO_MESH.get(topology)
    if not mesh_shape:
        # Unlisted topologies fall back to spreading the cores of each chip
        # along the first dimension, which is how the listed ones are built.
        scaled = [dimensions[0] * cores_per_chip] + dimensions[1:]
        mesh_shape = "x".join(str(d) for d in scaled)

    content = variables.get("benchmarks_yaml_escaped", "")
    variables["benchmarks_yaml_escaped"] = (
        content.replace("__MESH__", mesh_shape)
        .replace("__SHARDING__", mesh_shape)
        .replace("__ICI__", str(interconnect_width))
        .replace(
            "__START__",
            str(
                variables.get("message_size_start", max(8, interconnect_width))
            ),
        )
        .replace("__END__", str(variables.get("message_size_end", 16384)))
        .replace(
            "__MULTIPLIER__",
            str(variables.get("message_size_multiplier", 2)),
        )
    )
