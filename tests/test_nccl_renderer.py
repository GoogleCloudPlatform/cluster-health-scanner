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

from chs.plugins.nccl_cluster.renderer import NcclClusterRenderer
from chs.plugins.nccl_pairwise.renderer import NcclPairwiseRenderer
from chs.protos import bill_of_health_pb2 as pb
from rich.console import Console


def test_nccl_cluster_renderer_without_errors():
    renderer = NcclClusterRenderer()

    # Create mock result
    res = pb.TestResult()
    res.test_name = "nccl-cluster"
    res.test_level = "L1"

    # Add peak metric
    m1 = res.metrics.add()
    m1.key = "Bus Bandwidth"
    m1.value.number_value = 120.5
    m1.metadata.fields["context"].string_value = str(1024**3)  # 1 GB

    console = Console(width=200)  # Large width to avoid wrapping
    with console.capture() as capture:
        detail = renderer.render_detail([res])
        console.print(detail)

    output = capture.get()
    print(f"DEBUG OUTPUT:\n{output}")

    assert "NCCL Cluster Detail" in output
    assert "Peak Detected Cluster Bandwidth" in output
    assert "120.50" in output
    assert "Detected Errors/Warnings" not in output


def test_nccl_pairwise_renderer():
    renderer = NcclPairwiseRenderer()

    # Create mock result
    res = pb.TestResult()
    res.test_name = "nccl-pairwise"
    res.test_level = "L1"
    res.target_resources.add(node_name="node-1")
    res.target_resources.add(node_name="node-2")

    # Add peak metric
    m1 = res.metrics.add()
    m1.key = "Bus Bandwidth"
    m1.value.number_value = 60.2
    m1.metadata.fields["context"].string_value = str(1024**3)  # 1 GB

    console = Console(width=200)  # Large width to avoid wrapping
    with console.capture() as capture:
        detail = renderer.render_detail([res])
        console.print(detail)

    output = capture.get()
    print(f"DEBUG OUTPUT:\n{output}")

    assert "Pairwise Detail" in output
    assert "node-1 <-> node-2" in output
    assert "60.20" in output


def test_build_node_to_location_and_get_topology(caplog):
    from chs.utils.topology import (
        build_node_to_location,
        get_topology,
        get_topological_context,
    )
    import logging

    # Test empty / None / invalid topology with debug log
    with caplog.at_level(logging.DEBUG):
        loc_empty = build_node_to_location(None)
        assert loc_empty == {}
        assert "Topology is missing or not a dictionary" in caplog.text

        caplog.clear()
        loc_invalid = build_node_to_location(
            "invalid"
        )  # pyrefly: ignore[bad-argument-type]
        assert loc_invalid == {}
        assert "Topology is missing or not a dictionary" in caplog.text

    # Test topology mapping
    topology = {
        "block1": {
            "subblock1": ["node1", "node2"],
            "subblock2": ["node3"],
        },
        "block2": {
            "subblock3": ["node4"],
        },
    }
    loc_map = build_node_to_location(topology)
    assert loc_map["node1"] == {"block": "block1", "subblock": "subblock1"}
    assert loc_map["node2"] == {"block": "block1", "subblock": "subblock1"}
    assert loc_map["node3"] == {"block": "block1", "subblock": "subblock2"}
    assert loc_map["node4"] == {"block": "block2", "subblock": "subblock3"}

    # Test topological context & get_topology
    assert (
        get_topological_context("node1", "node2", loc_map) == "Intra-Subblock"
    )
    assert get_topology("node1", "node2", loc_map) == "intradomain"

    assert (
        get_topological_context("node1", "node3", loc_map) == "Inter-Subblock"
    )
    assert get_topology("node1", "node3", loc_map) == "interdomain"

    assert get_topological_context("node1", "node4", loc_map) == "Inter-Block"
    assert get_topology("node1", "node4", loc_map) == "interdomain"

    assert (
        get_topological_context("node1", "unknown_node", loc_map) == "Unknown"
    )
    assert get_topology("node1", "unknown_node", loc_map) == "unknown"
