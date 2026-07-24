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

from chs.plugins.tpu_collectives.renderer import TpuCollectivesRenderer
from chs.plugins.tpu_collectives_single_node.renderer import (
    TpuCollectivesSingleNodeRenderer,
)
from chs.protos import bill_of_health_pb2 as pb
from rich.console import Console
import os
import unittest

os.environ["TERM"] = "xterm-256color"


class TestTpuCollectivesRenderer(unittest.TestCase):
    def test_render_detail_all_gather(self):
        """Verifies rendering of all_gather workload with 2-part context."""
        renderer = TpuCollectivesRenderer()

        # 1. Single node all_gather
        res1 = pb.TestResult()
        res1.test_name = "tpu-collectives"
        res1.test_level = "all_gather"
        res1.target_resources.add(node_name="node-1")
        m1 = res1.metrics.add()
        m1.key = "Bandwidth (median)"
        m1.value.number_value = 7.92
        m1.metadata.fields["context"].string_value = "node-1,8,float32"

        # 2. Multi-node all_gather
        res2 = pb.TestResult()
        res2.test_name = "tpu-collectives"
        res2.test_level = "all_gather"
        res2.target_resources.add(node_name="node-1")
        res2.target_resources.add(node_name="node-2")
        m2 = res2.metrics.add()
        m2.key = "Bandwidth (median)"
        m2.value.number_value = 111.01
        m2.metadata.fields["context"].string_value = "node-1,1024,float32"
        m3 = res2.metrics.add()
        m3.key = "Bandwidth (median)"
        m3.value.number_value = 111.01
        m3.metadata.fields["context"].string_value = "node-2,1024,float32"

        console = Console(width=200)
        with console.capture() as capture:
            detail = renderer.render_detail([res1, res2])
            console.print(detail)

        output = capture.get()
        print(f"DEBUG OUTPUT (all_gather):\n{output}")

        self.assertIn("TPU Collectives Performance Summary", output)
        self.assertIn(
            "TPU Collectives (all_gather) Bandwidth Detail (node-1, node-2)",
            output,
        )
        self.assertIn("all_gather", output)
        self.assertIn("8", output)
        self.assertIn("float32", output)
        self.assertIn("7.92", output)
        self.assertIn("1024", output)
        self.assertIn("111.01", output)
        self.assertNotIn("Time Taken", output)
        self.assertNotIn("N/A", output)

    def test_render_detail_single_node_renderer(self):
        """Verifies single node renderer title and multiple nodes rendering."""
        renderer = TpuCollectivesSingleNodeRenderer()

        res1 = pb.TestResult()
        res1.test_name = "tpu-collectives-single-node"
        res1.test_level = "all_gather"
        res1.target_resources.add(node_name="node-1")
        res1.target_resources.add(node_name="node-2")
        m1 = res1.metrics.add()
        m1.key = "Bandwidth (median)"
        m1.value.number_value = 7.92
        m1.metadata.fields["context"].string_value = "node-1,8,float32"
        m2 = res1.metrics.add()
        m2.key = "Bandwidth (median)"
        m2.value.number_value = 8.12
        m2.metadata.fields["context"].string_value = "node-2,8,float32"

        console = Console(width=200)
        with console.capture() as capture:
            detail = renderer.render_detail([res1])
            console.print(detail)

        output = capture.get()
        print(f"DEBUG OUTPUT (single-node):\n{output}")

        self.assertIn("TPU Collectives Single Node Performance Summary", output)
        self.assertIn(
            "TPU Collectives (all_gather) Bandwidth Detail (node-1)", output
        )
        self.assertIn(
            "TPU Collectives (all_gather) Bandwidth Detail (node-2)", output
        )
        self.assertIn("all_gather", output)
        self.assertIn("7.92", output)
        self.assertIn("8.12", output)


if __name__ == "__main__":
    unittest.main()
