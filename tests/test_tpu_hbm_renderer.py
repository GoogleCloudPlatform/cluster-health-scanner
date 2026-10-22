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

from chs.tests.tpu_hbm.renderer import TpuHbmRenderer
from chs.protos import bill_of_health_pb2 as pb
from rich.console import Console
import os
import unittest

os.environ["TERM"] = "xterm-256color"


class TestTpuHbmRenderer(unittest.TestCase):
    def test_render_detail(self):
        renderer = TpuHbmRenderer()

        res = pb.TestResult()
        res.test_name = "tpu-hbm"
        res.test_level = "hbm"
        res.target_resources.add(node_name="node-1")

        m1 = res.metrics.add()
        m1.key = "Bandwidth (median)"
        m1.value.number_value = 280.5
        m1.metadata.fields["context"].string_value = "node-1,134217728,float32"

        m2 = res.metrics.add()
        m2.key = "Time Taken (median)"
        m2.value.number_value = 1.91
        m2.metadata.fields["context"].string_value = "node-1,134217728,float32"

        m3 = res.metrics.add()
        m3.key = "Tensor Size"
        m3.value.number_value = 512.0
        m3.metadata.fields["context"].string_value = "node-1,134217728,float32"

        console = Console(width=200)
        with console.capture() as capture:
            detail = renderer.render_detail([res])
            console.print(detail)

        output = capture.get()
        print(f"DEBUG OUTPUT:\n{output}")

        self.assertIn("TPU HBM Bandwidth Detail (node-1)", output)
        self.assertIn("134,217,728", output)  # formatted with commas
        self.assertIn("512.0", output)
        self.assertIn("float32", output)
        self.assertIn("280.50", output)
        self.assertIn("1.910", output)  # formatted with 3 decimals


if __name__ == "__main__":
    unittest.main()
