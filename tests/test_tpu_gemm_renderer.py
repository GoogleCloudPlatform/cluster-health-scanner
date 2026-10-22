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

from chs.tests.tpu_gemm.renderer import TpuGemmRenderer
from chs.protos import bill_of_health_pb2 as pb
from rich.console import Console
import os
import unittest

os.environ["TERM"] = "xterm-256color"


class TestTpuGemmRenderer(unittest.TestCase):
    def test_render_detail(self):
        renderer = TpuGemmRenderer()

        res = pb.TestResult()
        res.test_name = "tpu-gemm"
        res.test_level = "gemm"
        res.target_resources.add(node_name="node-1")

        m1 = res.metrics.add()
        m1.key = "Throughput"
        m1.value.number_value = 120.5
        m1.metadata.fields[
            "context"
        ].string_value = "node-1,16384x16384x16384,bfloat16"

        m2 = res.metrics.add()
        m2.key = "Total Throughput"
        m2.value.number_value = 482.0
        m2.metadata.fields[
            "context"
        ].string_value = "node-1,16384x16384x16384,bfloat16"

        m3 = res.metrics.add()
        m3.key = "MFU"
        m3.value.number_value = 65.2
        m3.metadata.fields[
            "context"
        ].string_value = "node-1,16384x16384x16384,bfloat16"

        m4 = res.metrics.add()
        m4.key = "Step Time"
        m4.value.number_value = 0.05
        m4.metadata.fields[
            "context"
        ].string_value = "node-1,16384x16384x16384,bfloat16"

        m5 = res.metrics.add()
        m5.key = "Total FLOPs"
        m5.value.number_value = 8.79e12
        m5.metadata.fields[
            "context"
        ].string_value = "node-1,16384x16384x16384,bfloat16"

        console = Console(width=200)
        with console.capture() as capture:
            detail = renderer.render_detail([res])
            console.print(detail)

        output = capture.get()
        print(f"DEBUG OUTPUT:\n{output}")

        self.assertIn("TPU GEMM Performance Summary", output)
        self.assertIn("16384x16384x16384,bfloat16", output)
        self.assertIn("node-1", output)
        self.assertIn("120.50", output)
        self.assertIn("482.00", output)
        self.assertIn("65.2%", output)
        self.assertIn("0.05", output)
        self.assertIn("8.79e+12", output)


if __name__ == "__main__":
    unittest.main()
