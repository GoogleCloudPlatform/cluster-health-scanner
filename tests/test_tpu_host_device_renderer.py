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

from chs.tests.tpu_host_device.renderer import TpuHostDeviceRenderer
from chs.protos import bill_of_health_pb2 as pb
from rich.console import Console
import os
import unittest

os.environ["TERM"] = "xterm-256color"


class TestTpuHostDeviceRenderer(unittest.TestCase):
    def test_render_detail(self):
        renderer = TpuHostDeviceRenderer()

        res = pb.TestResult()
        res.test_name = "tpu-host-device"
        res.test_level = "host_device"
        res.target_resources.add(node_name="node-1")

        m1 = res.metrics.add()
        m1.key = "H2D Bandwidth"
        m1.value.number_value = 15.6
        m1.metadata.fields["context"].string_value = "node-1,1024MiB"

        m2 = res.metrics.add()
        m2.key = "D2H Bandwidth"
        m2.value.number_value = 16.2
        m2.metadata.fields["context"].string_value = "node-1,1024MiB"

        console = Console(width=200)
        with console.capture() as capture:
            detail = renderer.render_detail([res])
            console.print(detail)

        output = capture.get()
        print(f"DEBUG OUTPUT:\n{output}")

        self.assertIn("Host-Device I/O Bandwidth Detail", output)
        self.assertIn("node-1", output)
        self.assertIn("1024", output)
        self.assertIn("15.60", output)
        self.assertIn("16.20", output)
        self.assertIn("Node", output)
        self.assertIn("H2D Bandwidth (GB/s)", output)
        self.assertIn("D2H Bandwidth (GB/s)", output)

    def test_get_detail_data_single_table_for_multiple_nodes(self):
        renderer = TpuHostDeviceRenderer()

        results = []
        for idx, (h2d, d2h) in enumerate(
            [(113.1, 117.5), (113.2, 117.4)], start=1
        ):
            node = f"node-{idx}"
            res = pb.TestResult()
            res.test_name = "tpu-host-device"
            res.test_level = "host_device"
            res.target_resources.add(node_name=node)

            m1 = res.metrics.add()
            m1.key = "H2D Bandwidth"
            m1.value.number_value = h2d
            m1.metadata.fields["context"].string_value = f"{node},256MiB"

            m2 = res.metrics.add()
            m2.key = "D2H Bandwidth"
            m2.value.number_value = d2h
            m2.metadata.fields["context"].string_value = f"{node},256MiB"
            results.append(res)

        report = renderer.get_detail_data(results)
        self.assertIsNotNone(report)
        self.assertEqual(len(report.tables), 1)
        self.assertEqual(
            report.tables[0].title, "Host-Device I/O Bandwidth Detail"
        )
        self.assertEqual(len(report.tables[0].rows), 2)
        self.assertEqual(report.tables[0].rows[0]["node"], "node-1")
        self.assertEqual(report.tables[0].rows[1]["node"], "node-2")


if __name__ == "__main__":
    unittest.main()
