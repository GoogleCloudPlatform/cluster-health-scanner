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

"""Unit tests for chs.testkit.tpu."""

import unittest

from chs.testkit import tpu


class MeshParameterTest(unittest.TestCase):
    def test_a_listed_topology_uses_its_published_mesh(self):
        values = {
            "tpu_topology": "2x2x1",
            "cores_per_chip": 2,
            "benchmarks_yaml_escaped": (
                "mesh: __MESH__, sharding: __SHARDING__, ici: __ICI__"
            ),
        }

        tpu.apply_collective_mesh_parameters(values)

        self.assertEqual(
            values["benchmarks_yaml_escaped"],
            "mesh: 2x2x2, sharding: 2x2x2, ici: 8",
        )

    def test_an_unlisted_topology_spreads_cores_along_one_axis(self):
        values = {
            "tpu_topology": "3x5x7",
            "cores_per_chip": 2,
            "benchmarks_yaml_escaped": "__MESH__ __SHARDING__ __ICI__",
        }

        tpu.apply_collective_mesh_parameters(values)

        self.assertEqual(values["benchmarks_yaml_escaped"], "6x5x7 6x5x7 210")

    def test_the_explicit_topology_wins_over_the_assigned_one(self):
        values = {
            "execution_topology": "2x2x1",
            "tpu_topology": "4x4x4",
            "cores_per_chip": 2,
            "benchmarks_yaml_escaped": "__MESH__",
        }

        tpu.apply_collective_mesh_parameters(values)

        self.assertEqual(values["benchmarks_yaml_escaped"], "2x2x2")

    def test_message_sizes_default_to_the_interconnect_width(self):
        values = {
            "tpu_topology": "4x4x4",
            "cores_per_chip": 2,
            "benchmarks_yaml_escaped": "__START__ __END__ __MULTIPLIER__",
        }

        tpu.apply_collective_mesh_parameters(values)

        self.assertEqual(values["benchmarks_yaml_escaped"], "128 16384 2")

    def test_message_size_start_never_drops_below_eight_elements(self):
        values = {
            "tpu_topology": "2x2x1",
            "cores_per_chip": 1,
            "benchmarks_yaml_escaped": "__ICI__ __START__",
        }

        tpu.apply_collective_mesh_parameters(values)

        self.assertEqual(values["benchmarks_yaml_escaped"], "4 8")

    def test_message_sizes_can_be_overridden(self):
        values = {
            "tpu_topology": "2x2x1",
            "cores_per_chip": 2,
            "message_size_start": 64,
            "message_size_end": 128,
            "message_size_multiplier": 4,
            "benchmarks_yaml_escaped": "__START__ __END__ __MULTIPLIER__",
        }

        tpu.apply_collective_mesh_parameters(values)

        self.assertEqual(values["benchmarks_yaml_escaped"], "64 128 4")

    def test_an_unmeasurable_run_is_refused_rather_than_guessed(self):
        for values, message in (
            ({"cores_per_chip": 2}, "topology"),
            ({"tpu_topology": "2x2x1"}, "cores_per_chip"),
            (
                {"tpu_topology": "small", "cores_per_chip": 2},
                "Invalid topology",
            ),
            (
                {"tpu_topology": "2x2x1", "cores_per_chip": "bad"},
                "invalid literal",
            ),
        ):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    tpu.apply_collective_mesh_parameters(dict(values))


if __name__ == "__main__":
    unittest.main()
