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

from tpu_network_diagnostics import (
    build_permutation_pairs,
    calculate_bandwidth,
)
import unittest
import sys
from unittest.mock import MagicMock

sys.modules["jax"] = MagicMock()
sys.modules["jax.numpy"] = MagicMock()
sys.modules["jax.sharding"] = MagicMock()
sys.modules["jax.experimental"] = MagicMock()
sys.modules["jax.experimental.multihost_utils"] = MagicMock()


class TpuNetworkDiagnosticsTest(unittest.TestCase):
    def test_build_permutation_pairs_even_no_wrap(self):
        # 2x2x4 mesh, axis size 4, even pass
        pairs = build_permutation_pairs(4, False, (2, 2, 4))
        self.assertEqual(pairs, [(0, 1), (1, 0), (2, 3), (3, 2)])

    def test_build_permutation_pairs_odd_no_wrap(self):
        # 2x2x4 mesh, axis size 4, odd pass (should not wrap 3 to 0 because 2x2x4 is not all multiples of 4)
        pairs = build_permutation_pairs(4, True, (2, 2, 4))
        self.assertEqual(pairs, [(1, 2), (2, 1)])

    def test_build_permutation_pairs_even_with_wrap(self):
        # 4x4x4 mesh, axis size 4, even pass
        pairs = build_permutation_pairs(4, False, (4, 4, 4))
        self.assertEqual(pairs, [(0, 1), (1, 0), (2, 3), (3, 2)])

    def test_build_permutation_pairs_odd_with_wrap(self):
        # 4x4x4 mesh, axis size 4, odd pass (should wrap 3 to 0 because all dims % 4 == 0)
        pairs = build_permutation_pairs(4, True, (4, 4, 4))
        self.assertEqual(pairs, [(1, 2), (2, 1), (3, 0), (0, 3)])

    def test_calculate_bandwidth(self):
        # 1 GB payload, 1 core, 1 second duration
        # bandwidth should be 1000 * 1 * 2 / 1 = 2000 GB/s
        payload_nbytes = 1024**3
        bw = calculate_bandwidth(payload_nbytes, 1, 1.0)
        self.assertEqual(bw, 2000.0)

        # 512 MB payload, 2 cores, 0.5 seconds
        # 1000 * (0.5 GB) * 2 * 2 / 0.5 = 4000 GB/s

        payload_nbytes = 512 * 1024**2
        bw = calculate_bandwidth(payload_nbytes, 2, 0.5)
        self.assertEqual(bw, 4000.0)

    def test_build_permutation_pairs_small_mesh(self):
        # 2x2x2 mesh
        self.assertEqual(
            build_permutation_pairs(2, False, (2, 2, 2)), [(0, 1), (1, 0)]
        )
        self.assertEqual(build_permutation_pairs(2, True, (2, 2, 2)), [])

    def test_build_permutation_pairs_axis_size_8(self):
        # 4x4x8 mesh
        pairs = build_permutation_pairs(8, False, (4, 4, 8))
        self.assertEqual(
            pairs,
            [(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4), (6, 7), (7, 6)],
        )

        pairs = build_permutation_pairs(8, True, (4, 4, 8))
        self.assertEqual(
            pairs,
            [(1, 2), (2, 1), (3, 4), (4, 3), (5, 6), (6, 5), (7, 0), (0, 7)],
        )


if __name__ == "__main__":
    unittest.main()
