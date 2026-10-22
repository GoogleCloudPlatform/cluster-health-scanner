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

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "../src/chs/tests/tpu-compute-stress"
    ),
)

# Mock JAX before importing the script
sys.modules["jax"] = MagicMock()
sys.modules["jax.numpy"] = MagicMock()

# Import the module under test
import tpu_compute_stress


class TpuComputeStressTest(unittest.TestCase):
    @patch("tpu_compute_stress.logging.info")
    @patch("tpu_compute_stress.logging.error")
    @patch("tpu_compute_stress.time.time")
    @patch("tpu_compute_stress.np.asarray")
    @patch("tpu_compute_stress.jax.local_devices")
    def test_main_success(
        self, mock_local_devices, mock_asarray, mock_time, mock_error, mock_info
    ):
        mock_local_devices.return_value = ["dev0", "dev1"]
        mock_asarray.return_value = [[1.0, 2.0], [1.0, 2.0]]

        mock_time.side_effect = [10.0, 15.0]  # 5 seconds duration

        # We also need to mock init_payloads and hardware_health_stress behavior,
        # but wait, since jax is mocked, those functions might be MagicMocks.
        # Let's see if we need to explicitly patch them.
        tpu_compute_stress.init_payloads = MagicMock(
            return_value=("A", "W1", "W2")
        )

        mock_hardware_health_diag = MagicMock()
        tpu_compute_stress.hardware_health_stress = mock_hardware_health_diag

        os.environ["MY_NODE_NAME"] = "test-node"

        tpu_compute_stress.main()

        # Verify success logs
        mock_info.assert_any_call("NODE:test-node EVENT:STRESS_PASS")
        mock_error.assert_not_called()

    @patch("tpu_compute_stress.logging.info")
    @patch("tpu_compute_stress.logging.error")
    @patch("tpu_compute_stress.time.time")
    @patch("tpu_compute_stress.np.asarray")
    @patch("tpu_compute_stress.jax.local_devices")
    def test_main_sdc_failure(
        self, mock_local_devices, mock_asarray, mock_time, mock_error, mock_info
    ):
        mock_local_devices.return_value = ["dev0", "dev1"]
        mock_asarray.return_value = [[1.0, 2.0], [1.0, 99.0]]

        mock_time.side_effect = [10.0, 15.0]

        tpu_compute_stress.init_payloads = MagicMock(
            return_value=("A", "W1", "W2")
        )
        mock_hardware_health_diag = MagicMock()
        tpu_compute_stress.hardware_health_stress = mock_hardware_health_diag

        os.environ["MY_NODE_NAME"] = "test-node-sdc"

        tpu_compute_stress.main()

        mock_error.assert_any_call(
            "NODE:test-node-sdc EVENT:STRESS_FAIL. SDC (Bit flips) Detected"
        )


if __name__ == "__main__":
    unittest.main()
