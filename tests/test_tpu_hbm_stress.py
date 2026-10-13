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

import unittest
import sys
import os
import re
import importlib.util
from unittest.mock import MagicMock, patch

# Mock jax and jax.numpy
mock_jax = MagicMock()
mock_jnp = MagicMock()
sys.modules["jax"] = mock_jax
sys.modules["jax.numpy"] = mock_jnp


class TestTpuHbmStressRegex(unittest.TestCase):
    def setUp(self):
        # FOM Regexes from application.py
        self.node_regex = re.compile(
            r"NODE:(?P<node>[a-zA-Z0-9_-]+) EVENT:MEASURE_END.*"
        )
        self.fom_regex = re.compile(
            r"NODE:.* EVENT:MEASURE_END DURATION:.*s BANDWIDTH_GBPS:(?P<bandwidth>[0-9.]+)"  # noqa: E501
        )

    def test_node_regex(self):
        log_line = "2026-08-09 03:00:00 - INFO - NODE:test-node-1 EVENT:MEASURE_END DURATION:10.50s BANDWIDTH_GBPS:15.53"  # noqa: E501
        match = self.node_regex.search(log_line)
        self.assertIsNotNone(match)
        self.assertEqual(match.group("node"), "test-node-1")

        log_line2 = (
            "NODE:tpu-worker-0 EVENT:MEASURE_END DURATION:5s BANDWIDTH_GBPS:5"
        )
        match2 = self.node_regex.search(log_line2)
        self.assertIsNotNone(match2)
        self.assertEqual(match2.group("node"), "tpu-worker-0")

        # Shouldn't match non-MEASURE_END
        log_line3 = "NODE:tpu-worker-0 EVENT:MEASURE_START"
        match3 = self.node_regex.search(log_line3)
        self.assertIsNone(match3)

    def test_bandwidth_regex(self):
        log_line = "2026-08-09 03:00:00 - INFO - NODE:test-node-1 EVENT:MEASURE_END DURATION:10.50s BANDWIDTH_GBPS:15.53"  # noqa: E501
        match = self.fom_regex.search(log_line)
        self.assertIsNotNone(match)
        self.assertEqual(match.group("bandwidth"), "15.53")


class TestTpuHbmStressApp(unittest.TestCase):
    def load_app_with_env(self, accel_type):
        os.environ["ACCELERATOR_TYPE"] = accel_type
        if "LIBTPU_INIT_ARGS" in os.environ:
            del os.environ["LIBTPU_INIT_ARGS"]

        # Load module manually because folder contains hyphens
        module_name = "tpu_hbm_stress"
        module_path = os.path.join(
            os.path.dirname(__file__),
            "../src/chs/tests/tpu-hbm-stress/tpu_hbm_stress.py",
        )
        module_path = os.path.abspath(module_path)

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        app = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = app
        spec.loader.exec_module(app)
        return app

    def test_v6e_env_vars(self):
        self.load_app_with_env("v6e-256")
        self.assertIn(
            "--xla_tpu_dvfs_p_state=3", os.environ["LIBTPU_INIT_ARGS"]
        )
        self.assertIn(
            "--xla_tpu_scoped_vmem_limit_kib=65536",
            os.environ["LIBTPU_INIT_ARGS"],
        )
        self.assertIn(
            "--megascale_error_reporter_abort_on_hang=true",
            os.environ["LIBTPU_INIT_ARGS"],
        )
        self.assertIn(
            "--tpu_link_up_check_timeout=15s", os.environ["LIBTPU_INIT_ARGS"]
        )
        self.assertIn(
            "--xla_jf_bounds_check=false", os.environ["LIBTPU_INIT_ARGS"]
        )

    def test_v5e_env_vars(self):
        self.load_app_with_env("v5e-256")
        self.assertIn(
            "--xla_tpu_dvfs_p_state=3", os.environ["LIBTPU_INIT_ARGS"]
        )

    def test_ct6e_env_vars(self):
        self.load_app_with_env("ct6e-256")
        self.assertIn(
            "--xla_tpu_dvfs_p_state=3", os.environ["LIBTPU_INIT_ARGS"]
        )

    def test_v4_env_vars(self):
        self.load_app_with_env("v4-128")
        self.assertIn(
            "--xla_tpu_dvfs_p_state=7", os.environ["LIBTPU_INIT_ARGS"]
        )

    def test_v5p_env_vars(self):
        self.load_app_with_env("v5p-256")
        self.assertIn(
            "--xla_tpu_dvfs_p_state=7", os.environ["LIBTPU_INIT_ARGS"]
        )

    def test_main_pass(self):
        app = self.load_app_with_env("v4-128")

        # mock device behavior
        app.jax.local_devices.return_value = [MagicMock()]
        app.jax.local_devices()[0].memory_stats.return_value = {
            "bytes_limit": 16 * 1024**3
        }

        app.hbm_burn = MagicMock(return_value="mock_outputs")
        app.hbm_checksum = MagicMock(return_value="mock_checksum_expr")
        app.jax.device_get.return_value = [12345]  # golden checksum
        app.jax.process_index.return_value = 0

        with (
            patch.object(app.logging, "info") as mock_info,
            patch.object(app.logging, "error") as mock_error,
        ):
            app.main()
            mock_error.assert_not_called()
            # Assert "STRESS_PASS" is logged
            pass_called = any(
                "STRESS_PASS" in call_args[0][0]
                for call_args in mock_info.call_args_list
            )
            self.assertTrue(pass_called)

    def test_main_fail(self):
        app = self.load_app_with_env("v4-128")

        app.jax.local_devices.return_value = [
            MagicMock(),
            MagicMock(),
        ]  # 2 devices
        app.jax.process_index.return_value = 0
        app.hbm_burn = MagicMock(return_value="mock_outputs")
        app.hbm_checksum = MagicMock(return_value="mock_checksum_expr")

        # One device matches golden, second device corrupts
        app.jax.device_get.return_value = [12345, 99999]

        with (
            patch.object(app.logging, "info"),
            patch.object(app.logging, "error") as mock_error,
        ):
            app.main()
            mock_error.assert_called()
            # Assert "STRESS_FAIL" is logged in error
            fail_called = any(
                "STRESS_FAIL" in call_args[0][0]
                for call_args in mock_error.call_args_list
            )
            self.assertTrue(fail_called)


if __name__ == "__main__":
    unittest.main()
