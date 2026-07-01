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
from unittest import mock
from chs.preflights.gcp_preflight import verify_gcp_credentials
from chs.exceptions import ChsPreflightException
from google.auth.exceptions import DefaultCredentialsError


class TestGcpPreflight(unittest.TestCase):
    @mock.patch("chs.preflights.gcp_preflight.google.auth.default")
    def test_verify_gcp_credentials_success(self, mock_auth_default):
        mock_auth_default.return_value = (mock.MagicMock(), "my-project")

        # Should not raise any exception
        verify_gcp_credentials()
        mock_auth_default.assert_called_once()

    @mock.patch("chs.preflights.gcp_preflight.google.auth.default")
    def test_verify_gcp_credentials_failure(self, mock_auth_default):
        mock_auth_default.side_effect = DefaultCredentialsError(
            "Missing credentials"
        )

        with self.assertRaisesRegex(
            ChsPreflightException,
            r"GCP Application Default Credentials \(ADC\) not found",
        ):
            verify_gcp_credentials()


if __name__ == "__main__":
    unittest.main()
