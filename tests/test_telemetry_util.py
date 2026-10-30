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

"""Unit tests for chs.telemetry.telemetry_util."""

import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from chs.telemetry import telemetry_util

_GOOGLE_DOMAIN = "google.com"
_GOOGLER_EMAIL = f"someone@{_GOOGLE_DOMAIN}"


class IsGooglerTest(unittest.TestCase):
    """Tests for is_googler, which mirrors Cluster Toolkit's checks."""

    def setUp(self):
        super().setUp()
        self.gcloud_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(
            mock.patch.dict(
                os.environ, {"CLOUDSDK_CONFIG": self.gcloud_dir}, clear=True
            )
        )
        telemetry_util.is_googler.cache_clear()
        self.addCleanup(telemetry_util.is_googler.cache_clear)

    def _set_gcloud_account(self, account: str, config: str = "work") -> None:
        pathlib.Path(self.gcloud_dir, "active_config").write_text(config)
        configurations = pathlib.Path(self.gcloud_dir, "configurations")
        configurations.mkdir(exist_ok=True)
        (configurations / f"config_{config}").write_text(
            "[compute]\nzone = us-central1-a\n"
            f"[core]\naccount = {account}  # the active account\n"
        )

    def _set_adc_key(self, client_email: str) -> None:
        key = pathlib.Path(self.gcloud_dir, "key.json")
        key.write_text(json.dumps({"client_email": client_email}))
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key)

    def test_internal_accounts(self):
        for email in (
            _GOOGLER_EMAIL,
            "robot@prod.google.com",
            "ci@hpc-toolkit-dev.iam.gserviceaccount.com",
            "508417052821@cloudbuild.gserviceaccount.com",
        ):
            with self.subTest(email=email):
                self.assertTrue(telemetry_util._is_internal_account(email))

    def test_external_accounts(self):
        for email in (
            "",
            "someone@gmail.com",
            f"{_GOOGLER_EMAIL}.evil.example",
            "ci@customer-project.iam.gserviceaccount.com",
            "123@cloudbuild.gserviceaccount.com",
        ):
            with self.subTest(email=email):
                self.assertFalse(telemetry_util._is_internal_account(email))

    def test_googler_from_active_gcloud_account(self):
        self._set_gcloud_account(_GOOGLER_EMAIL)

        self.assertTrue(telemetry_util.is_googler())

    def test_external_gcloud_account(self):
        self._set_gcloud_account("someone@example.com")

        self.assertFalse(telemetry_util.is_googler())

    def test_internal_ci_key_wins_over_gcloud_account(self):
        self._set_gcloud_account("someone@example.com")
        self._set_adc_key("ci@hpc-toolkit-dev.iam.gserviceaccount.com")

        self.assertTrue(telemetry_util.is_googler())

    def test_external_key_falls_back_to_gcloud_account(self):
        self._set_gcloud_account(_GOOGLER_EMAIL)
        self._set_adc_key("ci@customer.iam.gserviceaccount.com")

        self.assertTrue(telemetry_util.is_googler())

    def test_missing_or_broken_config_is_external(self):
        self.assertFalse(telemetry_util.is_googler())
        telemetry_util.is_googler.cache_clear()
        pathlib.Path(self.gcloud_dir, "active_config").write_text("gone")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/nonexistent.json"

        self.assertFalse(telemetry_util.is_googler())

    def test_percent_sign_in_gcloud_account(self):
        self._set_gcloud_account(f"100%@{_GOOGLE_DOMAIN}")

        self.assertTrue(telemetry_util.is_googler())

    def test_non_utf8_gcloud_config_is_external(self):
        self._set_gcloud_account(_GOOGLER_EMAIL)
        pathlib.Path(
            self.gcloud_dir, "configurations", "config_work"
        ).write_bytes(
            b"[core]\naccount = \xff@" + _GOOGLE_DOMAIN.encode() + b"\n"
        )

        self.assertFalse(telemetry_util.is_googler())


if __name__ == "__main__":
    unittest.main()
