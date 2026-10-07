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

"""Unit tests for CHS version reporting.

Covers the places the version surfaces: the helper itself, the
``chs --version`` flag, the ``chs_version`` field of a Bill of Health,
and the interactive HTML report.
"""

import importlib.metadata
import json
import os
import tempfile
import unittest
from unittest import mock

from click import testing

import chs.boh_generator
from chs import cli
from chs import version


class GetVersionTest(unittest.TestCase):
    """Tests for chs.version.get_version."""

    def setUp(self):
        super().setUp()
        version.get_version.cache_clear()
        self.addCleanup(version.get_version.cache_clear)

    def test_reads_installed_distribution_metadata(self):
        """The version comes from the distribution, not a literal."""
        with mock.patch.object(
            importlib.metadata, "version", return_value="9.8.7"
        ) as mock_version:
            self.assertEqual(version.get_version(), "9.8.7")
        mock_version.assert_called_once_with("chs")

    def test_falls_back_when_not_installed(self):
        """An uninstalled source tree degrades instead of raising."""
        with mock.patch.object(
            importlib.metadata,
            "version",
            side_effect=importlib.metadata.PackageNotFoundError("chs"),
        ):
            self.assertEqual(version.get_version(), version.UNKNOWN_VERSION)

    def test_metadata_is_read_only_once(self):
        """Resolving distribution metadata walks sys.path, so cache it."""
        with mock.patch.object(
            importlib.metadata, "version", return_value="9.8.7"
        ) as mock_version:
            for _ in range(3):
                version.get_version()

        mock_version.assert_called_once()


class CliVersionFlagTest(unittest.TestCase):
    """Tests for the `chs --version` flag."""

    def test_version_flag_prints_version_and_exits_zero(self):
        result = testing.CliRunner().invoke(cli.cli, ["--version"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("chs", result.output)
        self.assertIn(version.get_version(), result.output)

    def test_short_v_is_still_verbose_not_version(self):
        """`-v` must keep meaning --verbose; only the long flag is version."""
        result = testing.CliRunner().invoke(cli.cli, ["-v", "--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertNotIn(version.get_version(), result.output)


class BillOfHealthVersionTest(unittest.TestCase):
    """Tests that a generated Bill of Health carries the real version."""

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.json_path = os.path.join(self.temp_dir.name, "results.json")
        with open(self.json_path, "w") as f:
            json.dump({"workspace_name": "test_hash", "experiments": []}, f)

    def test_report_uses_the_version_helper(self):
        with mock.patch.object(
            chs.boh_generator, "get_version", return_value="9.8.7"
        ):
            generator = chs.boh_generator.BOHGenerator(self.json_path)
            report = generator.generate()

        self.assertEqual(report.chs_version, "9.8.7")


class HtmlReportVersionTest(unittest.TestCase):
    """Tests that HtmlReportGenerator carries the real version."""

    def test_metadata_uses_the_version_helper(self):
        from chs import html_generator
        from chs.protos import bill_of_health_pb2 as pb

        generator = html_generator.HtmlReportGenerator()
        boh = pb.BillOfHealth()
        boh.report_id = "test-run"
        with mock.patch.object(
            html_generator, "get_version", return_value="9.8.7"
        ):
            report_data = generator.generate_report_data(boh)

        self.assertEqual(report_data["metadata"]["chs_version"], "9.8.7")

    def test_metadata_prefers_boh_chs_version(self):
        from chs import html_generator
        from chs.protos import bill_of_health_pb2 as pb

        generator = html_generator.HtmlReportGenerator()
        boh = pb.BillOfHealth()
        boh.report_id = "test-run"
        boh.chs_version = "1.2.3"
        report_data = generator.generate_report_data(boh)

        self.assertEqual(report_data["metadata"]["chs_version"], "1.2.3")


if __name__ == "__main__":
    unittest.main()
