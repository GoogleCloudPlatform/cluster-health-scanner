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
from chs import cli
import io


class TestGcsResults(unittest.TestCase):
    @mock.patch("chs.cli.tempfile.NamedTemporaryFile")
    @mock.patch("chs.cli.storage.Client")
    @mock.patch("chs.cli.ResultsViewer")
    def test_show_results_gcs_file_success(
        self, mock_viewer, mock_storage_client, mock_tempfile
    ):
        """Test reading BoH directly from GCS file."""
        mock_tmp = mock.Mock()
        mock_tmp.name = "/tmp/fake_file.json"
        mock_tempfile.return_value.__enter__.return_value = mock_tmp

        mock_client = mock.Mock()
        mock_storage_client.return_value = mock_client
        mock_bucket = mock.Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_blob = mock.Mock()
        mock_bucket.blob.return_value = mock_blob

        mock_app_data = mock.Mock()
        mock_app_data.get_workspaces_dir.return_value = "/tmp/fake_workspaces"

        cli._show_results(mock_app_data, run_id=None, gcs="bucket/boh.json")

        mock_storage_client.assert_called_once()
        mock_client.bucket.assert_called_once_with("bucket")
        mock_bucket.blob.assert_called_once_with("boh.json")
        mock_blob.download_to_filename.assert_called_once_with(
            "/tmp/fake_file.json"
        )
        mock_viewer.assert_called_once_with("/tmp/fake_file.json")

    @mock.patch("chs.cli.tempfile.NamedTemporaryFile")
    @mock.patch("chs.cli.storage.Client")
    @mock.patch("chs.cli.ResultsViewer")
    def test_show_results_gcs_dir_success(
        self, mock_viewer, mock_storage_client, mock_tempfile
    ):
        """Test reading BoH from GCS directory."""
        mock_tmp = mock.Mock()
        mock_tmp.name = "/tmp/fake_file.json"
        mock_tempfile.return_value.__enter__.return_value = mock_tmp

        mock_client = mock.Mock()
        mock_storage_client.return_value = mock_client
        mock_bucket = mock.Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_blob = mock.Mock()
        mock_bucket.blob.return_value = mock_blob

        mock_app_data = mock.Mock()
        mock_app_data.get_workspaces_dir.return_value = "/tmp/fake_workspaces"

        cli._show_results(mock_app_data, run_id=None, gcs="bucket/workspace/")

        mock_storage_client.assert_called_once()
        mock_client.bucket.assert_called_once_with("bucket")
        mock_bucket.blob.assert_called_once_with(
            "workspace/bill_of_health.json"
        )
        mock_blob.download_to_filename.assert_called_once_with(
            "/tmp/fake_file.json"
        )
        mock_viewer.assert_called_once_with("/tmp/fake_file.json")

    @mock.patch("chs.cli.storage.Client")
    @mock.patch("chs.cli.ResultsViewer")
    @mock.patch("chs.cli.tarfile.open")
    @mock.patch("chs.cli.tempfile.NamedTemporaryFile")
    def test_show_results_gcs_archive_success(
        self, mock_tempfile, mock_tar_open, mock_viewer, mock_storage_client
    ):
        """Test reading BoH from GCS archive root."""
        mock_tmp = mock.Mock()
        mock_tmp.name = "/tmp/fake_file.json"
        mock_tempfile.return_value.__enter__.return_value = mock_tmp

        mock_client = mock.Mock()
        mock_storage_client.return_value = mock_client
        mock_bucket = mock.Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_blob = mock.Mock()
        mock_bucket.blob.return_value = mock_blob

        mock_gcs_file = io.BytesIO(b"fake tar content")
        mock_blob.open = mock.MagicMock()
        mock_blob.open.return_value.__enter__.return_value = mock_gcs_file

        mock_tar = mock.Mock()
        mock_tar_open.return_value.__enter__.return_value = mock_tar

        mock_file = io.BytesIO(b'{"report_id": "test_archive"}')
        mock_tar.extractfile.return_value = mock_file

        mock_app_data = mock.Mock()
        mock_app_data.get_workspaces_dir.return_value = "/tmp/fake_workspaces"

        cli._show_results(
            mock_app_data, run_id=None, gcs="bucket/workspace.tar.gz"
        )

        mock_storage_client.assert_called_once()
        mock_client.bucket.assert_called_once_with("bucket")
        mock_bucket.blob.assert_called_once_with("workspace.tar.gz")
        mock_blob.open.assert_called_once_with("rb")
        mock_viewer.assert_called_once_with("/tmp/fake_file.json")

    @mock.patch("chs.cli.storage.Client")
    @mock.patch("chs.cli.ResultsViewer")
    @mock.patch("chs.cli.tarfile.open")
    @mock.patch("chs.cli.tempfile.NamedTemporaryFile")
    def test_show_results_gcs_archive_fallback_success(
        self, mock_tempfile, mock_tar_open, mock_viewer, mock_storage_client
    ):
        """Test reading BoH from GCS archive with fallback search."""
        mock_tmp = mock.Mock()
        mock_tmp.name = "/tmp/fake_file.json"
        mock_tempfile.return_value.__enter__.return_value = mock_tmp

        mock_client = mock.Mock()
        mock_storage_client.return_value = mock_client
        mock_bucket = mock.Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_blob = mock.Mock()
        mock_bucket.blob.return_value = mock_blob

        mock_gcs_file = io.BytesIO(b"fake tar content")
        mock_blob.open = mock.MagicMock()
        mock_blob.open.return_value.__enter__.return_value = mock_gcs_file

        mock_tar = mock.Mock()
        mock_tar_open.return_value.__enter__.return_value = mock_tar

        # First call to extractfile (root) raises KeyError
        mock_tar.extractfile.side_effect = [
            KeyError("not found"),
            io.BytesIO(b'{"report_id": "test_archive_fallback"}'),
        ]

        mock_member = mock.Mock()
        mock_member.name = "nested/dir/bill_of_health.json"
        mock_tar.getmembers.return_value = [mock_member]

        mock_app_data = mock.Mock()
        mock_app_data.get_workspaces_dir.return_value = "/tmp/fake_workspaces"

        cli._show_results(
            mock_app_data, run_id=None, gcs="bucket/workspace.tar.gz"
        )

        mock_tar.extractfile.assert_has_calls(
            [mock.call("bill_of_health.json"), mock.call(mock_member)]
        )
        mock_viewer.assert_called_once_with("/tmp/fake_file.json")


if __name__ == "__main__":
    unittest.main()
