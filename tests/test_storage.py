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

"""Unit tests for CHS storage providers and factory."""

import os
import pathlib
import tempfile
import unittest
from unittest import mock

from chs.exceptions import (
    ChsException,
    ChsValidationError,
)
from chs.storage import StorageProviderFactory
from chs.storage import gcs
from chs.storage.gcs import GcsStorageProvider
from chs.storage.path import PathStorageProvider
from chs.utils import gcp
from google.api_core import exceptions as gcp_ex


class TestStorageProviderFactory(unittest.TestCase):
    """Tests for StorageProviderFactory."""

    def test_get_provider_gcs(self):
        """Test instantiating GcsStorageProvider."""
        config = {"type": "gcs", "gcs": {"bucket_name": "my-bucket"}}
        with mock.patch("chs.storage.gcs.storage.Client"):
            provider = StorageProviderFactory.get_provider(config)
            self.assertIsInstance(provider, GcsStorageProvider)
            self.assertEqual(provider.bucket_name, "my-bucket")

    def test_get_provider_path(self):
        """Test instantiating PathStorageProvider."""
        config = {"type": "path", "path": {"path": "/my/path"}}
        provider = StorageProviderFactory.get_provider(config)
        self.assertIsInstance(provider, PathStorageProvider)
        self.assertEqual(
            provider.target_path, pathlib.Path("/my/path").resolve()
        )

    def test_get_provider_path_tilde(self):
        """Test instantiating PathStorageProvider with tilde path."""
        config = {"type": "path", "path": {"path": "~/my/path"}}
        provider = StorageProviderFactory.get_provider(config)
        self.assertIsInstance(provider, PathStorageProvider)
        self.assertEqual(
            provider.target_path,
            pathlib.Path("~/my/path").expanduser().resolve(),
        )

    def test_get_provider_invalid_type(self):
        """Test get_provider raises on unsupported type."""
        config = {"type": "s3", "s3": {"bucket": "foo"}}
        with self.assertRaises(ChsException) as ctx:
            StorageProviderFactory.get_provider(config)
        self.assertIn("Unsupported storage type: 's3'", str(ctx.exception))

    def test_get_provider_missing_type(self):
        """Test get_provider raises if type is missing."""
        config = {"path": {"path": "/my/path"}}
        with self.assertRaises(ChsException) as ctx:
            StorageProviderFactory.get_provider(config)
        self.assertIn("Storage type is not specified", str(ctx.exception))

    @mock.patch("chs.storage.ChsAppData")
    def test_get_active_provider_gcs(self, mock_app_data_cls):
        """Test get_active_provider with active GCS config."""
        mock_app_data = mock_app_data_cls.return_value
        mock_app_data.get_storage_config.return_value = {
            "type": "gcs",
            "gcs": {"bucket_name": "active-bucket"},
        }
        with mock.patch("chs.storage.gcs.storage.Client"):
            provider = StorageProviderFactory.get_active_provider()
            self.assertIsInstance(provider, GcsStorageProvider)
            self.assertEqual(provider.bucket_name, "active-bucket")

    @mock.patch("chs.storage.ChsAppData")
    def test_get_active_provider_not_configured(self, mock_app_data_cls):
        """Test get_active_provider raises if no config found."""
        mock_app_data = mock_app_data_cls.return_value
        mock_app_data.get_storage_config.return_value = {}

        with self.assertRaises(ChsException) as ctx:
            StorageProviderFactory.get_active_provider()
        self.assertIn("No storage provider configured", str(ctx.exception))


class TestPathStorageProvider(unittest.TestCase):
    """Tests for PathStorageProvider."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.target_path = os.path.join(
            self.temp_dir.name, "archive", "results"
        )
        self.config = {"path": self.target_path}
        self.provider = PathStorageProvider(self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_validate_creates_dirs_and_files(self):
        """Test validate creates directory structure and write test succeeds."""
        self.assertFalse(os.path.exists(self.target_path))
        self.provider.validate()
        self.assertTrue(os.path.exists(self.target_path))
        # Canary file should be deleted after validation
        self.assertEqual(os.listdir(self.target_path), [])

    def test_validate_cleanup_on_write_failure(self):
        """Test validate cleans up created directories if write test fails."""
        # Create a directory path that we cannot write to (simulated by making a
        # parent read-only)
        parent_dir = os.path.join(self.temp_dir.name, "readonly_parent")
        os.makedirs(parent_dir)

        target_path = os.path.join(parent_dir, "new_dir", "results")
        provider = PathStorageProvider({"path": target_path})

        # Make parent read-only before validation attempt
        # Wait, if we make parent read-only, directory creation might fail.
        # We want to test case where directory creation succeeds but write
        # fails.
        # This is harder to simulate portably without permission tricks.
        # Alternative: Mock mkdir to succeed but mock NamedTemporaryFile to
        # fail.
        with mock.patch("tempfile.NamedTemporaryFile") as mock_tempfile:
            mock_tempfile.side_effect = OSError("No space left on device")

            self.assertFalse(os.path.exists(target_path))
            with self.assertRaises(ChsValidationError):
                provider.validate()

            # Should have cleaned up 'new_dir' and 'results'
            self.assertFalse(
                os.path.exists(os.path.join(parent_dir, "new_dir"))
            )

    def test_upload_and_download(self):
        """Test file copying logic."""
        os.makedirs(self.target_path)
        src_file = os.path.join(self.temp_dir.name, "src.txt")
        with open(src_file, "w") as f:
            f.write("hello")

        # Upload
        self.provider.upload_file(src_file, "bills_of_health/run1.json")
        dest_file = os.path.join(
            self.target_path, "bills_of_health", "run1.json"
        )
        self.assertTrue(os.path.exists(dest_file))
        with open(dest_file, "r") as f:
            self.assertEqual(f.read(), "hello")

        # Download
        download_dest = os.path.join(self.temp_dir.name, "downloaded.txt")
        self.provider.download_file("bills_of_health/run1.json", download_dest)
        self.assertTrue(os.path.exists(download_dest))
        with open(download_dest, "r") as f:
            self.assertEqual(f.read(), "hello")


class TestGcsStorageProvider(unittest.TestCase):
    """Tests for GcsStorageProvider."""

    def setUp(self):
        super().setUp()
        gcs._storage_client.cache_clear()
        self.addCleanup(gcs._storage_client.cache_clear)
        patcher = mock.patch.object(
            gcp,
            "get_default_credentials",
            return_value=(mock.sentinel.credentials, "my-project"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @mock.patch("chs.storage.gcs.storage.Client")
    def test_validate_success(self, mock_storage_client):
        """Test validate succeeds if write and delete succeed."""
        mock_client = mock_storage_client.return_value
        mock_bucket = mock_client.bucket.return_value
        mock_blob = mock_bucket.blob.return_value

        provider = GcsStorageProvider({"bucket_name": "my-bucket"})
        provider.validate()

        mock_storage_client.assert_called_once()
        mock_client.bucket.assert_called_once_with("my-bucket")
        mock_blob.upload_from_string.assert_called_once_with(b"")
        mock_blob.delete.assert_called_once()

    @mock.patch("chs.storage.gcs.storage.Client")
    def test_validate_write_failure(self, mock_storage_client):
        """Test validate raises ChsValidationError if upload fails."""
        mock_client = mock_storage_client.return_value
        mock_bucket = mock_client.bucket.return_value
        mock_blob = mock_bucket.blob.return_value
        mock_blob.upload_from_string.side_effect = Exception("API Error")

        provider = GcsStorageProvider({"bucket_name": "my-bucket"})
        with self.assertRaises(ChsValidationError) as ctx:
            provider.validate()
        self.assertIn("Failed to access GCS bucket", str(ctx.exception))
        mock_blob.delete.assert_not_called()

    @mock.patch("chs.storage.gcs.storage.Client")
    def test_validate_write_failure_not_found(self, mock_storage_client):
        """Test validate raises ChsValidationError if bucket not found (404)."""
        mock_client = mock_storage_client.return_value
        mock_bucket = mock_client.bucket.return_value
        mock_blob = mock_bucket.blob.return_value
        mock_blob.upload_from_string.side_effect = gcp_ex.NotFound("Not Found")

        provider = GcsStorageProvider({"bucket_name": "my-bucket"})
        with self.assertRaises(ChsValidationError) as ctx:
            provider.validate()
        self.assertIn(
            "not found. Please verify the bucket name", str(ctx.exception)
        )
        self.assertIn("gcloud storage buckets create", str(ctx.exception))
        mock_blob.delete.assert_not_called()

    @mock.patch("chs.storage.gcs.storage.Client")
    def test_validate_write_failure_forbidden(self, mock_storage_client):
        """Test validate raises ChsValidationError if write forbidden (403)."""
        mock_client = mock_storage_client.return_value
        mock_bucket = mock_client.bucket.return_value
        mock_blob = mock_bucket.blob.return_value
        mock_blob.upload_from_string.side_effect = gcp_ex.Forbidden("Forbidden")

        provider = GcsStorageProvider({"bucket_name": "my-bucket"})
        with self.assertRaises(ChsValidationError) as ctx:
            provider.validate()
        self.assertIn(
            "Access denied: Write permission missing", str(ctx.exception)
        )
        mock_blob.delete.assert_not_called()

    @mock.patch("chs.storage.gcs.storage.Client")
    def test_validate_delete_failure_ignored(self, mock_storage_client):
        """Test validate passes if delete fails (append-only bucket)."""
        mock_client = mock_storage_client.return_value
        mock_bucket = mock_client.bucket.return_value
        mock_blob = mock_bucket.blob.return_value

        from google.api_core.exceptions import Forbidden

        mock_blob.delete.side_effect = Forbidden("Access Denied")

        provider = GcsStorageProvider({"bucket_name": "my-bucket"})
        # Should not raise exception
        provider.validate()
        mock_blob.upload_from_string.assert_called_once()
        mock_blob.delete.assert_called_once()


class TestGcsStorageProviderClientReuse(unittest.TestCase):
    """Tests that GCS providers share one process-wide GCS client."""

    def setUp(self):
        super().setUp()
        gcs._storage_client.cache_clear()
        self.addCleanup(gcs._storage_client.cache_clear)
        self.credentials = mock.sentinel.credentials
        patcher = mock.patch.object(
            gcp,
            "get_default_credentials",
            return_value=(self.credentials, "my-project"),
        )
        self.mock_get_credentials = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch("chs.storage.gcs.storage.Client")
        self.mock_client_cls = patcher.start()
        self.addCleanup(patcher.stop)
        self.mock_blob = (
            self.mock_client_cls.return_value.bucket.return_value.blob
        ).return_value
        self.provider = GcsStorageProvider({"bucket_name": "my-bucket"})

    def test_client_is_not_created_until_needed(self):
        """Test constructing the provider does not create a client."""
        self.mock_client_cls.assert_not_called()

    def test_client_is_reused_across_calls(self):
        """Test uploads, downloads and existence checks share one client."""
        self.mock_blob.exists.return_value = True

        self.provider.upload_file("/tmp/boh.json", "bills_of_health/r.json")
        self.provider.upload_file("/tmp/boh.html", "bills_of_health/r.html")
        self.provider.download_file("bills_of_health/r.json", "/tmp/r.json")
        self.assertTrue(self.provider.exists("bills_of_health/r.json"))

        self.mock_client_cls.assert_called_once_with(
            credentials=self.credentials, project="my-project"
        )
        self.assertEqual(self.mock_client_cls.return_value.bucket.call_count, 4)

    def test_client_is_shared_across_providers(self):
        """Test a second provider reuses the first provider's client."""
        other = GcsStorageProvider({"bucket_name": "other-bucket"})

        self.provider.upload_file("/tmp/boh.json", "bills_of_health/r.json")
        other.upload_file("/tmp/boh.json", "bills_of_health/r.json")

        self.mock_client_cls.assert_called_once()
        self.mock_get_credentials.assert_called_once_with()
        self.assertEqual(
            self.mock_client_cls.return_value.bucket.call_args_list,
            [mock.call("my-bucket"), mock.call("other-bucket")],
        )

    def test_failed_client_creation_is_retried(self):
        """Test a failed client creation is not cached."""
        self.mock_client_cls.side_effect = [
            Exception("no credentials"),
            mock.DEFAULT,
        ]

        with self.assertRaises(OSError) as ctx:
            self.provider.upload_file("/tmp/boh.json", "bills_of_health/r.json")
        self.assertEqual(
            str(ctx.exception),
            "Failed to upload /tmp/boh.json to "
            "gs://my-bucket/bills_of_health/r.json: no credentials",
        )

        self.provider.upload_file("/tmp/boh.json", "bills_of_health/r.json")
        self.assertEqual(self.mock_client_cls.call_count, 2)
        self.mock_blob.upload_from_filename.assert_called_once_with(
            "/tmp/boh.json"
        )

    def test_exists_is_false_when_client_creation_fails(self):
        """Test exists() still reports False if no client can be created."""
        self.mock_client_cls.side_effect = Exception("no credentials")
        self.assertFalse(self.provider.exists("bills_of_health/r.json"))


if __name__ == "__main__":
    unittest.main()
