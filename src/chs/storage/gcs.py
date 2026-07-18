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

"""GCS storage provider implementation for CHS."""

import logging
import uuid
from chs.exceptions import ChsValidationError, StorageObjectNotFoundException
from chs.storage.base import StorageProvider
from google.cloud import storage
from google.api_core import exceptions as gcp_ex

logger = logging.getLogger(__name__)


class GcsStorageProvider(StorageProvider):
    """Storage provider for Google Cloud Storage buckets."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.bucket_name = config.get("bucket_name", "")

    def upload_file(self, local_src: str, remote_dest: str) -> None:
        """Uploads a local file to the GCS bucket."""
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(remote_dest)
            blob.upload_from_filename(local_src)
            logger.debug(
                "Uploaded %s to gs://%s/%s",
                local_src,
                self.bucket_name,
                remote_dest,
            )
        except Exception as e:
            raise OSError(
                f"Failed to upload {local_src} to "
                f"gs://{self.bucket_name}/{remote_dest}: {e}"
            ) from e

    def download_file(self, remote_src: str, local_dest: str) -> None:
        """Downloads a file from the GCS bucket to local path."""
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(remote_src)
            blob.download_to_filename(local_dest)
            logger.debug(
                "Downloaded gs://%s/%s to %s",
                self.bucket_name,
                remote_src,
                local_dest,
            )
        except gcp_ex.NotFound as e:
            raise StorageObjectNotFoundException(
                f"File gs://{self.bucket_name}/{remote_src} not found."
            ) from e
        except Exception as e:
            raise OSError(
                f"Failed to download gs://{self.bucket_name}/{remote_src} "
                f"to {local_dest}: {e}"
            ) from e

    def validate(self) -> None:
        """Validates writeability of GCS bucket via write/delete test."""
        if not self.bucket_name:
            raise ChsValidationError("GCS bucket name cannot be empty.")

        client = storage.Client()
        bucket = client.bucket(self.bucket_name)

        canary_name = f".chs-connectivity-test-{uuid.uuid4()}"
        blob = bucket.blob(canary_name)

        # Write phase
        try:
            blob.upload_from_string(b"")
            logger.debug(
                "Wrote validation canary gs://%s/%s",
                self.bucket_name,
                canary_name,
            )
        except gcp_ex.NotFound as e:
            raise ChsValidationError(
                f"GCS bucket '{self.bucket_name}' not found. "
                f"Please verify the bucket name. "
                f"Details: {e}"
            ) from e
        except gcp_ex.Forbidden as e:
            raise ChsValidationError(
                f"Access denied: Write permission missing for GCS bucket "
                f"'{self.bucket_name}'. "
                f"Details: {e}"
            ) from e
        except (gcp_ex.GoogleAPICallError, Exception) as e:
            raise ChsValidationError(
                f"Failed to access GCS bucket '{self.bucket_name}'. "
                f"Details: {e}"
            ) from e

        # Delete phase
        try:
            blob.delete()
            logger.debug(
                "Deleted validation canary gs://%s/%s",
                self.bucket_name,
                canary_name,
            )
        except gcp_ex.GoogleAPICallError as e:
            if e.code == 403:
                logger.debug(
                    "GCS validation delete test failed (403 Forbidden) "
                    "for bucket '%s'. Assuming append-only permissions: %s",
                    self.bucket_name,
                    e,
                )
            else:
                logger.debug(
                    "GCS validation delete test failed with status %d: %s",
                    e.code,
                    e,
                )
        except Exception as e:
            logger.debug(
                "Unexpected error during GCS validation delete test: %s", e
            )

    def get_uri(self, remote_path: str) -> str:
        """Returns the gs:// URI for the remote path."""
        return f"gs://{self.bucket_name}/{remote_path}"

    def exists(self, remote_path: str) -> bool:
        """Checks if blob exists in GCS bucket."""
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(remote_path)
            return blob.exists()
        except Exception:
            return False
