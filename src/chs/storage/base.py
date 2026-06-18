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

"""Base storage provider interface for CHS."""

import abc


class StorageProvider(abc.ABC):
    """Abstract base class for CHS storage providers."""

    def __init__(self, config: dict) -> None:
        """Initializes the provider with its configuration.

        Args:
            config: A dictionary containing provider-specific settings.
        """
        self.config = config

    @abc.abstractmethod
    def upload_file(self, local_src: str, remote_dest: str) -> None:
        """Uploads/copies a file from local source to remote destination.

        Args:
            local_src: Absolute path to the local file.
            remote_dest: Relative path in the storage target (e.g.
              'bills_of_health/run.json').
        """
        pass

    @abc.abstractmethod
    def download_file(self, remote_src: str, local_dest: str) -> None:
        """Downloads/copies a file from remote source to local destination.

        Args:
            remote_src: Relative path in the storage target.
            local_dest: Absolute path to the local destination file.
        """
        pass

    @abc.abstractmethod
    def validate(self) -> None:
        """Performs writeability validation on the storage target.

        Raises:
            ChsValidationError: If validation fails.
        """
        pass

    @abc.abstractmethod
    def get_uri(self, remote_path: str) -> str:
        """Returns the full URI or path to the given remote destination."""
        pass
