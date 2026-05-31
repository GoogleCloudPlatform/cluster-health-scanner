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

"""Storage provider factory and lazy loader."""

import importlib
from chs.exceptions import ChsException
from chs.storage.base import StorageProvider
from chs.utils.app_data import ChsAppData


class StorageProviderFactory:
    """Factory to instantiate storage providers dynamically."""

    @staticmethod
    def get_provider(storage_config: dict) -> StorageProvider:
        """Instantiates a provider based on the passed configuration.

        Args:
            storage_config: Storage configuration dictionary.

        Returns:
            An instantiated StorageProvider.

        Raises:
            ChsException: If configuration is invalid or type is unsupported.
        """
        if not storage_config:
            raise ChsException("Storage configuration is empty.")

        provider_type = storage_config.get("type")
        if not provider_type:
            raise ChsException(
                "Storage type is not specified in configuration."
            )

        if provider_type == "gcs":
            try:
                module = importlib.import_module("chs.storage.gcs")
                return module.GcsStorageProvider(storage_config.get("gcs", {}))
            except ImportError as e:
                raise ChsException(
                    f"Failed to load GCS storage provider. "
                    f"Ensure google-cloud-storage is installed: {e}"
                ) from e
        elif provider_type == "path":
            module = importlib.import_module("chs.storage.path")
            return module.PathStorageProvider(storage_config.get("path", {}))
        else:
            raise ChsException(f"Unsupported storage type: '{provider_type}'")

    @staticmethod
    def get_active_provider() -> StorageProvider:
        """Instantiates the currently active storage provider.

        Returns:
            An instantiated StorageProvider.

        Raises:
            ChsException: If no active provider is configured.
        """
        app_data = ChsAppData()
        config = app_data.get_storage_config()

        if not config:
            raise ChsException(
                "No storage provider configured. "
                "Run 'chs configure storage' to set up storage."
            )

        return StorageProviderFactory.get_provider(config)
