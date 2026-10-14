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

import logging
from google.auth.exceptions import DefaultCredentialsError
from chs.exceptions import ChsPreflightException
from chs.utils import gcp

logger = logging.getLogger(__name__)


def verify_gcp_credentials() -> None:
    """Verifies that GCP Application Default Credentials are available."""
    try:
        gcp.get_default_credentials()
        logger.debug("GCP Credentials verified successfully.")
    except DefaultCredentialsError as e:
        raise ChsPreflightException(
            "GCP Application Default Credentials (ADC) not found.\n"
            "To configure credentials, please run:\n"
            "    gcloud auth application-default login"
        ) from e
