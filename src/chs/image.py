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

"""The published CHS image that TPU test pods run in."""

from chs.exceptions import ChsValidationError
from chs.version import UNKNOWN_VERSION, get_version

IMAGE_REPOSITORY = "us-docker.pkg.dev/gce-ai-infra/chs/chs"


def get_image() -> str:
    """Returns the published CHS image of the installed CHS version.

    Raises:
        ChsValidationError: If the installed CHS version is unknown.
    """
    version = get_version()
    if version == UNKNOWN_VERSION:
        raise ChsValidationError(
            "The installed CHS version is unknown, so the CHS image for test"
            " pods can't be chosen. Install CHS with pip."
        )
    return f"{IMAGE_REPOSITORY}:{version}"
