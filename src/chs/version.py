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

"""Single source of truth for the installed CHS version.

The version is declared once in ``pyproject.toml`` and bumped by
``bin/create_release_candidate.py`` at each release cut. Setuptools copies it
into the installed distribution metadata, so reading it back keeps this module
from drifting away from the release process.

This module imports nothing from :mod:`chs`, so it is safe to import from
anywhere in the package.
"""

import functools
import importlib.metadata
import logging

logger = logging.getLogger(__name__)

_DISTRIBUTION_NAME = "chs"

UNKNOWN_VERSION = "0.0.0+unknown"


@functools.cache
def get_version() -> str:
    """Returns the installed CHS version, e.g. ``"0.3.0"``.

    Cached: resolving distribution metadata walks ``sys.path``, and the answer
    cannot change within a process. Tests that patch the metadata lookup must
    call ``get_version.cache_clear()`` first.

    Returns:
        The version from the installed distribution metadata, or
        :data:`UNKNOWN_VERSION` if CHS is not installed as a distribution.
    """
    try:
        return importlib.metadata.version(_DISTRIBUTION_NAME)
    except importlib.metadata.PackageNotFoundError:
        logger.debug(
            "Distribution %r is not installed; reporting %s. Run "
            "'pip install -e .' to get a real version.",
            _DISTRIBUTION_NAME,
            UNKNOWN_VERSION,
        )
        return UNKNOWN_VERSION
