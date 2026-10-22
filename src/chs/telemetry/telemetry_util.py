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

"""Googler detection and Clearcut requests for chs.telemetry."""

import configparser
import functools
import json
import logging
import os
import re
import urllib.error
import urllib.request

from chs.version import get_version

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://play.googleapis.com/log?format=json_proto"

_INTERNAL_EMAIL_SUFFIXES = ("@example.com", ".google.com")
_INTERNAL_SERVICE_ACCOUNT_RES = (
    re.compile(
        r"(hpc-toolkit-dev|hpc-toolkit-demo|hpc-toolkit-gsc)"
        r".*gserviceaccount\.com$"
    ),
    re.compile(
        r"(508417052821|858831239249|266450182917)"
        r".*@cloudbuild\.gserviceaccount\.com$"
    ),
)


def _account_from_adc_key() -> str:
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not path:
        return ""
    try:
        with open(path) as f:
            return json.load(f).get("client_email", "")
    except (OSError, ValueError, AttributeError):
        return ""


def _account_from_gcloud_config() -> str:
    config_dir = os.environ.get("CLOUDSDK_CONFIG")
    if not config_dir:
        config_dir = os.path.expanduser("~/.config/gcloud")
    parser = configparser.ConfigParser(
        interpolation=None, inline_comment_prefixes=("#", ";")
    )
    try:
        with open(os.path.join(config_dir, "active_config")) as f:
            active_config = f.read().strip()
        if not active_config:
            return ""
        parser.read(
            os.path.join(
                config_dir, "configurations", f"config_{active_config}"
            )
        )
        return parser.get("core", "account", fallback="").strip()
    except (OSError, ValueError, configparser.Error):
        return ""


def _is_internal_account(email: str) -> bool:
    return bool(email) and (
        email.endswith(_INTERNAL_EMAIL_SUFFIXES)
        or any(p.search(email) for p in _INTERNAL_SERVICE_ACCOUNT_RES)
    )


@functools.cache
def is_googler() -> bool:
    return _is_internal_account(
        _account_from_adc_key()
    ) or _is_internal_account(_account_from_gcloud_config())


def post(body: bytes, timeout: float) -> bool:
    logger.debug("Sending telemetry request: %s", body.decode())
    try:
        request = urllib.request.Request(
            os.environ.get("CHS_TELEMETRY_ENDPOINT", _DEFAULT_ENDPOINT),
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"CHS-telemetry/{get_version()}",
            },
            method="POST",
        )
        urllib.request.urlopen(request, timeout=timeout).close()
    except urllib.error.HTTPError as e:
        logger.debug("Telemetry request failed: HTTP %s", e.code)
        return e.code >= 500
    return False
