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

"""Cluster name retrieval and sanitization utilities for CHS."""

import logging
import random
import re
import string
from typing import TYPE_CHECKING
from chs.exceptions import ClusterDiscoveryException

if TYPE_CHECKING:
    from chs.schedulers.base import BaseScheduler

logger = logging.getLogger(__name__)


def generate_random_suffix() -> str:
    """Generates a 4-character lowercase alphanumeric random suffix."""
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(4))


def sanitize_dns_1123(s: str, max_length: int | None = None) -> str:
    """Sanitizes a string to comply with DNS-1123 subdomain format.

    1. Convert to lowercase.
    2. Replace spaces and underscores with dashes.
    3. Strip all non-alphanumeric characters (except dashes).
    4. Truncate to max_length if specified.
    5. Strip any dangling leading or trailing dashes.
    """
    s = s.lower()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"[^a-z0-9\-]", "", s)
    if max_length is not None:
        s = s[:max_length]
    s = s.strip("-")
    return s


def get_sanitized_cluster_name(scheduler_obj: "BaseScheduler") -> str:
    """Retrieves and sanitizes the cluster name using the scheduler object.

    Returns:
        A sanitized cluster name.

    Raises:
        ClusterDiscoveryException: If the cluster name cannot be retrieved.
    """
    name = scheduler_obj.get_cluster_name()
    if not name:
        raise ClusterDiscoveryException(
            "Failed to retrieve cluster name from the scheduler."
        )
    sanitized = sanitize_dns_1123(name, max_length=15)
    if not sanitized:
        raise ClusterDiscoveryException(
            f"Cluster name '{name}' could not be sanitized to "
            "comply with DNS-1123 rules."
        )
    return sanitized
