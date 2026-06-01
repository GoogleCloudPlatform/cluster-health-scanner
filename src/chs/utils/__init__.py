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

"""Utility functions for CHS."""

from chs.utils.app_data import ChsAppData
from chs.utils.suites import load_suite
from chs.utils import suites
from chs.utils import tests
from chs.utils.cluster import (
    sanitize_dns_1123,
    get_sanitized_cluster_name,
    generate_random_suffix,
)
from chs.utils.theme import (
    create_styled_table,
    CHS_HEADER_STYLE,
    CHS_WARNING_STYLE,
    CHS_TITLE_STYLE,
    CHS_SUCCESS_STYLE,
    CHS_FAILURE_STYLE,
    CHS_NEUTRAL_STYLE,
)

__all__ = [
    "ChsAppData",
    "load_suite",
    "create_styled_table",
    "CHS_HEADER_STYLE",
    "CHS_WARNING_STYLE",
    "CHS_TITLE_STYLE",
    "CHS_SUCCESS_STYLE",
    "CHS_FAILURE_STYLE",
    "CHS_NEUTRAL_STYLE",
    "suites",
    "tests",
    "sanitize_dns_1123",
    "get_sanitized_cluster_name",
    "generate_random_suffix",
]
