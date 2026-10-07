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
from chs.utils.suites import load_suite, is_suite_tiered
from chs.utils import suites
from chs.utils import tests
from chs.utils import formatting
from chs.utils.cluster import (
    sanitize_dns_1123,
    get_sanitized_cluster_name,
    generate_random_suffix,
    is_in_cluster,
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
from chs.utils.topology import (
    build_node_to_location,
    build_subblock_to_blocks,
    get_domain_label,
    get_topological_context,
    get_topology,
    load_topology,
)

__all__ = [
    "ChsAppData",
    "load_suite",
    "is_suite_tiered",
    "create_styled_table",
    "CHS_HEADER_STYLE",
    "CHS_WARNING_STYLE",
    "CHS_TITLE_STYLE",
    "CHS_SUCCESS_STYLE",
    "CHS_FAILURE_STYLE",
    "CHS_NEUTRAL_STYLE",
    "suites",
    "tests",
    "formatting",
    "sanitize_dns_1123",
    "get_sanitized_cluster_name",
    "generate_random_suffix",
    "is_in_cluster",
    "build_node_to_location",
    "build_subblock_to_blocks",
    "get_domain_label",
    "get_topological_context",
    "get_topology",
    "load_topology",
]
