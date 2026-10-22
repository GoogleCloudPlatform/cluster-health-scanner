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

"""Names of the telemetry metrics CHS supports."""

import enum


class MetricName(str, enum.Enum):
    PARENT_COMMAND = "PARENT_COMMAND"
    COMMAND_FLAGS = "COMMAND_FLAGS"
    STORAGE_TYPE = "STORAGE_TYPE"
    CLUSTER_TYPE = "CLUSTER_TYPE"
    MACHINE_TYPE = "MACHINE_TYPE"
    ZONE = "ZONE"
    NODE_COUNT = "NODE_COUNT"
    TEST_SUITE = "TEST_SUITE"
    TEST_NAME = "TEST_NAME"
    TEST_RESULTS = "TEST_RESULTS"
    DETECTED_ERRORS = "DETECTED_ERRORS"
    TEST_DURATION_MS = "TEST_DURATION_MS"
    EXIT_CODE = "EXIT_CODE"
    ERROR_CATEGORY = "ERROR_CATEGORY"
    IS_TEST_DATA = "IS_TEST_DATA"
