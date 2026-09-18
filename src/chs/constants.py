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

"""Constants used across the CHS package."""

CHS_NAMESPACE = "chs"

SUPPORTED_DYNAMIC_SLICE_TOPOLOGY = "4x4x4"

TOPOLOGY_TO_MESH = {
    # tpu7x
    "2x2x1": "2x2x2",
    "2x2x2": "4x2x2",
    "2x2x4": "4x4x2",
    "2x4x4": "8x4x2",
    "4x4x4": "16x4x2",
    # v6e
    "2x2": "2x2",
    "2x4": "4x2",
    "4x4": "4x4",
    "4x8": "8x4",
    "8x8": "8x8",
    "8x16": "16x8",
    "16x16": "16x16",
}
