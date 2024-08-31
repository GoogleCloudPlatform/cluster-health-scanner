# Copyright 2024 Google LLC
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

"""Methods for health check metrics.
"""

from typing import Any, Dict


def log_dict(
    test_name: str,
    did_pass: bool,
    node_name: str,
    result_data: Dict[str, Any],
) -> Dict[str, Any]:
  return {
      "test_name": test_name,
      "did_pass": did_pass,
      "node_name": node_name,
      "result_data": result_data,
  }
