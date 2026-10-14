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

from typing import Any

from chs.testkit import config as chs_config
from chs.testkit import loader


def build_catalog() -> dict[str, Any]:
    tests = []
    for test_name, config in loader.enabled_tests():
        plugin = loader.load_plugin(test_name)
        tests.append(
            {
                "name": test_name,
                "category": plugin.category,
                "description": plugin.description,
                "long_description": plugin.long_description,
                "platforms": [
                    {"scheduler": scheduler, "machine_type": machine_type}
                    for scheduler, machine_type in sorted(
                        chs_config.supported_platforms(config)
                    )
                ],
            }
        )
    return {"tests": tests}
