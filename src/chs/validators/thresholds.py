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

"""Validator for thresholds configuration."""

import os
from typing import Any
import yaml

from chs.validators.base import Validator


class ThresholdsValidator(Validator):
    """Validates the thresholds configuration file."""

    def validate(self, config: dict[str, Any]) -> list[str]:
        """Validates the thresholds file specified in the configuration.

        Args:
            config: Configuration dictionary containing 'thresholds_path'.

        Returns:
            List of error messages if validation fails, empty list otherwise.
        """
        thresholds_path = config.get("thresholds_path")
        if not thresholds_path:
            return []

        if not os.path.exists(thresholds_path):
            return [f"Thresholds file not found at {thresholds_path}"]

        errors = []
        try:
            with open(thresholds_path, "r") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            return [f"Failed to parse YAML in {thresholds_path}: {e}"]
        except OSError as e:
            return [f"Failed to read {thresholds_path}: {e}"]

        if not isinstance(data, dict):
            return ["Thresholds data must be a dictionary"]

        for machine_type, tests in data.items():
            if not isinstance(tests, dict):
                errors.append(
                    f"Tests for machine type '{machine_type}' "
                    f"must be a dictionary"
                )
                continue

            for test_name, cfg in tests.items():
                if not isinstance(cfg, dict):
                    errors.append(
                        f"Config for test '{test_name}' must be a dictionary"
                    )
                    continue

                if "type" not in cfg:
                    errors.append(
                        f"Missing 'type' in config for test '{test_name}'"
                    )
                elif cfg["type"] not in ["min", "relative", "scalar"]:
                    errors.append(
                        f"Invalid type '{cfg['type']}' in test '{test_name}'"
                    )

                if "thresholds" not in cfg:
                    errors.append(
                        f"Missing 'thresholds' in config for test '{test_name}'"
                    )
                elif not isinstance(cfg["thresholds"], dict):
                    errors.append(
                        f"'thresholds' in test '{test_name}' "
                        f"must be a dictionary"
                    )
                else:
                    for metric, val in cfg["thresholds"].items():
                        if not isinstance(val, (int, float, bool, str)):
                            errors.append(
                                f"Invalid threshold value for "
                                f"metric '{metric}' "
                                f"in test '{test_name}'"
                            )

        return errors
