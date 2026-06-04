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

"""Validator for cluster homogeneity."""

from typing import Any

from chs.validators.base import Validator


class HomogeneityValidator(Validator):
    """Validates that the detected machine types are homogeneous."""

    def validate(self, config: dict[str, Any]) -> list[str]:
        """Validates that the detected machine types are homogeneous.

        Args:
            config: Configuration dictionary containing 'detected_machine_types'
              (list of str) and 'scheduler' (str).

        Returns:
            List of error messages if validation fails, empty list otherwise.
        """
        detected_machine_types = config.get("detected_machine_types", [])
        scheduler = config.get("scheduler")

        if len(detected_machine_types) > 1:
            error_msg = [
                "Heterogeneous cluster detected. CHS runs must target a "
                "homogeneous set of nodes.",
                f"Detected machine types: {', '.join(detected_machine_types)}",
            ]
            if scheduler == "gke":
                error_msg.append(
                    "Please target a homogeneous subset using: "
                    "--nodepools <pool_names> or --nodes <node_names>"
                )
            elif scheduler == "slurm":
                error_msg.append(
                    "Please target a homogeneous subset using: "
                    "--partitions <partition_names>, --nodelists <nodelists>, "
                    "or --nodes <node_names>"
                )
            return ["\n".join(error_msg)]
        return []
