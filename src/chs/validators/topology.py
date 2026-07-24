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
from chs.validators.base import Validator


class TpuTopologyValidator(Validator):
    def validate(self, config: dict[str, Any]) -> list[str]:
        machine_type = config.get("machine_type", "")
        scheduler_obj = config.get("scheduler_obj")
        scheduler_name = config.get("scheduler")

        is_tpu = machine_type.startswith("tpu") or machine_type.startswith("ct")

        if is_tpu and scheduler_obj:
            error_msg = []

            if scheduler_name == "kubernetes":
                if getattr(scheduler_obj, "nodes_arg", None):
                    error_msg.append(
                        "TPU tests need entire slice, please use --nodepools."
                    )
                elif not getattr(scheduler_obj, "nodepools_arg", None):
                    error_msg.append(
                        "TPU tests need entire slice, please use "
                        "--nodepools to target specific nodepools."
                    )

            elif scheduler_name == "slurm":
                if getattr(scheduler_obj, "nodes_arg", None) or getattr(
                    scheduler_obj, "nodelists_arg", None
                ):
                    error_msg.append(
                        "TPU tests need entire slice, please use --partitions."
                    )
                elif not getattr(scheduler_obj, "partitions_arg", None):
                    error_msg.append(
                        "TPU tests need entire slice, please use "
                        "--partitions to target specific partitions."
                    )

            if error_msg:
                return ["\n".join(error_msg)]

        return []
