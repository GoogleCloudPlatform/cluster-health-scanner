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

from chs import resources, testkit
from chs.validators.base import Validator


class TpuTopologyValidator(Validator):
    """Validates TPU topology requirements.

    On Kubernetes, single-host TPU tests support targeting individual nodes with
    --nodes, while multi-host TPU tests require whole slices via --nodepools.

    TPUs are not supported on Slurm; no TPU platform lists the Slurm scheduler,
    so TestsValidator rejects TPU tests there.
    """

    def validate(self, config: dict[str, Any]) -> list[str]:
        scheduler_obj = config.get("scheduler_obj")
        if not scheduler_obj or not resources.is_tpu(
            config.get("machine_type", "")
        ):
            return []

        scheduler_name = config.get("scheduler")
        tests = config.get("tests") or []
        nodes_arg = getattr(scheduler_obj, "nodes_arg", None)

        if scheduler_name == "kubernetes":
            if (
                nodes_arg
                and tests
                and all(testkit.requires_whole_slice(t) for t in tests)
            ):
                names = ", ".join(
                    f"'{t.get('name') if isinstance(t, dict) else t}'"
                    for t in tests
                )
                return [
                    f"TPU multi-host test(s) {names} require an entire "
                    "slice (use --nodepools)."
                ]

        return []
