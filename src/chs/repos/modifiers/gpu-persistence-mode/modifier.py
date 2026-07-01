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

from ramble.modkit import *


class GpuPersistenceMode(BasicModifier):
    """Modifier to ensure NVIDIA persistence mode is enabled before execution."""

    name = "gpu-persistence-mode"

    tags("gpu")

    maintainers("bytetwin")

    mode("standard", description="Standard injection mode")
    default_mode("standard")

    register_builtin("source_pm", injection_method="prepend")

    def source_pm(self):
        return ["nvidia-smi -pm 1 || true"]
