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

from chs.plugins.tpu_collectives.renderer import TpuCollectivesRenderer
from rich.text import Text


class TpuCollectivesSingleNodeRenderer(TpuCollectivesRenderer):
    def _get_summary_title(self) -> Text:
        return Text(
            "\nTPU Collectives Single Node Performance Summary",
            style="bold underline cyan",
        )
