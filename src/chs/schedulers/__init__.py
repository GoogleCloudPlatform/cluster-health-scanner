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

"""Module for managing cluster schedulers."""

from chs.schedulers.base import BaseScheduler
from chs.schedulers.default import DefaultScheduler
from chs.schedulers.slurm import SlurmScheduler
from chs.schedulers.gke import GKEScheduler


def get_scheduler(scheduler: str, **kwargs) -> BaseScheduler:
    """Factory to get the right scheduler instance."""
    app_data = kwargs.get("app_data")
    if scheduler == "slurm":
        return SlurmScheduler(
            nodes=kwargs.get("nodes"),
            nodelists=kwargs.get("nodelists"),
            partitions=kwargs.get("partitions"),
            app_data=app_data,
        )
    if scheduler == "gke":
        return GKEScheduler(
            nodes=kwargs.get("nodes"),
            nodepools=kwargs.get("nodepools"),
            app_data=app_data,
        )
    return DefaultScheduler(scheduler, app_data=app_data)
