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

import os
import shutil

from chs import exceptions as ex
from chs.schedulers.base import BaseScheduler
from chs.schedulers.slurm import SlurmScheduler
from chs.schedulers.kubernetes import KubernetesScheduler


def get_scheduler(scheduler: str, **kwargs) -> BaseScheduler:
    """Factory to get the right scheduler instance."""
    if scheduler == "slurm":
        return SlurmScheduler(
            nodes=kwargs.get("nodes"),
            nodelists=kwargs.get("nodelists"),
            partitions=kwargs.get("partitions"),
        )
    if scheduler == "kubernetes":
        return KubernetesScheduler(
            nodes=kwargs.get("nodes"),
            nodepools=kwargs.get("nodepools"),
        )
    raise ex.UnsupportedSchedulerException(
        f"Unsupported scheduler: {scheduler}"
    )


def _has_kubeconfig() -> bool:
    """Checks if a valid kubeconfig exists on the system."""
    kubeconfig = os.environ.get("KUBECONFIG")
    if kubeconfig:
        for path in kubeconfig.split(os.pathsep):
            if os.path.exists(os.path.expanduser(path.strip())):
                return True
        return False
    return os.path.exists(os.path.expanduser("~/.kube/config"))


def detect_scheduler() -> str:
    """Detects the active scheduler by verifying connectivity."""

    if shutil.which("sinfo") and shutil.which("squeue"):
        return "slurm"

    if (
        os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token")
        or "KUBERNETES_SERVICE_HOST" in os.environ
    ):
        return "kubernetes"

    if shutil.which("kubectl") and _has_kubeconfig():
        return "kubernetes"

    raise ex.ChsSchedulerException(
        "Auto-detection failed: No active Slurm or Kubernetes "
        "environment detected."
    )
