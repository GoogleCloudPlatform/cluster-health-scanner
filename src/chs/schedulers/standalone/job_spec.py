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

"""The job spec the standalone workflow manager writes per experiment."""

import dataclasses
import os
import re

import yaml

from chs.exceptions import ChsValidationError

SPEC_FILE = "job_spec.yaml"
SCRIPT_FILE = "job_script"

_JOB_NAME_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}")
_NODE_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*")


@dataclasses.dataclass(frozen=True)
class JobSpec:
    """One test to run on the local VM."""

    job_name: str
    node: str
    image: str
    script: str
    timeout: int
    order: int
    privileged: bool
    run_id: str
    runtime: str = ""

    def __post_init__(self):
        if not _JOB_NAME_RE.fullmatch(self.job_name):
            raise ChsValidationError(f"Invalid job name '{self.job_name}'.")
        if not _NODE_RE.fullmatch(self.node):
            raise ChsValidationError(
                f"Job '{self.job_name}' has an invalid node name '{self.node}'."
            )
        if self.timeout <= 0:
            raise ChsValidationError(
                f"Job '{self.job_name}' needs a positive timeout."
            )

    def read_script(self) -> str:
        """Returns the test's script."""
        with open(self.script) as f:
            return f.read()


def _parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text not in ("true", "false"):
        raise ValueError(f"expected true or false, got '{value}'")
    return text == "true"


def load_job_spec(experiment_dir: str) -> JobSpec:
    """Reads the spec the workflow manager wrote into experiment_dir."""
    path = os.path.join(experiment_dir, SPEC_FILE)
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return JobSpec(
            job_name=str(raw["job_name"]),
            node=str(raw["node"]),
            image=str(raw["image"]),
            script=str(raw["script"]),
            timeout=int(raw["timeout"]),
            order=int(raw["order"]),
            privileged=_parse_bool(raw["privileged"]),
            run_id=str(raw["run_id"]),
            runtime=str(raw["runtime"]),
        )
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as e:
        raise ChsValidationError(f"Invalid job spec {path}: {e}") from e
