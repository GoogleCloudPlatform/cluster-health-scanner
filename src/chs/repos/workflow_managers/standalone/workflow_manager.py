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

import os
import re
import uuid

import yaml

from ramble.wmkit import *
from ramble.experiment_result import ExperimentStatus

from chs.schedulers.standalone.job_spec import SCRIPT_FILE, SPEC_FILE

_MAX_JOB_NAME_BASE_LEN = 100


class Standalone(WorkflowManagerBase):
    """Workflow manager that writes a job spec and script per experiment."""

    name = "standalone"

    tags("workflow", "standalone")

    workflow_manager_variable(
        name="job_name",
        default="{chs_run_id}-{experiment_name}",
        description="Base name of the job and its container",
    )
    workflow_manager_variable(
        name="container_image",
        default="",
        description="Container image of the test",
    )
    workflow_manager_variable(
        name="privileged",
        default="",
        description="Whether the test's container is privileged (true/false)",
    )
    workflow_manager_variable(
        name="runtime",
        default="",
        description="Native runtime of the test; empty runs a container",
    )
    workflow_manager_variable(
        name="batch_submit",
        default="{experiment_run_dir}/batch_submit",
        description="Placeholder path written to all_experiments",
    )

    def _job_vars(self) -> dict[str, str]:
        """Returns the experiment's template variables."""
        expander = self.app_inst.expander
        run_dir = expander.expand_var_name("experiment_run_dir")
        base_name = re.sub(
            r"[^a-zA-Z0-9_.-]+", "-", expander.expand_var_name("job_name")
        )[:_MAX_JOB_NAME_BASE_LEN].strip("-._")
        command = re.sub(
            r'>>\s*"[^"]+"\s*2>&1', "", expander.expand_var_name("command")
        )
        spec = {
            "job_name": f"{base_name}-{uuid.uuid4().hex[:6]}",
            "node": expander.expand_var_name("node"),
            "image": expander.expand_var_name("container_image"),
            "script": os.path.join(run_dir, SCRIPT_FILE),
            "timeout": expander.expand_var_name("timeout"),
            "order": expander.expand_var_name("experiment_index"),
            "privileged": expander.expand_var_name("privileged") or "false",
            "run_id": expander.expand_var_name("chs_run_id"),
            "runtime": expander.expand_var_name("runtime"),
        }
        return {
            "job_spec_yaml": yaml.safe_dump(spec, sort_keys=False),
            "standalone_command": command,
        }

    register_template(
        name="job_spec",
        src_path="job_spec.yaml.tpl",
        dest_path=SPEC_FILE,
        extra_vars_func="job_vars",
    )
    register_template(
        name="job_script",
        src_path="job_script.tpl",
        dest_path=SCRIPT_FILE,
        extra_vars_func="job_vars",
    )

    def _prepare_analysis(self, workspace):
        pass

    def get_status(self, workspace):
        return ExperimentStatus.UNKNOWN
