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
import subprocess

import tempfile
import textwrap
import uuid


from ramble.wmkit import *
from ramble.experiment_result import ExperimentStatus

_POD_SPEC_INDENT = 6
_CONTAINER_INDENT = 8
_CONTAINER_FIELD_INDENT = 10

_WORKER_LABEL = "chs.gke.io/worker"
_DNS_WAIT_TIMEOUT_SECONDS = 300


class KubernetesJobset(WorkflowManagerBase):
    """Kubernetes workflow manager that runs each test as an Indexed Job"""

    name = "kubernetes-jobset"

    maintainers("bytetwin")

    tags("workflow", "kubernetes", "jobset")
    variable(
        "processes_per_node",
        default="1",
        description="Number of processes per node",
    )
    variable("n_ranks", default="1", description="Number of ranks")

    workflow_manager_variable(
        name="job_name",
        default="{chs_run_id}-{application_name}-{workload_name}",
        description="Kubernetes Job resource base name",
    )

    workflow_manager_variable(
        name="kubernetes_namespace",
        default="default",
        description="Kubernetes namespace",
    )

    workflow_manager_variable(
        name="jobset_template",
        default="jobset_standard.yaml.tpl",
        description="Default Job template file",
    )

    workflow_manager_variable(
        name="mpi_command",
        default="",
        description="Default MPI command to use",
    )

    workflow_manager_variable(
        name="jobset_labels_yaml",
        default="",
        description="Raw YAML string for Job metadata.labels",
    )

    workflow_manager_variable(
        name="suspend_jobset",
        default="false",
        description="Whether to create the Job in a suspended state",
    )

    workflow_manager_variable(
        name="jobset_annotations_yaml",
        default="",
        description="Raw YAML string for Job metadata.annotations",
    )

    workflow_manager_variable(
        name="pod_labels_yaml",
        default="",
        description="Raw YAML string for Pod spec.template.metadata.labels",
    )

    workflow_manager_variable(
        name="pod_annotations_yaml",
        default="",
        description="Raw YAML string for Pod spec.template.metadata.annotations",
    )

    workflow_manager_variable(
        name="extra_pod_annotations_yaml",
        default="",
        description="Raw YAML string for extra Pod spec.template.metadata.annotations",
    )

    workflow_manager_variable(
        name="node_selector_yaml",
        default="",
        description="Raw YAML string for spec.template.spec.nodeSelector",
    )

    workflow_manager_variable(
        name="affinity_yaml",
        default="",
        description="Raw YAML string for Pod spec.affinity",
    )

    workflow_manager_variable(
        name="tolerations_yaml",
        default="",
        description="Raw YAML string for spec.template.spec.tolerations",
    )

    workflow_manager_variable(
        name="extra_volume_mounts_yaml",
        default="",
        description="Raw YAML string for extra container volumeMounts",
    )

    workflow_manager_variable(
        name="extra_volumes_yaml",
        default="",
        description="Raw YAML string for extra pod volumes",
    )

    workflow_manager_variable(
        name="init_containers_yaml",
        default="",
        description="Raw YAML string for extra initContainers",
    )

    workflow_manager_variable(
        name="container_image",
        default="",
        description="url to the container image",
    )

    workflow_manager_variable(
        name="ssh_port",
        default="222",
        description="SSH port used for MPI host discovery",
    )

    workflow_manager_variable(
        name="ssh_secret_name",
        default="mpi-ssh",
        description="Kubernetes secret name containing ssh keys",
    )

    workflow_manager_variable(
        name="app_workdir",
        default="",
        description="Optional working directory override defaults to experiment dir",
    )

    workflow_manager_variable(
        name="container_resources_yaml",
        default="",
        description="Raw YAML for container resources (e.g. nvidia.com/gpu)",
    )

    workflow_manager_variable(
        name="extra_container_resources_yaml",
        default="",
        description="Raw YAML string for extra container resources (e.g. claims)",
    )

    workflow_manager_variable(
        name="container_security_context_yaml",
        default="",
        description="Raw YAML for container securityContext (e.g. privileged)",
    )

    workflow_manager_variable(
        name="custom_resources_yaml",
        default="",
        description="Raw YAML for extra top-level Kubernetes resources",
    )

    workflow_manager_variable(
        name="pod_resource_claims_yaml",
        default="",
        description="Raw YAML for Pod spec.template.spec.resourceClaims",
    )

    def __init__(self, file_path):
        super().__init__(file_path)
        self._jobset_vars_by_experiment_run_dir = {}

    def _yaml_block(self, var_name, key_name, indent_spaces):
        """Constructs a full YAML key-value block with proper indentation."""
        val = self.app_inst.expander.expand_var_name(var_name)
        if not val or not val.strip():
            return ""
        prefix = " " * indent_spaces
        key_line = f"{prefix}{key_name}:"
        block = textwrap.indent(val.strip(), prefix + "  ")
        return f"{key_line}\n{block}"

    def _yaml_append(self, var_name, indent_spaces):
        """Appends list items without wrapping them in a top-level key."""
        val = self.app_inst.expander.expand_var_name(var_name)
        if not val or not val.strip():
            return ""
        return textwrap.indent(val.strip(), " " * indent_spaces)

    def _generate_ssh_keys(self):
        priv_key = ""
        pub_key = ""
        try:
            with tempfile.TemporaryDirectory() as td:
                key_path = os.path.join(td, "id_ed25519")
                subprocess.run(
                    [
                        "ssh-keygen",
                        "-t",
                        "ed25519",
                        "-q",
                        "-N",
                        "",
                        "-f",
                        key_path,
                    ],
                    check=True,
                    capture_output=True,
                )
                with open(key_path, "r") as f:
                    priv_key = f.read().strip()
                with open(key_path + ".pub", "r") as f:
                    pub_key = f.read().strip()
        except Exception as e:
            print(f"Warning: Failed to generate SSH keys: {e}")

        return priv_key, pub_key

    def _jobset_vars(self) -> dict[str, str]:
        """Returns the experiment's template variables, computed once."""
        experiment_run_dir = self.app_inst.expander.expand_var_name(
            "experiment_run_dir"
        )
        if experiment_run_dir not in self._jobset_vars_by_experiment_run_dir:
            self._jobset_vars_by_experiment_run_dir[experiment_run_dir] = (
                self._compute_jobset_vars()
            )
        return dict(self._jobset_vars_by_experiment_run_dir[experiment_run_dir])

    def _pod_labels_section(
        self, raw_job_name: str, sanitized_job_name: str
    ) -> str:
        """Returns the pod labels block, always including the CHS worker label."""
        user_labels = self._yaml_append(
            "pod_labels_yaml", _CONTAINER_INDENT
        ).replace(raw_job_name, sanitized_job_name)
        chs_labels = f'{" " * _CONTAINER_INDENT}{_WORKER_LABEL}: "true"'
        labels = "\n".join(s for s in [user_labels, chs_labels] if s)
        return f"{' ' * _POD_SPEC_INDENT}labels:\n{labels}"

    def _compute_jobset_vars(self) -> dict[str, str]:
        expander = self.app_inst.expander
        raw_job_name = expander.expand_var_name("job_name")
        # K8S resource names can be max 63 chars and no `_` is supported
        sanitized_job_name = raw_job_name.replace("_", "-").lower()

        u = uuid.uuid4().hex[:6]
        max_base_len = 40

        if len(sanitized_job_name) > max_base_len:
            sanitized_job_name = (
                sanitized_job_name[:max_base_len].strip("-") + "-" + u
            )
        else:
            sanitized_job_name = sanitized_job_name.strip("-") + "-" + u

        raw_cmd = expander.expand_var_name("command")
        raw_cmd = re.sub(r'>>\s*"[^"]+"\s*2>&1', "", raw_cmd)

        experiment_run_dir = expander.expand_var_name("experiment_run_dir")
        app_workdir = expander.expand_var_name("app_workdir")

        container_image = expander.expand_var_name("container_image")

        workdir = app_workdir if app_workdir else experiment_run_dir

        indented_mpi_command = raw_cmd.replace(
            "\n", "\n" + (" " * (_CONTAINER_FIELD_INDENT + 2))
        )
        indented_standard_command = raw_cmd.replace(
            "\n", "\n" + (" " * _CONTAINER_FIELD_INDENT)
        )

        # Top-Level blocks
        jl = self._yaml_block("jobset_labels_yaml", "labels", 2).replace(
            raw_job_name, sanitized_job_name
        )
        ja = self._yaml_block(
            "jobset_annotations_yaml", "annotations", 2
        ).replace(raw_job_name, sanitized_job_name)

        # Pod Spec blocks
        pl = self._pod_labels_section(raw_job_name, sanitized_job_name)
        base_pa = self._yaml_append("pod_annotations_yaml", _CONTAINER_INDENT)
        extra_pa = self._yaml_append(
            "extra_pod_annotations_yaml", _CONTAINER_INDENT
        )
        all_pa = "\n".join(s for s in [base_pa, extra_pa] if s).replace(
            raw_job_name, sanitized_job_name
        )
        pa = f"{' ' * _POD_SPEC_INDENT}annotations:\n{all_pa}" if all_pa else ""
        ns = self._yaml_block(
            "node_selector_yaml", "nodeSelector", _POD_SPEC_INDENT
        ).replace(raw_job_name, sanitized_job_name)
        aff = self._yaml_block(
            "affinity_yaml", "affinity", _POD_SPEC_INDENT
        ).replace(raw_job_name, sanitized_job_name)
        tol = self._yaml_block(
            "tolerations_yaml", "tolerations", _POD_SPEC_INDENT
        ).replace(raw_job_name, sanitized_job_name)

        base_cr = self._yaml_append(
            "container_resources_yaml", _CONTAINER_FIELD_INDENT
        )
        extra_cr = self._yaml_append(
            "extra_container_resources_yaml", _CONTAINER_FIELD_INDENT
        )
        all_cr = "\n".join(s for s in [base_cr, extra_cr] if s)
        c_res = (
            f"{' ' * _CONTAINER_INDENT}resources:\n{all_cr}" if all_cr else ""
        )
        c_sec = self._yaml_block(
            "container_security_context_yaml",
            "securityContext",
            _CONTAINER_INDENT,
        )

        # Standard templating (where 'volumes' keys are entirely optional)
        svol = self._yaml_block(
            "extra_volumes_yaml", "volumes", _POD_SPEC_INDENT
        )
        smnt = self._yaml_block(
            "extra_volume_mounts_yaml", "volumeMounts", _CONTAINER_INDENT
        )
        init_c = self._yaml_block(
            "init_containers_yaml", "initContainers", _POD_SPEC_INDENT
        )
        mvol_add = self._yaml_append("extra_volumes_yaml", _POD_SPEC_INDENT)
        mmnt_add = self._yaml_append(
            "extra_volume_mounts_yaml", _CONTAINER_INDENT
        )

        custom_res = self.app_inst.expander.expand_var_name(
            "custom_resources_yaml"
        ).replace(raw_job_name, sanitized_job_name)
        pod_claims = self._yaml_block(
            "pod_resource_claims_yaml", "resourceClaims", _POD_SPEC_INDENT
        ).replace(raw_job_name, sanitized_job_name)

        priv_key, pub_key = self._generate_ssh_keys()
        suspend = expander.expand_var_name("suspend_jobset")
        suspend_section = "suspend: true" if suspend == "true" else ""

        return {
            "suspend_section": suspend_section,
            "custom_resources_section": custom_res,
            "pod_resource_claims_section": pod_claims,
            "jobset_labels_section": jl,
            "jobset_annotations_section": ja,
            "pod_labels_section": pl,
            "pod_annotations_section": pa,
            "node_selector_section": ns,
            "affinity_section": aff,
            "tolerations_section": tol,
            "standard_volumes_section": svol,
            "standard_volume_mounts_section": smnt,
            "init_containers_section": init_c,
            "mpi_volumes_append": mvol_add,
            "mpi_volume_mounts_append": mmnt_add,
            "dns_wait_timeout": str(_DNS_WAIT_TIMEOUT_SECONDS),
            "sanitized_job_name": sanitized_job_name,
            "container_image": container_image,
            "indented_mpi_command": indented_mpi_command,
            "indented_standard_command": indented_standard_command,
            "container_experiment_run_dir": experiment_run_dir,
            "workdir": workdir,
            "container_resources_section": c_res,
            "container_security_context_section": c_sec,
            "ssh_private_key_indented": textwrap.indent(priv_key, "    "),
            "ssh_public_key_indented": textwrap.indent(pub_key, "    "),
        }

    register_template(
        name="jobset_yaml",
        src_path="{jobset_template}",
        dest_path="jobset.yaml",
        extra_vars_func="jobset_vars",
    )

    register_template(
        name="batch_submit",
        src_path="batch_submit.tpl",
        dest_path="batch_submit",
        extra_vars_func="jobset_vars",
    )

    register_template(
        name="execute_experiment",
        src_path="batch_submit.tpl",
        dest_path="execute_experiment",
        extra_vars_func="jobset_vars",
    )

    def _prepare_analysis(self, workspace):
        pass

    def get_status(self, workspace):
        """Return status of a given job"""
        return ExperimentStatus.UNKNOWN
