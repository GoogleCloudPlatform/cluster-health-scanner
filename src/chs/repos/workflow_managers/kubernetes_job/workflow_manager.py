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

import yaml

from ramble.wmkit import *
from ramble.experiment_result import ExperimentStatus

from chs.constants import (
    CHS_NAMESPACE,
    COMPONENT_LABEL,
    DRA_API_GROUP,
    DRA_API_VERSION,
    GPU_CLAIM_NAME,
    GPU_CLAIM_TEMPLATE_SUFFIX,
    GANG_SCHEDULING_TIMEOUT_SECONDS,
    INSTANCE_LABEL,
    K8S_JOB_STARTUP_BUDGET_SEC,
    MANAGED_BY_CHS,
    MANAGED_BY_LABEL,
    NVIDIA_GPU_DEVICE_CLASS,
    NVIDIA_GPU_RESOURCE,
    WORKER_COMPONENT,
)

_POD_SPEC_INDENT = 6
_CONTAINER_INDENT = 8
_CONTAINER_FIELD_INDENT = 10
_GPU_RESOURCE = NVIDIA_GPU_RESOURCE
_JOB_DEADLINE_MARGIN_SEC = 60


def _job_deadline(timeout: str) -> str:
    """Returns timeout + scheduling + startup + margin; non-numbers as is."""
    if not timeout.isdigit():
        return timeout
    return str(
        int(timeout)
        + GANG_SCHEDULING_TIMEOUT_SECONDS
        + K8S_JOB_STARTUP_BUDGET_SEC
        + _JOB_DEADLINE_MARGIN_SEC
    )


class KubernetesJob(WorkflowManagerBase):
    """Kubernetes workflow manager that runs each test as an Indexed Job"""

    name = "kubernetes-job"

    maintainers("bytetwin")

    tags("workflow", "kubernetes", "job")
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
        name="job_template",
        default="job_standard.yaml.tpl",
        description="Default Job template file",
    )

    workflow_manager_variable(
        name="mpi_command",
        default="",
        description="Default MPI command to use",
    )

    workflow_manager_variable(
        name="job_labels_yaml",
        default="",
        description="Raw YAML string for Job metadata.labels",
    )

    workflow_manager_variable(
        name="suspend_job",
        default="false",
        description="Whether to create the Job in a suspended state",
    )

    workflow_manager_variable(
        name="job_annotations_yaml",
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
        name="use_gpu_dra",
        default="false",
        description="Whether to allocate GPUs through DRA instead of "
        "the device plugin",
    )

    workflow_manager_variable(
        name="pod_resource_claims_yaml",
        default="",
        description="Raw YAML for Pod spec.template.spec.resourceClaims",
    )

    def __init__(self, file_path):
        super().__init__(file_path)
        self._job_vars_by_experiment_run_dir = {}

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

    @staticmethod
    def _labels_section(user_labels_yaml, chs_labels, indent_spaces):
        """Returns a labels block of the user's labels and the CHS labels."""
        labels = {**(yaml.safe_load(user_labels_yaml) or {}), **chs_labels}
        block = yaml.safe_dump(labels, sort_keys=False).strip()
        prefix = " " * indent_spaces
        return f"{prefix}labels:\n{textwrap.indent(block, prefix + '  ')}"

    @staticmethod
    def _labeled_custom_resources(resources_yaml, chs_labels):
        """Returns the custom resources YAML with the CHS labels added."""
        resources = [r for r in yaml.safe_load_all(resources_yaml) if r]
        if not resources:
            return ""
        for resource in resources:
            metadata = resource.setdefault("metadata", {})
            metadata["labels"] = {
                **(metadata.get("labels") or {}),
                **chs_labels,
            }
        return yaml.safe_dump_all(resources, sort_keys=False) + "---"

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

    @staticmethod
    def _merge_container_resources(*resource_maps):
        """Merges container resources maps, concatenating their claims.

        Claims are deduplicated by name, keeping the first occurrence, since
        Kubernetes rejects containers that reference a claim twice.
        """
        merged = {}
        for resource_map in resource_maps:
            for key, value in (resource_map or {}).items():
                if key == "claims":
                    claims = merged.setdefault("claims", [])
                    names = {c.get("name") for c in claims}
                    for claim in value or []:
                        if claim.get("name") not in names:
                            claims.append(claim)
                            names.add(claim.get("name"))
                elif isinstance(value, dict):
                    merged.setdefault(key, {}).update(value)
                else:
                    merged[key] = value
        return merged

    def _standard_sections(self, raw_job_name, sanitized_job_name, run_labels):
        """Returns the container resources, pod claims and custom resources."""
        expander = self.app_inst.expander
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
        custom_res = self._labeled_custom_resources(
            expander.expand_var_name("custom_resources_yaml").replace(
                raw_job_name, sanitized_job_name
            ),
            run_labels,
        )
        pod_claims = self._yaml_block(
            "pod_resource_claims_yaml", "resourceClaims", _POD_SPEC_INDENT
        ).replace(raw_job_name, sanitized_job_name)
        return c_res, pod_claims, custom_res

    def _gpu_dra_sections(self, raw_job_name, sanitized_job_name, run_labels):
        """Returns the sections that request the GPUs through DRA.

        The container's nvidia.com/gpu request is replaced by a claim on a
        ResourceClaimTemplate for the same number of NVIDIA GPUs. Other
        claims, e.g. the IMEX channel of a ComputeDomain, are kept. Jobs that
        do not request nvidia.com/gpu, e.g. the root experiment, get the
        standard sections, exactly as with the device plugin.

        Returns:
          The container resources, pod resourceClaims and custom resources
          sections.

        Raises:
          ValueError: If nvidia.com/gpu is requested with a non-positive count.
        """
        expander = self.app_inst.expander

        def load(var_name):
            value = expander.expand_var_name(var_name)
            if not value or not value.strip():
                return None
            return yaml.safe_load(
                value.replace(raw_job_name, sanitized_job_name)
            )

        resources = self._merge_container_resources(
            load("container_resources_yaml"),
            load("extra_container_resources_yaml"),
        )
        gpu_counts = []
        for section in ("limits", "requests"):
            quantities = resources.get(section)
            if not isinstance(quantities, dict):
                continue
            if _GPU_RESOURCE in quantities:
                gpu_counts.append(int(quantities.pop(_GPU_RESOURCE)))
            if not quantities:
                del resources[section]
        if not gpu_counts:
            return self._standard_sections(
                raw_job_name, sanitized_job_name, run_labels
            )
        gpu_count = max(gpu_counts)
        if gpu_count <= 0:
            raise ValueError(
                f"Cannot request GPUs through DRA for Job "
                f"'{sanitized_job_name}': GPU count must be positive, "
                f"got {gpu_count}."
            )

        template_name = f"{sanitized_job_name}{GPU_CLAIM_TEMPLATE_SUFFIX}"
        container_claims = resources.setdefault("claims", [])
        if not any(c.get("name") == GPU_CLAIM_NAME for c in container_claims):
            container_claims.append({"name": GPU_CLAIM_NAME})
        pod_claims = load("pod_resource_claims_yaml") or []
        if not any(c.get("name") == GPU_CLAIM_NAME for c in pod_claims):
            pod_claims.append(
                {
                    "name": GPU_CLAIM_NAME,
                    "resourceClaimTemplateName": template_name,
                }
            )

        claim_template = {
            "apiVersion": f"{DRA_API_GROUP}/{DRA_API_VERSION}",
            "kind": "ResourceClaimTemplate",
            "metadata": {
                "name": template_name,
                "namespace": (
                    expander.expand_var_name("kubernetes_namespace")
                    or CHS_NAMESPACE
                ),
            },
            "spec": {
                "spec": {
                    "devices": {
                        "requests": [
                            {
                                "name": "gpus",
                                "exactly": {
                                    "deviceClassName": NVIDIA_GPU_DEVICE_CLASS,
                                    "allocationMode": "ExactCount",
                                    "count": gpu_count,
                                },
                            }
                        ]
                    }
                }
            },
        }
        custom_resources_yaml = expander.expand_var_name(
            "custom_resources_yaml"
        ).replace(raw_job_name, sanitized_job_name)
        custom_resources = [claim_template] + [
            r for r in yaml.safe_load_all(custom_resources_yaml) if r
        ]

        resources_yaml = yaml.safe_dump(resources, sort_keys=False).strip()
        pod_claims_yaml = yaml.safe_dump(pod_claims, sort_keys=False).strip()
        return (
            f"{' ' * _CONTAINER_INDENT}resources:\n"
            + textwrap.indent(resources_yaml, " " * _CONTAINER_FIELD_INDENT),
            f"{' ' * _POD_SPEC_INDENT}resourceClaims:\n"
            + textwrap.indent(pod_claims_yaml, " " * (_POD_SPEC_INDENT + 2)),
            self._labeled_custom_resources(
                yaml.safe_dump_all(custom_resources, sort_keys=False),
                run_labels,
            ),
        )

    def _job_vars(self) -> dict[str, str]:
        """Returns the experiment's template variables, computed once."""
        experiment_run_dir = self.app_inst.expander.expand_var_name(
            "experiment_run_dir"
        )
        if experiment_run_dir not in self._job_vars_by_experiment_run_dir:
            self._job_vars_by_experiment_run_dir[experiment_run_dir] = (
                self._compute_job_vars()
            )
        return dict(self._job_vars_by_experiment_run_dir[experiment_run_dir])

    def _compute_job_vars(self) -> dict[str, str]:
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

        indented_standard_command = raw_cmd.replace(
            "\n", "\n" + (" " * _CONTAINER_FIELD_INDENT)
        )

        run_labels = {
            MANAGED_BY_LABEL: MANAGED_BY_CHS,
            INSTANCE_LABEL: expander.expand_var_name("chs_run_id"),
        }
        worker_labels = {**run_labels, COMPONENT_LABEL: WORKER_COMPONENT}

        # Top-Level blocks
        jl = self._labels_section(
            expander.expand_var_name("job_labels_yaml").replace(
                raw_job_name, sanitized_job_name
            ),
            run_labels,
            2,
        )
        ja = self._yaml_block("job_annotations_yaml", "annotations", 2).replace(
            raw_job_name, sanitized_job_name
        )

        # Pod Spec blocks
        pl = self._labels_section(
            expander.expand_var_name("pod_labels_yaml").replace(
                raw_job_name, sanitized_job_name
            ),
            worker_labels,
            _POD_SPEC_INDENT,
        )
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

        use_gpu_dra = expander.expand_var_name("use_gpu_dra") == "true"
        if not use_gpu_dra:
            c_res, pod_claims, custom_res = self._standard_sections(
                raw_job_name, sanitized_job_name, run_labels
            )
        else:
            c_res, pod_claims, custom_res = self._gpu_dra_sections(
                raw_job_name, sanitized_job_name, run_labels
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

        priv_key, pub_key = self._generate_ssh_keys()
        suspend = expander.expand_var_name("suspend_job")
        suspend_section = "suspend: true" if suspend == "true" else ""

        return {
            "suspend_section": suspend_section,
            "custom_resources_section": custom_res,
            "pod_resource_claims_section": pod_claims,
            "job_labels_section": jl,
            "job_annotations_section": ja,
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
            "job_deadline": _job_deadline(expander.expand_var_name("timeout")),
            "sanitized_job_name": sanitized_job_name,
            "container_image": container_image,
            "indented_standard_command": indented_standard_command,
            "container_experiment_run_dir": experiment_run_dir,
            "workdir": workdir,
            "container_resources_section": c_res,
            "container_security_context_section": c_sec,
            "ssh_private_key_indented": textwrap.indent(priv_key, "    "),
            "ssh_public_key_indented": textwrap.indent(pub_key, "    "),
        }

    register_template(
        name="job_yaml",
        src_path="{job_template}",
        dest_path="job.yaml",
        extra_vars_func="job_vars",
    )

    register_template(
        name="batch_submit",
        src_path="batch_submit.tpl",
        dest_path="batch_submit",
        extra_vars_func="job_vars",
    )

    register_template(
        name="execute_experiment",
        src_path="batch_submit.tpl",
        dest_path="execute_experiment",
        extra_vars_func="job_vars",
    )

    def _prepare_analysis(self, workspace):
        pass

    def get_status(self, workspace):
        """Return status of a given job"""
        return ExperimentStatus.UNKNOWN
