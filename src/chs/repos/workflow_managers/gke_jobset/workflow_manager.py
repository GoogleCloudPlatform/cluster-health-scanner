import os
import re
import subprocess
import sys
import tempfile
import textwrap
import uuid


from ramble.repository import ObjectTypes, paths
from ramble.wmkit import *
from ramble.experiment_result import ExperimentStatus

config_dir = (
    paths[ObjectTypes.workflow_managers]
    .repo_for_obj("gke-suite-configs")
    .dirname_for_object_name("gke-suite-configs")
)
sys.path.append(config_dir)


class GkeJobset(WorkflowManagerBase):
    """GKE workflow manager that uses JobSet CRD"""

    name = "gke-jobset"

    maintainers("bytetwin")

    tags("workflow", "gke", "jobset")
    variable(
        "processes_per_node",
        default="1",
        description="Number of processes per node",
    )
    variable("n_ranks", default="1", description="Number of ranks")

    workflow_manager_variable(
        name="job_name",
        default="{application_name}-{workload_name}-{experiment_name}",
        description="GKE job name",
    )

    workflow_manager_variable(
        name="gke_namespace",
        default="default",
        description="GKE Kubernetes namespace",
    )

    workflow_manager_variable(
        name="jobset_template",
        default="jobset_standard.yaml.tpl",
        description="Default JobSet template file",
    )

    workflow_manager_variable(
        name="mpi_command",
        default="",
        description="Default MPI command to use",
    )

    workflow_manager_variable(
        name="jobset_labels_yaml",
        default="",
        description="Raw YAML string for JobSet metadata.labels (e.g. kueue queues)",
    )

    workflow_manager_variable(
        name="jobset_annotations_yaml",
        default="",
        description="Raw YAML string for JobSet metadata.annotations",
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
        name="container_work_dir",
        default="/workspace",
        description="Path in container for experiment directory tree mapping",
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

    def _jobset_vars(self):
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

        container_work_dir = expander.expand_var_name("container_work_dir")
        workspace_dir = expander.expand_var_name("workspace_dir")
        container_image = expander.expand_var_name("container_image")

        if container_work_dir and workspace_dir:
            raw_cmd = raw_cmd.replace(workspace_dir, container_work_dir)
            experiment_run_dir = experiment_run_dir.replace(
                workspace_dir, container_work_dir
            )
            app_workdir = app_workdir.replace(workspace_dir, container_work_dir)

        workdir = app_workdir if app_workdir else experiment_run_dir

        indented_mpi_command = raw_cmd.replace("\n", "\n" + (" " * 20))
        indented_standard_command = raw_cmd.replace("\n", "\n" + (" " * 18))

        # Top-Level blocks
        jl = self._yaml_block("jobset_labels_yaml", "labels", 2).replace(
            raw_job_name, sanitized_job_name
        )
        ja = self._yaml_block(
            "jobset_annotations_yaml", "annotations", 2
        ).replace(raw_job_name, sanitized_job_name)

        # Pod Spec blocks
        pl = self._yaml_block("pod_labels_yaml", "labels", 14).replace(
            raw_job_name, sanitized_job_name
        )
        pa = self._yaml_block(
            "pod_annotations_yaml", "annotations", 14
        ).replace(raw_job_name, sanitized_job_name)
        ns = self._yaml_block("node_selector_yaml", "nodeSelector", 14).replace(
            raw_job_name, sanitized_job_name
        )
        aff = self._yaml_block("affinity_yaml", "affinity", 14).replace(
            raw_job_name, sanitized_job_name
        )
        tol = self._yaml_block("tolerations_yaml", "tolerations", 14).replace(
            raw_job_name, sanitized_job_name
        )

        c_res = self._yaml_block("container_resources_yaml", "resources", 16)
        c_sec = self._yaml_block(
            "container_security_context_yaml", "securityContext", 16
        )

        # Standard templating (where 'volumes' keys are entirely optional)
        svol = self._yaml_block("extra_volumes_yaml", "volumes", 14)
        smnt = self._yaml_block("extra_volume_mounts_yaml", "volumeMounts", 16)
        init_c = self._yaml_block("init_containers_yaml", "initContainers", 14)
        mvol_add = self._yaml_append("extra_volumes_yaml", 14)
        mmnt_add = self._yaml_append("extra_volume_mounts_yaml", 16)

        custom_res = self.app_inst.expander.expand_var_name(
            "custom_resources_yaml"
        ).replace(raw_job_name, sanitized_job_name)
        pod_claims = self._yaml_block(
            "pod_resource_claims_yaml", "resourceClaims", 14
        ).replace(raw_job_name, sanitized_job_name)

        priv_key, pub_key = self._generate_ssh_keys()
        return {
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

    register_template(
        name="batch_cancel",
        src_path="batch_cancel.tpl",
        dest_path="batch_cancel",
        extra_vars_func="jobset_vars",
    )

    register_template(
        name="batch_wait",
        src_path="batch_wait.tpl",
        dest_path="batch_wait",
        extra_vars_func="jobset_vars",
    )

    def _prepare_analysis(self, workspace):
        pass

    def get_status(self, workspace):
        """Return status of a given job"""
        return ExperimentStatus.UNKNOWN
