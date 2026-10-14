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

"""Unit tests for the kubernetes-jobset workflow manager."""

import os
import re
import sys
import unittest
from unittest import mock

import yaml

# Mock ramble module dependencies
ramble_mock = mock.Mock()
ramble_mock.wmkit = mock.Mock()


class _MockWorkflowManagerBase:
    def __init__(self, *args, **kwargs):
        pass


ramble_mock.wmkit.WorkflowManagerBase = _MockWorkflowManagerBase
ramble_mock.wmkit.workflow_manager_variable = mock.Mock()
ramble_mock.wmkit.maintainers = mock.Mock()
ramble_mock.wmkit.tags = mock.Mock()
ramble_mock.wmkit.variable = mock.Mock()
ramble_mock.wmkit.register_template = mock.Mock()
ramble_mock.wmkit.__all__ = [
    "WorkflowManagerBase",
    "workflow_manager_variable",
    "maintainers",
    "tags",
    "variable",
    "register_template",
]
ramble_mock.experiment_result = mock.Mock()
sys.modules["ramble"] = ramble_mock
sys.modules["ramble.wmkit"] = ramble_mock.wmkit
sys.modules["ramble.experiment_result"] = ramble_mock.experiment_result

from chs.repos.workflow_managers.kubernetes_jobset.workflow_manager import (  # noqa: E402
    KubernetesJobset,
)

_WORKER_SERVICE_NAME = "chs-run1"
_RUN_LABELS = {
    "app.kubernetes.io/managed-by": "chs",
    "app.kubernetes.io/instance": "run1",
}
_WORKER_LABELS = {**_RUN_LABELS, "app.kubernetes.io/component": "worker"}
_JOB_LABELS = {**_RUN_LABELS, "team": "infra"}
_POD_LABELS = {**_WORKER_LABELS, "team": "infra"}


_JOBSET_VARS_INPUTS = {
    "job_name": "chs-run1-nccl-tests-all-gather",
    "command": "mpirun -n 16 all_gather_perf",
    "experiment_run_dir": "/workspace/experiments/nccl/all-gather/exp1",
    "container_image": "image:latest",
    "pod_annotations_yaml": "networking.gke.io/default-interface: eth0",
    "node_selector_yaml": "cloud.google.com/gke-nodepool: pool-1",
    "tolerations_yaml": (
        "- key: nvidia.com/gpu\n  operator: Exists\n  effect: NoSchedule"
    ),
    "init_containers_yaml": (
        "- name: tcpxo-daemon\n  image: daemon:latest\n"
        "  restartPolicy: Always"
    ),
    "container_resources_yaml": "limits:\n  nvidia.com/gpu: 8",
    "suspend_jobset": "true",
    "chs_run_id": "run1",
    "jobset_labels_yaml": "app.kubernetes.io/managed-by: chs\nteam: infra",
    # CHS labels win over user labels with the same key.
    "pod_labels_yaml": "app.kubernetes.io/instance: other\nteam: infra",
}

_TEMPLATE_PLACEHOLDER = re.compile(r"(?<![$\w])\{(\w+)\}")


def _workflow_manager(var_map: dict[str, str]) -> KubernetesJobset:
    wm = KubernetesJobset(mock.Mock())
    wm.app_inst = mock.Mock()
    wm.app_inst.expander.expand_var_name.side_effect = lambda name: (
        var_map.get(name, "")
    )
    return wm


class KubernetesJobsetTest(unittest.TestCase):
    """Tests for the kubernetes-jobset workflow manager."""

    def setUp(self):
        super().setUp()
        self.generate_ssh_keys = self.enterContext(
            mock.patch.object(
                KubernetesJobset,
                "_generate_ssh_keys",
                autospec=True,
                return_value=("private-key", "public-key"),
            )
        )

    def test_kubernetes_jobset_extra_annotations_and_resources(self):
        wm = KubernetesJobset(mock.Mock())
        wm.app_inst = mock.Mock()

        var_map = {
            "job_name": "test-job",
            "command": "echo test",
            "experiment_run_dir": "/tmp/exp",
            "app_workdir": "",
            "container_image": "image:latest",
            "jobset_labels_yaml": "",
            "jobset_annotations_yaml": "",
            "pod_labels_yaml": "",
            "pod_annotations_yaml": "networking.gke.io/default-interface: eth0",
            "extra_pod_annotations_yaml": (
                "compute-domain.nvidia.com/domain-name: cd"
            ),
            "node_selector_yaml": "",
            "affinity_yaml": "",
            "tolerations_yaml": "",
            "container_resources_yaml": (
                "limits:\n  nvidia.com/gpu: 4\nrequests:\n  nvidia.com/gpu: 4"
            ),
            "extra_container_resources_yaml": (
                "claims:\n- name: compute-domain-channel"
            ),
            "container_security_context_yaml": "",
            "extra_volumes_yaml": "",
            "extra_volume_mounts_yaml": "",
            "init_containers_yaml": "",
            "custom_resources_yaml": "",
            "pod_resource_claims_yaml": "",
            "suspend_jobset": "false",
        }
        wm.app_inst.expander.expand_var_name.side_effect = (
            lambda k: var_map.get(k, "")
        )
        res = wm._jobset_vars()

        # Check annotations merged
        self.assertIn(
            "networking.gke.io/default-interface: eth0",
            res["pod_annotations_section"],
        )
        self.assertIn(
            "compute-domain.nvidia.com/domain-name: cd",
            res["pod_annotations_section"],
        )

        # Check container resources merged
        self.assertIn(
            "limits:\n            nvidia.com/gpu: 4",
            res["container_resources_section"],
        )
        self.assertIn(
            "claims:\n          - name: compute-domain-channel",
            res["container_resources_section"],
        )

        # When extra fields are empty
        var_map["extra_pod_annotations_yaml"] = ""
        var_map["extra_container_resources_yaml"] = ""
        res_empty = wm._compute_jobset_vars()
        self.assertIn(
            "networking.gke.io/default-interface: eth0",
            res_empty["pod_annotations_section"],
        )
        self.assertNotIn(
            "compute-domain.nvidia.com", res_empty["pod_annotations_section"]
        )
        self.assertNotIn("claims:", res_empty["container_resources_section"])

    def test_jobset_vars_are_computed_once_per_experiment(self):
        """All templates of one experiment share one Job name and key."""
        wm = _workflow_manager(dict(_JOBSET_VARS_INPUTS))

        rendered_vars = [wm._jobset_vars() for _ in range(5)]

        self.generate_ssh_keys.assert_called_once()
        self.assertEqual(
            {v["sanitized_job_name"] for v in rendered_vars},
            {rendered_vars[0]["sanitized_job_name"]},
        )
        self.assertTrue(all(v == rendered_vars[0] for v in rendered_vars))

    def test_jobset_vars_differ_between_experiments(self):
        """Different experiments never share a Job name or key pair."""
        var_map = dict(_JOBSET_VARS_INPUTS)
        wm = _workflow_manager(var_map)
        self.generate_ssh_keys.side_effect = [
            ("private-key-1", "public-key-1"),
            ("private-key-2", "public-key-2"),
        ]

        first = wm._jobset_vars()
        var_map["experiment_run_dir"] = (
            "/workspace/experiments/nccl/all-gather/exp2"
        )
        second = wm._jobset_vars()

        self.assertEqual(self.generate_ssh_keys.call_count, 2)
        self.assertNotEqual(
            first["sanitized_job_name"], second["sanitized_job_name"]
        )
        self.assertNotEqual(
            first["ssh_private_key_indented"],
            second["ssh_private_key_indented"],
        )

    def test_returned_jobset_vars_do_not_alias_the_cache(self):
        """Mutating returned variables does not leak into later templates."""
        wm = _workflow_manager(dict(_JOBSET_VARS_INPUTS))
        job_name = wm._jobset_vars()["sanitized_job_name"]

        wm._jobset_vars()["sanitized_job_name"] = "mutated"

        self.assertEqual(wm._jobset_vars()["sanitized_job_name"], job_name)

    def _render(self, template_name, var_map, **extra_vars):
        """Renders a kubernetes-jobset template like Ramble does."""
        wm = _workflow_manager(var_map)
        template_vars = {
            "kubernetes_namespace": "chs",
            "n_nodes": var_map.get("n_nodes", "2"),
            "timeout": "3600",
            "ssh_port": "222",
            "processes_per_node": "8",
            "worker_service_name": _WORKER_SERVICE_NAME,
            **extra_vars,
            **wm._jobset_vars(),
        }
        template_path = os.path.join(
            os.path.dirname(sys.modules[KubernetesJobset.__module__].__file__),
            template_name,
        )
        with open(template_path) as template_file:
            template = template_file.read()
        rendered = _TEMPLATE_PLACEHOLDER.sub(
            lambda match: template_vars[match.group(1)], template
        )
        return template_vars, [d for d in yaml.safe_load_all(rendered) if d]

    def _assert_job_spec(self, job, job_name, n_nodes):
        self.assertEqual(job["apiVersion"], "batch/v1")
        self.assertEqual(job["kind"], "Job")
        self.assertEqual(job["metadata"]["name"], job_name)
        self.assertEqual(job["metadata"]["namespace"], "chs")
        self.assertEqual(job["metadata"]["labels"], _JOB_LABELS)
        spec = job["spec"]
        self.assertEqual(
            {k: v for k, v in spec.items() if k != "template"},
            {
                "suspend": True,
                "completionMode": "Indexed",
                "backoffLimit": 0,
                "podFailurePolicy": {
                    "rules": [
                        {
                            "action": "FailJob",
                            "onExitCodes": {
                                "operator": "NotIn",
                                "values": [0],
                            },
                        }
                    ]
                },
                "parallelism": n_nodes,
                "completions": n_nodes,
                "activeDeadlineSeconds": 3600,
                "ttlSecondsAfterFinished": 86400,
            },
        )
        self.assertEqual(spec["template"]["metadata"]["labels"], _POD_LABELS)
        pod_spec = spec["template"]["spec"]
        self.assertEqual(pod_spec["subdomain"], _WORKER_SERVICE_NAME)
        self.assertEqual(pod_spec["restartPolicy"], "Never")
        self.assertEqual(pod_spec["terminationGracePeriodSeconds"], 5)
        return pod_spec

    def test_standard_template_renders_one_indexed_job(self):
        """Golden structure of the standard template for a TPU test."""
        var_map = {
            **_JOBSET_VARS_INPUTS,
            "command": "echo one\necho two",
            "n_nodes": "2",
            "affinity_yaml": (
                "nodeAffinity:\n"
                "  requiredDuringSchedulingIgnoredDuringExecution:\n"
                "    nodeSelectorTerms:\n"
                "    - matchExpressions:\n"
                "      - key: kubernetes.io/hostname\n"
                "        operator: In\n"
                "        values:\n"
                "        - node-1"
            ),
            "container_security_context_yaml": "privileged: true",
            "extra_volumes_yaml": "- name: shm\n  emptyDir: {}",
            "extra_volume_mounts_yaml": "- name: shm\n  mountPath: /dev/shm",
            "pod_resource_claims_yaml": (
                "- name: claim\n  resourceClaimTemplateName: tmpl"
            ),
        }

        template_vars, docs = self._render("jobset_standard.yaml.tpl", var_map)

        job_name = template_vars["sanitized_job_name"]
        self.assertEqual(len(docs), 1)
        pod_spec = self._assert_job_spec(docs[0], job_name, 2)
        self.assertEqual(
            docs[0]["spec"]["template"]["metadata"]["annotations"],
            {"networking.gke.io/default-interface": "eth0"},
        )
        self.assertEqual(
            pod_spec["nodeSelector"],
            {"cloud.google.com/gke-nodepool": "pool-1"},
        )
        self.assertEqual(
            pod_spec["tolerations"],
            [
                {
                    "key": "nvidia.com/gpu",
                    "operator": "Exists",
                    "effect": "NoSchedule",
                }
            ],
        )
        self.assertEqual(
            pod_spec["affinity"]["nodeAffinity"][
                "requiredDuringSchedulingIgnoredDuringExecution"
            ]["nodeSelectorTerms"][0]["matchExpressions"][0]["values"],
            ["node-1"],
        )
        self.assertEqual(
            pod_spec["resourceClaims"],
            [{"name": "claim", "resourceClaimTemplateName": "tmpl"}],
        )
        self.assertEqual(pod_spec["volumes"], [{"name": "shm", "emptyDir": {}}])
        (container,) = pod_spec["containers"]
        self.assertEqual(container["image"], "image:latest")
        self.assertEqual(container["securityContext"], {"privileged": True})
        self.assertEqual(
            container["resources"], {"limits": {"nvidia.com/gpu": 8}}
        )
        self.assertEqual(
            container["volumeMounts"],
            [{"name": "shm", "mountPath": "/dev/shm"}],
        )
        self.assertEqual(
            container["env"],
            [
                {
                    "name": "MY_NODE_NAME",
                    "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
                },
            ],
        )
        self.assertEqual(container["command"][:2], ["bash", "-c"])
        self.assertTrue(
            container["command"][2].endswith("echo one\necho two\n")
        )

    def test_mpi_template_renders_secret_and_one_indexed_job(self):
        """Golden structure of the MPI template."""
        var_map = {
            **_JOBSET_VARS_INPUTS,
            "command": "mpirun -n 16 \\\n  all_gather_perf",
            "n_nodes": "3",
            "extra_volumes_yaml": "- name: shm\n  emptyDir: {}",
            "extra_volume_mounts_yaml": "- name: shm\n  mountPath: /dev/shm",
        }

        template_vars, docs = self._render("jobset_mpi.yaml.tpl", var_map)

        job_name = template_vars["sanitized_job_name"]
        self.assertEqual([d["kind"] for d in docs], ["Secret", "Job"])
        self.assertEqual(docs[0]["metadata"]["name"], f"{job_name}-ssh")
        self.assertEqual(docs[0]["metadata"]["labels"], _JOB_LABELS)
        pod_spec = self._assert_job_spec(docs[1], job_name, 3)
        self.assertEqual(
            [c["name"] for c in pod_spec["initContainers"]], ["tcpxo-daemon"]
        )
        self.assertEqual(
            [v["name"] for v in pod_spec["volumes"]], ["mpi-id", "shm"]
        )
        self.assertEqual(
            pod_spec["volumes"][0]["secret"]["secretName"], f"{job_name}-ssh"
        )
        (container,) = pod_spec["containers"]
        self.assertEqual(
            [m["name"] for m in container["volumeMounts"]], ["mpi-id", "shm"]
        )
        self.assertEqual(
            [e["name"] for e in container["env"]],
            [
                "OMPI_ALLOW_RUN_AS_ROOT",
                "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM",
                "MY_NODE_NAME",
            ],
        )
        script = container["command"][2]
        self.assertIn(
            f'OTHER="{job_name}-${{i}}.{_WORKER_SERVICE_NAME}"', script
        )
        self.assertIn(
            f'HEAD_NODE="{job_name}-0.{_WORKER_SERVICE_NAME}"', script
        )
        self.assertNotIn("getent", script)
        # The launcher's ssh discovery is bounded only by the Job's
        # activeDeadlineSeconds.
        discovery = script[
            script.index("Discovering hosts") : script.index("STARTING MPI")
        ]
        self.assertNotIn("DNS_DEADLINE", discovery)
        self.assertIn("$(seq 0 $(( 3 - 1 )))", script)
        self.assertNotIn("POD_NAME", script)
        self.assertIn("\n  mpirun -n 16 \\\n    all_gather_perf\n", script)

    def test_custom_resources_get_run_labels(self):
        """Custom resources keep their labels and get the run's labels."""
        raw_job_name = _JOBSET_VARS_INPUTS["job_name"]
        var_map = {
            **_JOBSET_VARS_INPUTS,
            "custom_resources_yaml": (
                "apiVersion: resource.nvidia.com/v1beta1\n"
                "kind: ComputeDomain\n"
                "metadata:\n"
                f"  name: {raw_job_name}-cd\n"
                "  namespace: chs\n"
                "  labels:\n"
                f"    chs-run: {raw_job_name}\n"
                "spec:\n"
                "  numNodes: 2\n"
                "---\n"
            ),
        }

        template_vars, docs = self._render("jobset_mpi.yaml.tpl", var_map)

        job_name = template_vars["sanitized_job_name"]
        self.assertEqual(
            [d["kind"] for d in docs], ["ComputeDomain", "Secret", "Job"]
        )
        self.assertEqual(docs[0]["metadata"]["name"], f"{job_name}-cd")
        self.assertEqual(
            docs[0]["metadata"]["labels"], {"chs-run": job_name, **_RUN_LABELS}
        )
        self.assertEqual(docs[0]["spec"], {"numNodes": 2})

    def test_labels_without_user_labels(self):
        """Without user labels, the objects carry only the CHS labels."""
        var_map = {
            **_JOBSET_VARS_INPUTS,
            "jobset_labels_yaml": "",
            "pod_labels_yaml": "",
        }

        _, docs = self._render("jobset_mpi.yaml.tpl", var_map)

        secret, job = docs
        self.assertEqual(secret["metadata"]["labels"], _RUN_LABELS)
        self.assertEqual(job["metadata"]["labels"], _RUN_LABELS)
        self.assertEqual(
            job["spec"]["template"]["metadata"]["labels"], _WORKER_LABELS
        )

    def test_tpu_worker_env_is_left_to_gke(self):
        """CHS sets no TPU worker variables; GKE's TPU webhook injects them."""
        _, docs = self._render(
            "jobset_standard.yaml.tpl", dict(_JOBSET_VARS_INPUTS)
        )

        env_names = [
            e["name"]
            for e in docs[0]["spec"]["template"]["spec"]["containers"][0]["env"]
        ]
        self.assertEqual(env_names, ["MY_NODE_NAME"])


if __name__ == "__main__":
    unittest.main()
