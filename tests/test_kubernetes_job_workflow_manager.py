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

"""Unit tests for the kubernetes-job workflow manager."""

import os
import re
import subprocess
import sys
import tempfile
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

from chs.repos.workflow_managers.kubernetes_job.workflow_manager import (  # noqa: E402
    KubernetesJob,
)

_WORKER_SERVICE_NAME = "chs-run1"
_RUN_LABELS = {
    "app.kubernetes.io/managed-by": "chs",
    "app.kubernetes.io/instance": "run1",
}
_WORKER_LABELS = {**_RUN_LABELS, "app.kubernetes.io/component": "worker"}
_JOB_LABELS = {**_RUN_LABELS, "team": "infra"}
_POD_LABELS = {**_WORKER_LABELS, "team": "infra"}


_JOB_VARS_INPUTS = {
    "job_name": "chs-run1-nccl-tests-all-gather",
    "timeout": "3600",
    "job_template": "job_mpi.yaml.tpl",
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
    "suspend_job": "true",
    "chs_run_id": "run1",
    "job_labels_yaml": "app.kubernetes.io/managed-by: chs\nteam: infra",
    # CHS labels win over user labels with the same key.
    "pod_labels_yaml": "app.kubernetes.io/instance: other\nteam: infra",
}

_TEMPLATE_PLACEHOLDER = re.compile(r"(?<![$\w])\{(\w+)\}")


def _workflow_manager(var_map: dict[str, str]) -> KubernetesJob:
    wm = KubernetesJob(mock.Mock())
    wm.app_inst = mock.Mock()
    wm.app_inst.expander.expand_var_name.side_effect = lambda name: (
        var_map.get(name, "")
    )
    return wm


class KubernetesJobTest(unittest.TestCase):
    """Tests for the kubernetes-job workflow manager."""

    def setUp(self):
        super().setUp()
        self.generate_ssh_keys = self.enterContext(
            mock.patch.object(
                KubernetesJob,
                "_generate_ssh_keys",
                autospec=True,
                return_value=("private-key", "public-key"),
            )
        )

    def test_kubernetes_job_extra_annotations_and_resources(self):
        wm = KubernetesJob(mock.Mock())
        wm.app_inst = mock.Mock()

        var_map = {
            "job_name": "test-job",
            "command": "echo test",
            "experiment_run_dir": "/tmp/exp",
            "app_workdir": "",
            "container_image": "image:latest",
            "job_labels_yaml": "",
            "job_annotations_yaml": "",
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
            "suspend_job": "false",
            "timeout": "60",
        }
        wm.app_inst.expander.expand_var_name.side_effect = (
            lambda k: var_map.get(k, "")
        )
        res = wm._job_vars()

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
        res_empty = wm._compute_job_vars()
        self.assertIn(
            "networking.gke.io/default-interface: eth0",
            res_empty["pod_annotations_section"],
        )
        self.assertNotIn(
            "compute-domain.nvidia.com", res_empty["pod_annotations_section"]
        )
        self.assertNotIn("claims:", res_empty["container_resources_section"])

    def test_job_vars_are_computed_once_per_experiment(self):
        """All templates of one experiment share one Job name and key."""
        wm = _workflow_manager(dict(_JOB_VARS_INPUTS))

        rendered_vars = [wm._job_vars() for _ in range(5)]

        self.generate_ssh_keys.assert_called_once()
        self.assertEqual(
            {v["sanitized_job_name"] for v in rendered_vars},
            {rendered_vars[0]["sanitized_job_name"]},
        )
        self.assertTrue(all(v == rendered_vars[0] for v in rendered_vars))

    def test_job_vars_differ_between_experiments(self):
        """Different experiments never share a Job name or key pair."""
        var_map = dict(_JOB_VARS_INPUTS)
        wm = _workflow_manager(var_map)
        self.generate_ssh_keys.side_effect = [
            ("private-key-1", "public-key-1"),
            ("private-key-2", "public-key-2"),
        ]

        first = wm._job_vars()
        var_map["experiment_run_dir"] = (
            "/workspace/experiments/nccl/all-gather/exp2"
        )
        second = wm._job_vars()

        self.assertEqual(self.generate_ssh_keys.call_count, 2)
        self.assertNotEqual(
            first["sanitized_job_name"], second["sanitized_job_name"]
        )
        self.assertNotEqual(
            first["ssh_private_key_indented"],
            second["ssh_private_key_indented"],
        )

    def test_single_node_experiments_of_a_test_get_distinct_names(self):
        """Names differ per experiment and are stable for the same one.

        All experiments of a single-node test share job_name; only their run
        directory, which holds the node name, tells them apart.
        """
        var_map = dict(_JOB_VARS_INPUTS)
        names = []
        for node in ("node-0", "node-1", "node-2", "node-0"):
            var_map["experiment_run_dir"] = (
                f"/workspace/experiments/nccl/all-gather/exp-{node}"
            )
            names.append(
                _workflow_manager(var_map)._job_vars()["sanitized_job_name"]
            )

        self.assertEqual(len(set(names[:3])), 3)
        self.assertEqual(names[0], names[3])
        self.assertTrue(names[0].startswith("chs-run1-nccl-tests-all-gather-"))

    def test_job_names_fit_kubernetes_limits(self):
        """Job names and the names derived from them stay valid at 63."""
        var_map = {
            **_JOB_VARS_INPUTS,
            "job_name": "chs-" + "a" * 37 + "-_-" + "b" * 40,
        }
        name = _workflow_manager(var_map)._job_vars()["sanitized_job_name"]

        self.assertEqual(len(name), 51)
        for derived in (f"{name}-ssh", f"{name}-cdc", f"{name}-999"):
            self.assertLessEqual(len(derived), 63)
            self.assertRegex(derived, r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")

    def test_returned_job_vars_do_not_alias_the_cache(self):
        """Mutating returned variables does not leak into later templates."""
        wm = _workflow_manager(dict(_JOB_VARS_INPUTS))
        job_name = wm._job_vars()["sanitized_job_name"]

        wm._job_vars()["sanitized_job_name"] = "mutated"

        self.assertEqual(wm._job_vars()["sanitized_job_name"], job_name)

    def test_only_the_mpi_template_generates_ssh_keys(self):
        """Keys are generated for the MPI template's Secret, not otherwise."""
        standard_vars, standard_docs = self._render(
            "job_standard.yaml.tpl", dict(_JOB_VARS_INPUTS)
        )
        self.generate_ssh_keys.assert_not_called()
        self.assertEqual(standard_vars["ssh_private_key_indented"], "")
        self.assertEqual(standard_vars["ssh_public_key_indented"], "")
        self.assertEqual([d["kind"] for d in standard_docs], ["Job"])

        mpi_vars, _ = self._render("job_mpi.yaml.tpl", dict(_JOB_VARS_INPUTS))
        self.generate_ssh_keys.assert_called_once()
        self.assertEqual(
            mpi_vars["ssh_private_key_indented"], "    private-key"
        )

    def _render(self, template_name, var_map, **extra_vars):
        """Renders a kubernetes-job template like Ramble does."""
        wm = _workflow_manager({**var_map, "job_template": template_name})
        template_vars = {
            "kubernetes_namespace": "chs",
            "n_nodes": var_map.get("n_nodes", "2"),
            "timeout": var_map["timeout"],
            "ssh_port": "222",
            "processes_per_node": "8",
            "worker_service_name": _WORKER_SERVICE_NAME,
            **extra_vars,
            **wm._job_vars(),
        }
        template_path = os.path.join(
            os.path.dirname(sys.modules[KubernetesJob.__module__].__file__),
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
                "activeDeadlineSeconds": 4260,
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
            **_JOB_VARS_INPUTS,
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

        template_vars, docs = self._render("job_standard.yaml.tpl", var_map)

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
        script = container["command"][2]
        self.assertIn(
            "<<'CHS_TEST_COMMAND_EOF'\necho one\necho two\n"
            "CHS_TEST_COMMAND_EOF\n",
            script,
        )
        self.assertIn(
            "timeout --kill-after=10 3600 bash /tmp/chs_test_command.sh\n",
            script,
        )
        self.assertTrue(script.endswith("exit $CHS_EXIT_CODE\n"))

    def test_job_deadline_adds_startup_budget_and_margin(self):
        """Tests get the startup budget and a margin; no timeout is kept."""
        var_map = {**_JOB_VARS_INPUTS, "timeout": "240"}
        self.assertEqual(
            _workflow_manager(var_map)._job_vars()["job_deadline"], "900"
        )
        var_map["timeout"] = "{timeout}"
        self.assertEqual(
            _workflow_manager(var_map)._job_vars()["job_deadline"], "{timeout}"
        )

    def test_template_exit_codes(self):
        """Only a test that outlives its own timeout exits with 124.

        The MPI launcher runs as a single node. The scripts run on the host,
        so the fixed paths go to a temporary directory, sshd and ldconfig are
        skipped, ssh is a stub, and --kill-after is shortened to keep the test
        fast.
        """
        cases = {
            "exit 0": 0,
            "exit 3": 3,
            "sleep 30": 124,
            "trap '' TERM; while true; do sleep 1; done": 124,
            "kill -9 $$": 137,
            "exit 124": 1,
        }
        for template in ("job_standard.yaml.tpl", "job_mpi.yaml.tpl"):
            for command, expected in cases.items():
                with (
                    self.subTest(template=template, command=command),
                    tempfile.TemporaryDirectory() as d,
                ):
                    var_map = {
                        **_JOB_VARS_INPUTS,
                        "command": command,
                        "timeout": "2",
                        "n_nodes": "1",
                        "experiment_run_dir": d,
                    }
                    _, docs = self._render(template, var_map)
                    pod_spec = docs[-1]["spec"]["template"]["spec"]
                    (container,) = pod_spec["containers"]
                    shell, flag, script = container["command"]
                    for fixed, local in (
                        ("/tmp/chs_test_command.sh", f"{d}/test.sh"),
                        ("/tmp/hostfile", f"{d}/hostfile"),
                        ("/sbin/sshd", "true"),
                        ("/sbin/ldconfig", "true"),
                        ("--kill-after=10", "--kill-after=1"),
                    ):
                        script = script.replace(fixed, local)
                    os.mkdir(f"{d}/bin")
                    with open(f"{d}/bin/ssh", "w") as ssh:
                        ssh.write("#!/bin/sh\nexit 0\n")
                    os.chmod(f"{d}/bin/ssh", 0o755)
                    env = {
                        **os.environ,
                        "HOME": d,
                        "PATH": f"{d}/bin:{os.environ['PATH']}",
                        "JOB_COMPLETION_INDEX": "0",
                    }
                    result = subprocess.run(
                        [shell, flag, script],
                        capture_output=True,
                        env=env,
                        timeout=60,
                    )
                    self.assertEqual(result.returncode, expected, result.stderr)

    def test_mpi_template_renders_secret_and_one_indexed_job(self):
        """Golden structure of the MPI template."""
        var_map = {
            **_JOB_VARS_INPUTS,
            "command": "mpirun -n 16 \\\n  all_gather_perf",
            "n_nodes": "3",
            "extra_volumes_yaml": "- name: shm\n  emptyDir: {}",
            "extra_volume_mounts_yaml": "- name: shm\n  mountPath: /dev/shm",
        }

        template_vars, docs = self._render("job_mpi.yaml.tpl", var_map)

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
        self.assertIn(
            "<<'CHS_TEST_COMMAND_EOF'\nmpirun -n 16 \\\n  all_gather_perf\n"
            "CHS_TEST_COMMAND_EOF\n",
            script,
        )
        launch = script[script.index("STARTING MPI") :]
        self.assertIn(
            "timeout --kill-after=10 3600 bash /tmp/chs_test_command.sh\n",
            launch,
        )
        self.assertIn("DNS_DEADLINE=$(( $(date +%s) + 4260 ))", script)

    def test_custom_resources_get_run_labels(self):
        """Custom resources keep their labels and get the run's labels."""
        raw_job_name = _JOB_VARS_INPUTS["job_name"]
        var_map = {
            **_JOB_VARS_INPUTS,
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

        template_vars, docs = self._render("job_mpi.yaml.tpl", var_map)

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
            **_JOB_VARS_INPUTS,
            "job_labels_yaml": "",
            "pod_labels_yaml": "",
        }

        _, docs = self._render("job_mpi.yaml.tpl", var_map)

        secret, job = docs
        self.assertEqual(secret["metadata"]["labels"], _RUN_LABELS)
        self.assertEqual(job["metadata"]["labels"], _RUN_LABELS)
        self.assertEqual(
            job["spec"]["template"]["metadata"]["labels"], _WORKER_LABELS
        )

    def test_tpu_worker_env_is_left_to_gke(self):
        """CHS sets no TPU worker variables; GKE's TPU webhook injects them."""
        _, docs = self._render("job_standard.yaml.tpl", dict(_JOB_VARS_INPUTS))

        env_names = [
            e["name"]
            for e in docs[0]["spec"]["template"]["spec"]["containers"][0]["env"]
        ]
        self.assertEqual(env_names, ["MY_NODE_NAME"])

    def test_kubernetes_job_use_gpu_dra_standard(self):
        """Standard template renders claims when use_gpu_dra is true."""
        var_map = {
            **_JOB_VARS_INPUTS,
            "use_gpu_dra": "true",
            "container_resources_yaml": (
                "limits:\n  nvidia.com/gpu: 8\nrequests:\n  nvidia.com/gpu: 8"
            ),
        }

        template_vars, docs = self._render("job_standard.yaml.tpl", var_map)
        job_name = template_vars["sanitized_job_name"]

        self.assertEqual(
            [d["kind"] for d in docs], ["ResourceClaimTemplate", "Job"]
        )

        rct = docs[0]
        self.assertEqual(rct["apiVersion"], "resource.k8s.io/v1")
        self.assertEqual(rct["metadata"]["name"], f"{job_name}-gpu-template")
        self.assertEqual(rct["metadata"]["namespace"], "chs")
        self.assertEqual(rct["metadata"]["labels"], _RUN_LABELS)
        req = rct["spec"]["spec"]["devices"]["requests"][0]
        self.assertEqual(req["exactly"]["deviceClassName"], "gpu.nvidia.com")
        self.assertEqual(req["exactly"]["count"], 8)

        job = docs[1]
        pod_spec = job["spec"]["template"]["spec"]
        self.assertEqual(
            pod_spec["resourceClaims"],
            [
                {
                    "name": "gpu-claim",
                    "resourceClaimTemplateName": f"{job_name}-gpu-template",
                }
            ],
        )
        container = pod_spec["containers"][0]
        self.assertEqual(
            container["resources"],
            {"claims": [{"name": "gpu-claim"}]},
        )

    def test_kubernetes_job_use_gpu_dra_with_compute_domain(self):
        """DRA merges gpu-claim with existing compute-domain-channel claims."""
        raw_job_name = _JOB_VARS_INPUTS["job_name"]
        var_map = {
            **_JOB_VARS_INPUTS,
            "use_gpu_dra": "true",
            "container_resources_yaml": (
                "limits:\n  nvidia.com/gpu: 8\nrequests:\n  nvidia.com/gpu: 8"
            ),
            "extra_container_resources_yaml": (
                "claims:\n- name: compute-domain-channel"
            ),
            "pod_resource_claims_yaml": (
                "- name: compute-domain-channel\n"
                "  resourceClaimTemplateName: compute-domain-channel-template"
            ),
            "custom_resources_yaml": (
                "apiVersion: resource.nvidia.com/v1beta1\n"
                "kind: ComputeDomain\n"
                "metadata:\n"
                f"  name: {raw_job_name}-cd\n"
                "  namespace: chs\n"
                "spec:\n"
                "  numNodes: 2\n"
            ),
        }

        template_vars, docs = self._render("job_mpi.yaml.tpl", var_map)
        job_name = template_vars["sanitized_job_name"]

        self.assertEqual(
            [d["kind"] for d in docs],
            ["ResourceClaimTemplate", "ComputeDomain", "Secret", "Job"],
        )

        rct = docs[0]
        self.assertEqual(rct["apiVersion"], "resource.k8s.io/v1")
        self.assertEqual(rct["metadata"]["name"], f"{job_name}-gpu-template")
        self.assertEqual(
            rct["spec"]["spec"]["devices"]["requests"][0]["exactly"]["count"], 8
        )

        job = docs[3]
        pod_spec = job["spec"]["template"]["spec"]
        self.assertEqual(
            pod_spec["resourceClaims"],
            [
                {
                    "name": "compute-domain-channel",
                    "resourceClaimTemplateName": (
                        "compute-domain-channel-template"
                    ),
                },
                {
                    "name": "gpu-claim",
                    "resourceClaimTemplateName": f"{job_name}-gpu-template",
                },
            ],
        )
        container = pod_spec["containers"][0]
        self.assertEqual(
            container["resources"],
            {
                "claims": [
                    {"name": "compute-domain-channel"},
                    {"name": "gpu-claim"},
                ]
            },
        )

    def test_kubernetes_job_use_gpu_dra_non_gpu_job_renders_standard(self):
        """DRA only converts nvidia.com/gpu requests, like the device plugin.

        A Job that does not request nvidia.com/gpu, e.g. the root
        cluster-health experiment, gets no GPUs, even if gpus_per_node is set.
        """
        for gpus_per_node in ("", "8"):
            with self.subTest(gpus_per_node=gpus_per_node):
                var_map = {
                    **_JOB_VARS_INPUTS,
                    "use_gpu_dra": "true",
                    "container_resources_yaml": "",
                    "gpus_per_node": gpus_per_node,
                }

                _, docs = self._render("job_standard.yaml.tpl", var_map)

                self.assertEqual([d["kind"] for d in docs], ["Job"])
                pod_spec = docs[0]["spec"]["template"]["spec"]
                self.assertNotIn("resourceClaims", pod_spec)
                self.assertNotIn("resources", pod_spec["containers"][0])

    def test_merge_container_resources_deduplicates_claims(self):
        """Claims named twice across the resources maps are kept once."""
        self.assertEqual(
            KubernetesJob._merge_container_resources(
                {
                    "limits": {"nvidia.com/gpu": 4},
                    "claims": [{"name": "compute-domain-channel"}],
                },
                {
                    "limits": {"memory": "1Gi"},
                    "claims": [
                        {"name": "compute-domain-channel"},
                        {"name": "other"},
                    ],
                },
            ),
            {
                "limits": {"nvidia.com/gpu": 4, "memory": "1Gi"},
                "claims": [
                    {"name": "compute-domain-channel"},
                    {"name": "other"},
                ],
            },
        )

    def test_kubernetes_job_use_gpu_dra_claims_in_base_resources(self):
        """A4X-max keeps its IMEX claim from container_resources_yaml."""
        var_map = {
            **_JOB_VARS_INPUTS,
            "use_gpu_dra": "true",
            "container_resources_yaml": (
                "limits:\n  nvidia.com/gpu: 4\n  memory: 1Gi\n"
                "requests:\n  nvidia.com/gpu: 4\n"
                "claims:\n- name: compute-domain-channel"
            ),
        }

        _, docs = self._render("job_standard.yaml.tpl", var_map)

        container = docs[-1]["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(
            container["resources"],
            {
                "limits": {"memory": "1Gi"},
                "claims": [
                    {"name": "compute-domain-channel"},
                    {"name": "gpu-claim"},
                ],
            },
        )
        self.assertEqual(
            docs[0]["spec"]["spec"]["devices"]["requests"][0]["exactly"][
                "count"
            ],
            4,
        )

    def test_kubernetes_job_use_gpu_dra_invalid_gpu_count(self):
        """DRA fails if nvidia.com/gpu requested has non-positive count."""
        var_map = {
            **_JOB_VARS_INPUTS,
            "use_gpu_dra": "true",
            "container_resources_yaml": "limits:\n  nvidia.com/gpu: 0",
        }

        with self.assertRaisesRegex(ValueError, "GPU count must be positive"):
            _workflow_manager(var_map)._job_vars()


if __name__ == "__main__":
    unittest.main()
