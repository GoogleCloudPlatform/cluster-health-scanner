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

import contextlib
import glob
import os
import base64
import yaml
from unittest import mock
from chs.ramble_manager import RambleManager
from tests.integration.mocks.mock_utils import generate_pod_mock_command


@contextlib.contextmanager
def mock_ramble_workloads(output_generator):
    """Intercepts Ramble YAML generation to replace heavy TPU tests
    with simple alpine mocks.
    """

    original_setup = RambleManager.setup_workspace

    def hooked_setup_workspace(self, *args, **kwargs):
        # Let Ramble generate the full workspace natively
        original_setup(self, *args, **kwargs)

        # Before CHS executes the generated manifests, overwrite them!
        manifest_files = glob.glob(
            os.path.join(self.path, "experiments", "**", "*.yaml"),
            recursive=True,
        )
        for manifest_path in manifest_files:
            # manifest_path looks like:
            # .../experiments/<test_name>/<level>/<run>/manifest.yaml
            # We want to extract <test_name> after "experiments"
            parts = manifest_path.split(os.sep)
            try:
                exp_index = parts.index("experiments")
                test_name = parts[exp_index + 1]
            except (ValueError, IndexError):
                # Fallback to basename if structure is unusual
                exp_dir = os.path.dirname(manifest_path)
                test_name = os.path.basename(exp_dir)

            output_data = output_generator.get_output_dict(test_name)

            if not output_data:
                raise ValueError(
                    "Output generator provided an empty output_map for test "
                    f"'{test_name}'."
                )

            with open(manifest_path, "r") as f:
                docs = list(yaml.safe_load_all(f))

            cleaned_docs = []
            for doc in docs:
                if not doc:
                    continue
                # Filter out CRDs not installed in Kind (e.g. ComputeDomain)
                if doc.get("kind") in ("ComputeDomain",):
                    continue
                cleaned_docs.append(doc)

                specs = []
                if doc.get("kind") == "JobSet":
                    jobset_name = doc.get("metadata", {}).get("name")
                    for rj in doc.get("spec", {}).get("replicatedJobs", []):
                        job_spec = rj.get("template", {}).get("spec", {})
                        job_spec["activeDeadlineSeconds"] = 75
                        pod_template = job_spec.get("template", {})

                        if pod_template.get("metadata") is None:
                            pod_template["metadata"] = {}
                        if "labels" not in pod_template["metadata"]:
                            pod_template["metadata"]["labels"] = {}

                        pod_template["metadata"]["labels"][
                            "jobset.sigs.k8s.io/jobset-name"
                        ] = jobset_name

                        specs.append(pod_template.get("spec", {}))
                elif doc.get("kind") == "Job":
                    job_spec = doc.get("spec", {})
                    job_spec["activeDeadlineSeconds"] = 75
                    specs.append(job_spec.get("template", {}).get("spec", {}))
                elif doc.get("kind") == "Pod":
                    specs.append(doc.get("spec", {}))

                for spec in specs:
                    if not spec:
                        continue

                    # Strip DRA / hardware-specific resource claims
                    # and init containers
                    spec.pop("resourceClaims", None)
                    spec.pop("initContainers", None)

                    node_map = output_data.get("nodes", {})

                    b64_map = {
                        k: base64.b64encode(v.encode("utf-8")).decode("utf-8")
                        for k, v in node_map.items()
                    }
                    default_b64 = base64.b64encode(
                        output_data["default"].encode("utf-8")
                    ).decode("utf-8")

                    final_cmd = generate_pod_mock_command(b64_map, default_b64)

                    for c in spec.get("containers", []):
                        c["image"] = "alpine:latest"
                        c["imagePullPolicy"] = "IfNotPresent"
                        c["command"] = [
                            "/bin/sh",
                            "-c",
                            'MY_NODE_NAME="${MY_NODE_NAME:-'
                            '$(hostname)}"; '
                            f"sleep 1; {final_cmd}",
                        ]
                        if "resources" in c:
                            c["resources"] = {}

            with open(manifest_path, "w") as f:
                yaml.dump_all(cleaned_docs, f)

        # Strip the topologyName constraint so mock nodes don't deadlock
        # Kueue TAS
        kueue_yaml = os.path.join(self.path, "configs", "kueue-configs.yaml")
        if os.path.exists(kueue_yaml):
            with open(kueue_yaml, "r") as f:
                content = f.read()
            content = content.replace(
                "topologyName: chs-node-topology",
                "# topologyName: chs-node-topology",
            )
            with open(kueue_yaml, "w") as f:
                f.write(content)

    with mock.patch(
        "chs.ramble_manager.RambleManager.setup_workspace",
        new=hooked_setup_workspace,
    ):
        yield
