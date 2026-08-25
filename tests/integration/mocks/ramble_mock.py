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


@contextlib.contextmanager
def mock_ramble_workloads():
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
            exp_dir = os.path.dirname(manifest_path)
            test_name = os.path.basename(exp_dir)

            mock_text = f"Simulated output for {test_name}"

            b64_payload = base64.b64encode(mock_text.encode("utf-8")).decode(
                "utf-8"
            )

            with open(manifest_path, "r") as f:
                docs = list(yaml.safe_load_all(f))

            for doc in docs:
                specs = []
                if doc and doc.get("kind") == "JobSet":
                    for rj in doc.get("spec", {}).get("replicatedJobs", []):
                        specs.append(
                            rj.get("template", {})
                            .get("spec", {})
                            .get("template", {})
                            .get("spec", {})
                        )
                elif doc and doc.get("kind") == "Job":
                    specs.append(
                        doc.get("spec", {}).get("template", {}).get("spec", {})
                    )
                elif doc and doc.get("kind") == "Pod":
                    specs.append(doc.get("spec", {}))

                for spec in specs:
                    if not spec:
                        continue
                    for list_key in ("containers", "initContainers"):
                        for c in spec.get(list_key, []):
                            c["image"] = "alpine:latest"
                            c["command"] = [
                                "/bin/sh",
                                "-c",
                                f"sleep 1 && echo {b64_payload} | base64 -d",
                            ]
                            if "resources" in c:
                                for res_type in ("limits", "requests"):
                                    if res_type in c["resources"]:
                                        c["resources"][res_type].pop(
                                            "google.com/tpu", None
                                        )
                                        c["resources"][res_type].pop(
                                            "nvidia.com/gpu", None
                                        )

            with open(manifest_path, "w") as f:
                yaml.dump_all(docs, f)

    with mock.patch(
        "chs.ramble_manager.RambleManager.setup_workspace",
        new=hooked_setup_workspace,
    ):
        yield
