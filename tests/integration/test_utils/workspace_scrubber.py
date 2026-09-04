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

from __future__ import annotations
import os
import pathlib
import re
import shutil

ALLOW_LIST = {
    "configs/ramble.yaml",
}


EXPERIMENT_ARTIFACT_SUFFIXES = (
    "jobset.yaml",
    "sbatch",
    ".sh",
    "execute_experiment",
)


SCRUBBING_RULES = [
    (re.compile(r"#SBATCH --deadline=.*"), "#SBATCH --deadline=<DEADLINE>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2}\.\d+"), "<TIMESTAMP>"),
    (
        re.compile(r"ssh-rsa [A-Za-z0-9+/=]+(?:\s+\S+)?"),
        "ssh-rsa <SCRUBBED_KEY>",
    ),
    (
        re.compile(r"ssh-ed25519 [A-Za-z0-9+/=]+(?:\s+\S+)?"),
        "ssh-ed25519 <SCRUBBED_KEY>",
    ),
    (
        re.compile(
            r"-----BEGIN.*?PRIVATE KEY-----.*?-----END.*?PRIVATE KEY-----",
            flags=re.DOTALL,
        ),
        "<SCRUBBED_PRIVATE_KEY>",
    ),
    (re.compile(r"-[0-9a-f]{6}(?=[^a-zA-Z0-9]|$)"), "-<HASH>"),
    (re.compile(r"gke-[a-zA-Z0-9-]+-[0-9a-f]{8}-[a-z0-9]{4}"), "<GKE_NODE>"),
    (
        re.compile(r"cloud\.google\.com/gke-nodepool:\s*\S+"),
        "cloud.google.com/gke-nodepool: <GKE_NODEPOOL>",
    ),
]


def should_keep_file(rel_path: str) -> bool:
    if rel_path in ALLOW_LIST:
        return True

    # Capture execution artifacts across all generated experiments
    if rel_path.startswith("experiments/"):
        return rel_path.endswith(EXPERIMENT_ARTIFACT_SUFFIXES)
    return False


class WorkspaceScrubber:
    def __init__(
        self,
        workspace_root: str,
        additional_replacements: list[tuple[str, str]] | None = None,
    ):
        self.workspace_root = workspace_root
        self.home_dir = os.path.expanduser("~")
        self.workspace_name = os.path.basename(workspace_root)
        self.additional_replacements = additional_replacements or []

    def scrub_text(self, content: str) -> str:
        content = content.replace(self.workspace_root, "<WORKSPACE_ROOT>")
        content = content.replace(self.home_dir, "<HOME_DIR>")

        for raw, placeholder in self.additional_replacements:
            content = content.replace(raw, placeholder)

        # Apply default scrubbing rules
        for pattern, replacement in SCRUBBING_RULES:
            content = pattern.sub(replacement, content)

        return content

    def extract_and_scrub(self, snapshot_dir: str):
        """Extracts allow-listed files from the workspace, scrubs them,
        and saves to snapshot_dir.
        """
        snapshot_path = pathlib.Path(snapshot_dir)
        workspace_path = pathlib.Path(self.workspace_root)

        # Ensure snapshot dir exists and is empty
        if snapshot_path.exists():
            shutil.rmtree(snapshot_path)
        snapshot_path.mkdir(parents=True, exist_ok=True)

        for file_path in workspace_path.rglob("*"):
            if not file_path.is_file():
                continue

            rel_path = str(file_path.relative_to(workspace_path))

            if should_keep_file(rel_path):
                # It's a text file we want to keep
                try:
                    content = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue

                scrubbed_content = self.scrub_text(content)

                # Scrub relative path to remove node names from directories
                scrubbed_rel_path = self.scrub_text(rel_path)

                target_path = snapshot_path / scrubbed_rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)

                target_path.write_text(scrubbed_content, encoding="utf-8")
