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

import subprocess
import json


def get_node_metadata(cluster_name: str) -> dict:
    res = subprocess.run(
        ["kubectl", "get", "nodes", "-o", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(res.stdout)
    node_meta = {}

    for item in data.get("items", []):
        name = item["metadata"]["name"]

        labels = item["metadata"].get("labels", {})

        # Exclude control planes from the GCP mock entirely
        if "node-role.kubernetes.io/control-plane" in labels:
            continue

        machine_type = labels.get("node.kubernetes.io/instance-type", "unknown")
        instance_id = name
        node_meta[name] = {
            "machine_type": machine_type,
            "instance_id": instance_id,
        }

    return node_meta


def resolve_node_mapping() -> dict[str, list[str]]:
    """Maps configured nodepools to their sorted node hostnames."""
    res = subprocess.run(
        ["kubectl", "get", "nodes", "-o", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(res.stdout)
    pool_nodes = {}
    for item in data.get("items", []):
        name = item["metadata"]["name"]
        labels = item["metadata"].get("labels", {})
        pool = labels.get("chs-integration-nodepool")
        if pool:
            pool_nodes.setdefault(pool, []).append(name)

    for pool in pool_nodes:
        pool_nodes[pool].sort()
    return pool_nodes


def generate_pod_mock_command(
    node_b64_map: dict[str, str], default_b64: str
) -> str:
    """Generates shell command to route pod logs by node hostname."""
    script_lines = []
    for node, v in node_b64_map.items():
        script_lines.append(
            f'if [ "$MY_NODE_NAME" = "{node}" ]; then '
            f'echo {v} | base64 -d | sed "s/{{node}}/$MY_NODE_NAME/g"; '
            "exit 0; fi"
        )

    script_lines.append(
        f'echo {default_b64} | base64 -d | sed "s/{{node}}/$MY_NODE_NAME/g"'
    )
    return " && ".join(script_lines)
