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

"""Deterministic CHS Host / Subblock / Block IDs.

Each ID is a 128-bit BLAKE2s digest (32 lowercase hex chars) over the sorted
TPU chip IDs (from hardware_ids.json) of a host or of all hosts of a group.
A scope prefix keeps a one-host group from printing its host's digest. Node
names are never hashed: without chip IDs the ID is Unavailable.
"""

import hashlib

HOST_SCOPE = "chs-host-id:v1"
SUBBLOCK_SCOPE = "chs-subblock-id:v1"
BLOCK_SCOPE = "chs-block-id:v1"

UNAVAILABLE = "Unavailable"
NO_IDS = UNAVAILABLE
ID_FOOTER = (
    "CHS Host ID: BLAKE2s (32 hex chars) over the host's sorted TPU chip IDs."
    " CHS Subblock and Block IDs: BLAKE2s over the TPU chip IDs of all"
    " member hosts. These CHS-generated IDs depend only on the physical"
    " hardware, so they repeat across runs on the same hardware, and they"
    " differ from cloud reservation subblock or host IDs."
)


def sort_key(hardware_id: str) -> str:
    """Orders digests, with Unavailable IDs last."""
    return "~" if hardware_id.startswith(UNAVAILABLE) else hardware_id


def _digest(scope: str, ids) -> str:
    payload = scope + "\n" + "\n".join(sorted(ids))
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=16).hexdigest()


class AcceleratorIds:
    """TPU chip IDs per node from hardware_ids.json."""

    def __init__(self, hardware_ids: dict[str, list[str]]):
        self._ids = {
            node: {i.strip() for i in ids if i.strip()}
            for node, ids in hardware_ids.items()
        }

    @property
    def nodes(self) -> set[str]:
        """Nodes known from the accelerator inventory."""
        return set(self._ids)

    def host_id(self, node: str) -> str:
        if self._ids.get(node):
            return _digest(HOST_SCOPE, self._ids[node])
        return NO_IDS

    def group_id(self, scope: str, nodes) -> str:
        """Digest over all members' IDs; Unavailable if any member has none."""
        if any(not self._ids.get(n) for n in nodes):
            return UNAVAILABLE
        return _digest(scope, [i for n in nodes for i in self._ids[n]])
