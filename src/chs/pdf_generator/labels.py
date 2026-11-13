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

"""Node naming helpers for charts, pair cells, and bounded error/issue text."""

TestKey = tuple[str, str]  # (canonical test name, test level)


def shorten_nodes(nodes: list[str], max_len: int = 16) -> dict[str, str]:
    """Maps node names to prefix...suffix labels, or full names on collision."""
    unique = list(dict.fromkeys(nodes))
    short = {
        n: n if len(n) <= max_len else f"{n[:4]}...{n[-6:]}" for n in unique
    }
    if len(set(short.values())) < len(unique):
        return {n: n for n in unique}
    return short


def bounded_hosts(
    ordered_nodes: list[str], display_fn, *, limit: int = 2
) -> str:
    """Formats ordered nodes concisely: 'n1', 'n1, n2', or 'n1, n2 (+K)'."""
    if not ordered_nodes:
        return ""
    if len(ordered_nodes) == 1:
        return ordered_nodes[0]
    if len(ordered_nodes) <= limit:
        return ", ".join(display_fn(n) for n in ordered_nodes)
    head = ", ".join(display_fn(n) for n in ordered_nodes[:limit])
    return f"{head} (+{len(ordered_nodes) - limit} more)"


class Labels:
    """Shortened unique node names and compact host/pair text for a section."""

    def __init__(
        self,
        nodes: list[str],
        pairwise: set[TestKey],
    ):
        self.index = {node: i for i, node in enumerate(nodes, start=1)}
        self._pairwise = pairwise
        self._short = shorten_nodes(nodes)

    def __getitem__(self, node: str) -> str:
        return self._short.get(node, node)

    def ordered(self, nodes) -> tuple[str, ...]:
        return tuple(sorted(nodes, key=lambda n: self.index.get(n, 0)))

    def order(self, nodes) -> list[int]:
        return sorted(self.index.get(n, 0) for n in nodes)

    def hosts_text(self, nodes, limit: int = 2) -> str:
        """Full node name when single, else bounded shortened list."""
        ordered = list(self.ordered(set(nodes)))
        return bounded_hosts(ordered, self.__getitem__, limit=limit)

    def _is_pair(self, key: TestKey, targets) -> bool:
        return len(targets) == 2 and key in self._pairwise

    def entity(self, key: TestKey, targets) -> str:
        """Chart label of one result: short node, 'a↔b', 'All 16 hosts'."""
        targets = list(self.ordered(targets))
        if len(targets) == 1:
            return self[targets[0]]
        if self._is_pair(key, targets):
            return "↔".join(self[n] for n in targets)
        if set(targets) == self.index.keys():
            return f"All {len(targets)} hosts"
        return bounded_hosts(targets, self.__getitem__, limit=2)

    def cell_text(self, text: str, key: TestKey, targets, viewer: str) -> str:
        """How viewer's cell shows a result: '↔partner 240.20', '396.90'."""
        if len(targets) == 1:
            return text
        if self._is_pair(key, targets):
            partner = next((n for n in targets if n != viewer), viewer)
            return f"↔{self[partner]} {text}"
        if set(targets) == self.index.keys():
            return text
        ordered = list(self.ordered(targets))
        return f"{text} ({bounded_hosts(ordered, self.__getitem__, limit=2)})"
