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


"""Plain data the PDF renderer draws; produced by builder.build_report()."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Point:
    """One bar in a chart and one line in a host-table cell.

    label: x-axis label (node short name, 'a↔b', or 'All 16 hosts').
    value: judged numeric value, or None when there is no number.
    text:  what the cell shows ('1033', '396.90', '↔node 240.20',
           'TIMEOUT', 'INCOMPLETE', '8').
    degraded: True when this value/result did not meet its threshold or
              the result is degraded (drawn red).
    """

    label: str
    value: float | None
    text: str
    degraded: bool


@dataclasses.dataclass(frozen=True)
class Row:
    """A host row in a table."""

    label: str  # Shortened unique node name for charts/pairs
    name: str  # Full node name
    hardware_id: str  # 32-hex BLAKE2s digest or "Unavailable"
    # "HEALTHY" | "DEGRADED" | "POTENTIALLY DEGRADED" | "INCOMPLETE"
    status: str
    cells: list[list[Point]]  # one list per Section.columns entry
    pool: str = ""  # Nodepool (GKE) or Partition (Slurm)
    instance_id: str = ""  # GCE Instance ID when populated


@dataclasses.dataclass(frozen=True)
class Chart:
    test: str  # test display name, matching TestRow.name
    title: str  # full test name (+ level, + metric when >1 metric)
    unit: str
    threshold: float | None
    points: list[Point]


@dataclasses.dataclass(frozen=True)
class Event:
    """A potential error/warning from the tests' error detectors."""

    hosts: str  # Real node name, or bounded shortened list (+N more)
    section: str
    test: str
    type: str  # "Error" | "Warning"
    category: str
    message: str
    count: int  # raw events merged into this row


@dataclasses.dataclass(frozen=True)
class TestRow:
    """A row of the page-1 methodology table."""

    name: str
    scope: str  # "Single-Host" | "Host Pair" | "Slice-Wide" | "Multi-Host"
    description: str
    thresholds: list[str]


@dataclasses.dataclass(frozen=True)
class Section:
    """A CHS Subblock / Block / Slice / Hosts section."""

    title: str  # "CHS Subblock 1 of 2", "CHS Block", "Hosts"
    group_id_label: str | None  # "CHS Subblock ID" / "CHS Block ID" / None
    group_id: str | None  # digest / "Unavailable" / None
    topology: str | None
    columns: list[str]  # host-table test columns (full test names)
    hosts: list[Row]
    charts: list[Chart]
    events: list[Event]
    pool_label: str = "Nodepool"  # "Nodepool" | "Partition" | ""
    host_label: str = "Node"  # "GKE Node" | "Slurm Host" | "VM Host" | "Node"
    pool_summary: str = ""  # comma-joined unique pools/partitions in section


@dataclasses.dataclass(frozen=True)
class Report:
    title: str  # "<machine_type> Bill of Health"
    run_info: list[tuple[str, str]]  # key/value strip
    tests: list[TestRow]
    sections: list[Section]
    id_footer: str  # one-line ID derivation text for the page footer
