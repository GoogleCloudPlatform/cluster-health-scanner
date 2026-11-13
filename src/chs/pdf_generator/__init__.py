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

"""PDF Bill of Health.

Renders the same BillOfHealth proto as the HTML report into a TPU PDF: an
overview page, one section per TPU subblock, block, or slice the run tested as
a whole, and one shared section for all other hosts, anchored on
hardware-derived CHS Host / Subblock / Block IDs.
"""

import os

from chs.pdf_generator import report
from chs.pdf_generator.builder import build_report, populate_system_entities
from chs.protos import bill_of_health_pb2 as pb
from chs.schedulers.base import (
    HARDWARE_IDS_FILENAME,
    HOST_METADATA_FILENAME,
    TPU_SLICES_FILENAME,
)
from chs.utils.file_utils import load_json

__all__ = ["PdfReportGenerator"]


class PdfReportGenerator:
    """Generates a PDF Bill of Health from a BillOfHealth proto.

    Args:
      hardware_ids: node -> TPU chip IDs (hardware_ids.json).
      tpu_slices: multi-host TPU slices the run tested as a whole
        (tpu_slices.json), each {"topology": str, "nodes": [str, ...]}.
      host_metadata: node -> {"nodepool"|"partition", "instance_id"} from
        host_metadata.json.
    """

    def __init__(
        self,
        hardware_ids: dict[str, list[str]] | None = None,
        tpu_slices: list[dict] | None = None,
        host_metadata: dict[str, dict[str, str]] | None = None,
    ):
        self._hardware_ids = hardware_ids or {}
        self._tpu_slices = tpu_slices or []
        self._host_metadata = host_metadata or {}

    @classmethod
    def from_workspace(cls, workspace_dir: str) -> "PdfReportGenerator":
        """Reads the hardware inventory files the scheduler wrote."""
        return cls(
            load_json(
                os.path.join(workspace_dir, HARDWARE_IDS_FILENAME),
                dict,
                is_valid=_is_hardware_ids_entry,
                warn=True,
            ),
            load_json(
                os.path.join(workspace_dir, TPU_SLICES_FILENAME),
                list,
                is_valid=_is_tpu_slice,
                warn=True,
            ),
            load_json(
                os.path.join(workspace_dir, HOST_METADATA_FILENAME),
                dict,
                is_valid=_is_host_metadata_entry,
                warn=True,
            ),
        )

    def populate_entities(
        self, boh: pb.BillOfHealth, machine_type: str = ""
    ) -> None:
        """Populates CHS Host / Subblock / Block IDs onto boh.system_info."""
        populate_system_entities(
            boh,
            self._hardware_ids,
            self._tpu_slices,
            self._host_metadata,
            machine_type=machine_type,
        )

    def save_pdf(self, boh: pb.BillOfHealth, path: str) -> str:
        """Builds the report and atomically writes it to path.

        Raises whatever Bill of Health aggregation or rendering raises.
        """
        report.render(
            build_report(
                boh,
                self._hardware_ids,
                self._tpu_slices,
                self._host_metadata,
            ),
            path,
        )
        return path


def _is_hardware_ids_entry(item) -> bool:
    _, ids = item
    return isinstance(ids, list) and all(isinstance(i, str) for i in ids)


def _is_host_metadata_entry(item) -> bool:
    node, entry = item
    return (
        isinstance(node, str)
        and isinstance(entry, dict)
        and all(
            isinstance(k, str) and isinstance(v, str) for k, v in entry.items()
        )
    )


def _is_tpu_slice(entry) -> bool:
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("topology"), str)
        and isinstance(entry.get("nodes"), list)
        and all(isinstance(n, str) for n in entry["nodes"])
    )
