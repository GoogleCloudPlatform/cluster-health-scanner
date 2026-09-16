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

from .mock_utils import get_node_metadata

import contextlib
from unittest import mock

import hashlib


@contextlib.contextmanager
def mock_gce_instances(cluster_name: str):
    mock_payload = get_node_metadata(cluster_name)

    mock_instances = []
    for node_name, info in mock_payload.items():
        inst = mock.MagicMock()
        inst.name = node_name
        inst.machine_type = (
            f"zones/mock-zone/machineTypes/{info['machine_type']}"
        )
        # generate a numeric mock ID (GCP instance IDs are 64-bit integers)
        inst.id = int(
            hashlib.md5(node_name.encode("utf-8")).hexdigest()[:15], 16
        )
        mock_instances.append(inst)

    mock_in_zone = mock.MagicMock()
    mock_in_zone.instances = mock_instances
    page_result = [("mock-zone", mock_in_zone)]

    with mock.patch(
        "google.cloud.compute_v1.InstancesClient.aggregated_list",
        return_value=page_result,
    ) as m:
        yield m


@contextlib.contextmanager
def mock_gcs_bucket():
    with (
        mock.patch("google.cloud.storage.Client"),
        mock.patch("google.cloud.storage.Bucket"),
        mock.patch("google.cloud.storage.Blob"),
    ):
        yield
