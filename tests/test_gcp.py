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

import unittest
from unittest import mock
from chs.utils.gcp import query_gce_instances_info
from chs.exceptions import ChsSchedulerException
from google.api_core.exceptions import GoogleAPICallError


class TestGcpUtils(unittest.TestCase):
    @mock.patch("chs.utils.gcp.compute_v1.InstancesClient")
    def test_query_gce_instances_info_success(self, mock_client_class):
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client

        mock_instance1 = mock.MagicMock()
        mock_instance1.name = "node1"
        mock_instance1.machine_type = (
            "zones/us-central1-a/machineTypes/a3-megagpu-8g"
        )
        mock_instance1.id = 12345

        mock_instances_in_zone = mock.MagicMock()
        mock_instances_in_zone.instances = [mock_instance1]

        # page_result is an iterable of (zone, instances_in_zone)
        mock_client.aggregated_list.return_value = [
            ("zones/us-central1-a", mock_instances_in_zone)
        ]

        result = query_gce_instances_info("my-project", ["node1"])

        self.assertIn("node1", result)
        self.assertEqual(result["node1"]["machine_type"], "a3-megagpu-8g")
        self.assertEqual(result["node1"]["instance_id"], "12345")

        # Verify batching (filter string)
        mock_client.aggregated_list.assert_called_once()
        request = mock_client.aggregated_list.call_args.kwargs["request"]
        self.assertEqual(request.project, "my-project")
        self.assertIn('name = "node1"', request.filter)

    @mock.patch("chs.utils.gcp.compute_v1.InstancesClient")
    def test_query_gce_instances_info_api_error(self, mock_client_class):
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.aggregated_list.side_effect = GoogleAPICallError(
            "API failed"
        )

        with self.assertRaisesRegex(
            ChsSchedulerException, "Failed to query GCE API"
        ):
            query_gce_instances_info("my-project", ["node1"])


if __name__ == "__main__":
    unittest.main()
