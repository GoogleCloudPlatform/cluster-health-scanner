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

"""Tests for cluster name utilities."""

import unittest
from unittest import mock

from chs.exceptions import ClusterDiscoveryException
from chs.utils.cluster import get_sanitized_cluster_name, sanitize_dns_1123


class TestCluster(unittest.TestCase):
    """Tests cluster name sanitization and retrieval."""

    def test_sanitize_dns_1123(self):
        # Test simple lowercasing and stripping
        self.assertEqual(sanitize_dns_1123("Simple-Name"), "simple-name")

        # Test spaces and underscores to dashes
        self.assertEqual(
            sanitize_dns_1123("my_cluster name"), "my-cluster-name"
        )

        # Test non-alphanumeric stripping
        self.assertEqual(sanitize_dns_1123("cluster@name!"), "clustername")
        self.assertEqual(sanitize_dns_1123("cluster_name_$$$"), "cluster-name")

        # Test truncation
        self.assertEqual(
            sanitize_dns_1123(
                "very-long-cluster-name-that-exceeds-limit", max_length=15
            ),
            "very-long-clust",
        )

        # Test trailing dashes strip after truncation
        self.assertEqual(
            sanitize_dns_1123("production_gpu_slice!!!", max_length=15),
            "production-gpu",
        )

        # Test leading/trailing dashes strip
        self.assertEqual(sanitize_dns_1123("-my-cluster-"), "my-cluster")

        # Test empty result
        self.assertEqual(sanitize_dns_1123("!!!"), "")

    def test_get_sanitized_cluster_name(self):
        mock_scheduler = mock.Mock()

        # Case 1: Name found
        mock_scheduler.get_cluster_name.return_value = "GKE_cluster_123!!!"
        self.assertEqual(
            get_sanitized_cluster_name(mock_scheduler), "gke-cluster-123"
        )

        # Case 2: Name not found (raises ClusterDiscoveryException)
        mock_scheduler.get_cluster_name.return_value = None
        with self.assertRaisesRegex(
            ClusterDiscoveryException, "Failed to retrieve cluster name"
        ):
            get_sanitized_cluster_name(mock_scheduler)

        # Case 3: Name is empty after sanitization
        mock_scheduler.get_cluster_name.return_value = "!!!"
        with self.assertRaisesRegex(
            ClusterDiscoveryException, "could not be sanitized"
        ):
            get_sanitized_cluster_name(mock_scheduler)
