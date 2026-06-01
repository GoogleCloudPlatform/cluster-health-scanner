"""Tests for cluster name utilities."""

import pytest
from chs.utils.cluster import sanitize_dns_1123, get_sanitized_cluster_name
from chs.exceptions import ClusterDiscoveryException
from unittest import mock


def test_sanitize_dns_1123():
    # Test simple lowercasing and stripping
    assert sanitize_dns_1123("Simple-Name") == "simple-name"

    # Test spaces and underscores to dashes
    assert sanitize_dns_1123("my_cluster name") == "my-cluster-name"

    # Test non-alphanumeric stripping
    assert sanitize_dns_1123("cluster@name!") == "clustername"
    assert sanitize_dns_1123("cluster_name_$$$") == "cluster-name"

    # Test truncation
    assert (
        sanitize_dns_1123(
            "very-long-cluster-name-that-exceeds-limit", max_length=15
        )
        == "very-long-clust"
    )

    # Test trailing dashes strip after truncation
    assert (
        sanitize_dns_1123("production_gpu_slice!!!", max_length=15)
        == "production-gpu"
    )

    # Test leading/trailing dashes strip
    assert sanitize_dns_1123("-my-cluster-") == "my-cluster"

    # Test empty result
    assert sanitize_dns_1123("!!!") == ""


def test_get_sanitized_cluster_name():
    mock_scheduler = mock.Mock()

    # Case 1: Name found
    mock_scheduler.get_cluster_name.return_value = "GKE_cluster_123!!!"
    assert get_sanitized_cluster_name(mock_scheduler) == "gke-cluster-123"

    # Case 2: Name not found (raises ClusterDiscoveryException)
    mock_scheduler.get_cluster_name.return_value = None
    with pytest.raises(
        ClusterDiscoveryException, match="Failed to retrieve cluster name"
    ):
        get_sanitized_cluster_name(mock_scheduler)

    # Case 3: Name is empty after sanitization
    mock_scheduler.get_cluster_name.return_value = "!!!"
    with pytest.raises(
        ClusterDiscoveryException, match="could not be sanitized"
    ):
        get_sanitized_cluster_name(mock_scheduler)
