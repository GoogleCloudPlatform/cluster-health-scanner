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

import os
import yaml
import pytest
import glob
import subprocess
from tests.integration.create_kind_cluster import (
    resolve_config_path,
    delete_kind_cluster,
)

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")

_created_clusters = set()


def get_tpu_configs():
    """Finds all TPU yaml configurations."""
    pattern = os.path.join(CONFIGS_DIR, "tpu", "**", "*.yaml")
    files = glob.glob(pattern, recursive=True)
    return [os.path.relpath(f, start=CONFIGS_DIR) for f in files]


def get_gpu_configs():
    """Finds all GPU yaml configurations."""
    pattern = os.path.join(CONFIGS_DIR, "gpu", "**", "*.yaml")
    files = glob.glob(pattern, recursive=True)
    return [os.path.relpath(f, start=CONFIGS_DIR) for f in files]


def pytest_generate_tests(metafunc):
    if "active_cluster" in metafunc.fixturenames:
        if metafunc.definition.get_closest_marker("tpu"):
            metafunc.parametrize(
                "active_cluster", get_tpu_configs(), indirect=True
            )
        elif metafunc.definition.get_closest_marker("gpu"):
            metafunc.parametrize(
                "active_cluster", get_gpu_configs(), indirect=True
            )


@pytest.fixture(scope="module")
def active_cluster(request):
    config_path_str = request.param

    print(
        "\n[SETUP] Provisioning cluster natively via create_kind_cluster.py "
        f"for {config_path_str}"
    )

    # Intercept parsing the yaml to hand to the test environment
    abs_config_path = resolve_config_path(config_path_str)
    with open(abs_config_path, "r") as f:
        config_data = yaml.safe_load(f) or {}

    cluster_name = config_data.get("cluster_name", "chs-cluster")
    _created_clusters.add(cluster_name)
    config_data["_config_path_str"] = config_path_str

    # RELABEL the cluster dynamically so nodes match the test config exactly
    try:
        subprocess.run(
            [
                "python3",
                "tests/integration/create_kind_cluster.py",
                config_path_str,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Failed to provision kind cluster dynamically! {e}")

    yield config_data


@pytest.fixture(scope="session", autouse=True)
def kind_cluster_bootstrap_and_cleanup():
    print(
        "\n[SETUP] Bootstrapping global 6-node Kind cluster to prevent "
        "teardown thrash..."
    )
    subprocess.run(
        [
            "python3",
            "tests/integration/create_kind_cluster.py",
            "tpu/tpu7x/multi_pool_2x2x2_2x2x4.yaml",
        ],
        check=True,
    )

    yield
    print(
        "\n[TEARDOWN] Tearing down any clusters booted during this session..."
    )
    for cluster_name in _created_clusters:
        delete_kind_cluster(cluster_name)
