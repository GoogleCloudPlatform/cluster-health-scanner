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
from pathlib import Path
import pytest
import glob
from tests.integration.create_kind_cluster import delete_kind_cluster
from tests.integration.test_utils import chs_cluster_state
from tests.integration.test_utils import integration_test_util as test_utils

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")
KEEP_CLUSTER_ENV_VAR = "CHS_INTEGRATION_KEEP_CLUSTER"
DEFAULT_KUBECONFIG = Path.home() / ".kube" / "chs-integration-kind.yaml"

_created_clusters = set()


def pytest_configure(config):
    """Keeps Kind contexts out of the user's default kubeconfig.

    Kind and kubectl write to $KUBECONFIG, so pointing it at a dedicated file
    when it is unset leaves ~/.kube/config untouched.
    """
    if not os.environ.get("KUBECONFIG"):
        DEFAULT_KUBECONFIG.parent.mkdir(parents=True, exist_ok=True)
        os.environ["KUBECONFIG"] = str(DEFAULT_KUBECONFIG)


def get_tpu_configs():
    """Finds all TPU yaml configurations."""
    pattern = os.path.join(CONFIGS_DIR, "tpu", "**", "*.yaml")
    files = glob.glob(pattern, recursive=True)
    return sorted(os.path.relpath(f, start=CONFIGS_DIR) for f in files)


def get_gpu_configs():
    """Finds all GPU yaml configurations."""
    pattern = os.path.join(CONFIGS_DIR, "gpu", "**", "*.yaml")
    files = glob.glob(pattern, recursive=True)
    return sorted(os.path.relpath(f, start=CONFIGS_DIR) for f in files)


def pytest_generate_tests(metafunc):
    if "active_cluster" in metafunc.fixturenames:
        params = [
            pytest.param(c, marks=pytest.mark.tpu) for c in get_tpu_configs()
        ] + [pytest.param(c, marks=pytest.mark.gpu) for c in get_gpu_configs()]
        metafunc.parametrize("active_cluster", params, indirect=True)


@pytest.fixture(scope="module")
def active_cluster(request):
    config_path_str = request.param

    print(
        "\n[SETUP] Provisioning cluster natively via create_kind_cluster.py "
        f"for {config_path_str}"
    )
    config_data = test_utils.provision_kind_cluster(config_path_str)
    _created_clusters.add(config_data.get("cluster_name", "chs-cluster"))
    yield config_data


@pytest.fixture(autouse=True)
def clean_chs_namespace(kind_cluster_bootstrap_and_cleanup):
    """Clears CHS Jobs, Services and Secrets a previous test may have left."""
    chs_cluster_state.reset_namespace()


@pytest.fixture(scope="session", autouse=True)
def kind_cluster_bootstrap_and_cleanup():
    print(
        "\n[SETUP] Bootstrapping global 6-node Kind cluster to prevent "
        "teardown thrash..."
    )
    bootstrap_config = test_utils.provision_kind_cluster(
        "tpu/tpu7x/multi_pool_2x2x2_2x2x4.yaml"
    )
    _created_clusters.add(bootstrap_config.get("cluster_name", "chs-cluster"))

    yield
    if os.environ.get(KEEP_CLUSTER_ENV_VAR):
        print(f"\n[TEARDOWN] {KEEP_CLUSTER_ENV_VAR} is set; keeping clusters.")
        return
    print(
        "\n[TEARDOWN] Tearing down any clusters booted during this session..."
    )
    for cluster_name in _created_clusters:
        delete_kind_cluster(cluster_name)
