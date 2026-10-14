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

import contextlib
import io
import json
import os
import re
import subprocess
from typing import Any, Dict, Optional, Tuple
import pytest
import yaml

from chs.cli import cli
import chs.exceptions as ex
from chs.utils.app_data import ChsAppData
from tests.integration.create_kind_cluster import (
    resolve_config_path,
    sanitize_k8s_name,
)
from tests.integration.mocks.gcp_mock import mock_gce_instances, mock_gcs_bucket
from tests.integration.mocks.ramble_mock import mock_ramble_workloads
from tests.integration.test_utils.validator import (
    IntegrationValidator,
    ValidationReport,
)


import sys


class TeeStream:
    """Stream wrapper that writes to a stream and StringIO buffer."""

    def __init__(self, original_stream, buffer_io):
        self.original_stream = original_stream
        self.buffer_io = buffer_io

    def write(self, data: str) -> int:
        ret = self.original_stream.write(data)
        self.buffer_io.write(data)
        return ret

    def flush(self) -> None:
        self.original_stream.flush()
        self.buffer_io.flush()

    def isatty(self) -> bool:
        return getattr(self.original_stream, "isatty", lambda: False)()


def run_suite_and_capture(
    suite_name: str, nodepools: str
) -> Tuple[int, str, str]:
    """Runs a CHS suite while capturing exit code, stdout, and stderr."""
    return run_cli_and_capture(
        ["run", "--suite", suite_name, "--nodepools", nodepools, "-y"]
    )


def run_cli_and_capture(args: list[str]) -> Tuple[int, str, str]:
    """Runs a CHS CLI command while capturing exit code, stdout, and stderr."""
    stdout_io = io.StringIO()
    stderr_io = io.StringIO()
    tee_stdout = TeeStream(sys.stdout, stdout_io)
    tee_stderr = TeeStream(sys.stderr, stderr_io)
    exit_code = 0
    try:
        with (
            contextlib.redirect_stdout(tee_stdout),
            contextlib.redirect_stderr(tee_stderr),
        ):
            cli(args, standalone_mode=False)
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 1
    except Exception as e:
        exit_code = 1
        tee_stderr.write(str(e))

    return exit_code, stdout_io.getvalue(), stderr_io.getvalue()


def get_latest_boh(
    run_id: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Retrieves the Bill of Health dict and file path for the latest run."""
    app_data = ChsAppData()
    if not run_id:
        try:
            run_id = app_data.get_latest_run_id()
        except ex.NoPreviousRunsFoundException:
            return None, None

    if not run_id:
        return None, None
    return get_run_boh(app_data, run_id)


def get_run_boh(
    app_data: ChsAppData, run_id: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Retrieves the Bill of Health dict and file path for a run."""
    try:
        workspace_name = app_data.resolve_run_id(run_id)
    except ex.WorkspaceNotFoundException:
        return None, None
    workspace_path = os.path.join(app_data.get_workspaces_dir(), workspace_name)
    boh_file = os.path.join(workspace_path, "bill_of_health.json")

    if os.path.exists(boh_file):
        with open(boh_file, "r") as f:
            return json.load(f), boh_file
    return None, None


def validate_integration_run(
    blueprint_config: Dict[str, Any],
    exit_code: int,
    stdout: str = "",
    stderr: str = "",
) -> ValidationReport:
    """Evaluates the integration test run using IntegrationValidator."""
    combined_output = f"{stdout}\n{stderr}"
    run_id_match = re.search(r"Run ID:\s*([a-zA-Z0-9_\-]+)", combined_output)
    run_id = run_id_match.group(1) if run_id_match else None

    boh = None
    if run_id:
        boh, _ = get_latest_boh(run_id)
    elif exit_code == 0:
        boh, _ = get_latest_boh()

    validator = IntegrationValidator()
    return validator.validate_boh_and_output(
        blueprint_config=blueprint_config,
        boh=boh,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
    )


@contextlib.contextmanager
def mock_chs_environment(cluster_name: str, output_gen):
    """Consolidated context manager mocking all external network boundaries
    (GCP, GCS, Ramble) required to run CHS suites.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock_gce_instances(cluster_name))
        stack.enter_context(mock_gcs_bucket())
        stack.enter_context(mock_ramble_workloads(output_gen))
        yield


def get_nodepools_csv(cluster_config: dict) -> str:
    """Extracts a comma-separated string of nodepools from cluster config."""
    nodepools = [
        pool.get("name")
        for pool in cluster_config.get("nodepools", [])
        if isinstance(pool, dict)
    ]
    if not nodepools:
        config_path = cluster_config.get("_config_path_str", "Unknown config")
        pytest.skip(f"No nodepools defined in config: {config_path}")

    return ",".join(nodepools)


def provision_kind_cluster(config_path_str: str) -> Dict[str, Any]:
    """Relabels the Kind cluster to match a blueprint and returns its config."""
    with open(resolve_config_path(config_path_str), "r") as f:
        config_data = yaml.safe_load(f) or {}

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
    _require_kind_context(config_data.get("cluster_name", "chs-cluster"))
    config_data["_config_path_str"] = config_path_str
    return config_data


def _require_kind_context(cluster_name: str) -> None:
    """Aborts the session unless kubectl targets the local Kind cluster."""
    expected = f"kind-{sanitize_k8s_name(cluster_name)}"
    res = subprocess.run(
        ["kubectl", "config", "current-context"],
        capture_output=True,
        text=True,
        check=False,
    )
    current = res.stdout.strip()
    if current != expected:
        pytest.exit(
            f"Refusing to run integration tests: kubectl context is "
            f"'{current}', expected '{expected}'.",
            returncode=1,
        )
