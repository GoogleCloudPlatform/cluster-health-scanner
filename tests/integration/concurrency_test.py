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

"""Integration tests for concurrent CHS runs on disjoint nodes."""

import os
from pathlib import Path
import re
import subprocess
import sys
import time

import pytest

from chs.utils.app_data import ChsAppData
from tests.integration.mocks.mock_utils import resolve_node_mapping
from tests.integration.mocks.output_generator import RambleOutputGenerator
from tests.integration.test_utils import chs_cluster_state as cluster_state
from tests.integration.test_utils import integration_test_util as test_utils

REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_CONFIG = "concurrency/gpu_a4x_four_nodes.yaml"
TPU_CONFIG = "tpu/tpu7x/multi_pool_2x2x2_2x2x4.yaml"
TPU_DYNAMIC_SLICE_CONFIG = "concurrency/tpu7x_dynamic_slices.yaml"
GPU_TEST = "nvidia-smi"
TPU_TEST = "tpu-info"
OTHER_RUN_ID = "other-run"
UNKNOWN_RUN_ID = "never-ran"
UNKNOWN_RUN_WARNING = (
    f"No active CHS run '{UNKNOWN_RUN_ID}' was found on the cluster."
)
SKIP_WARNING_PATTERN = re.compile(
    r"Skipping (?P<count>\d+) node\(s\) targeted by CHS run "
    r"'(?P<run_id>[^']+)'\."
)
NO_NODES_ERROR = "No valid nodes remain available for testing."
INVALID_RUN_ID_ERROR = "is not a valid CHS run ID"
USAGE_ERROR_EXIT_CODE = 2
SUBPROCESS_TIMEOUT_SEC = 900
JOB_POLL_INTERVAL_SEC = 1

OTHER_RUN_JOB_FACTORIES = {
    "suspended": cluster_state.create_suspended_job,
    "running": cluster_state.create_running_job,
}


def _run_chs(cluster: dict, args: list[str]) -> tuple[int, str]:
    """Runs the CHS CLI in-process and returns its exit code and output."""
    output_gen = RambleOutputGenerator(cluster["_config_path_str"])
    with test_utils.mock_chs_environment(cluster["cluster_name"], output_gen):
        exit_code, stdout, stderr = test_utils.run_cli_and_capture(args)
    return exit_code, f"{stdout}\n{stderr}"


def _run_test(
    cluster: dict, test: str, target_args: list[str], run_id: str
) -> tuple[int, str]:
    return _run_chs(
        cluster,
        ["run", "--tests", test, *target_args, "--run-id", run_id, "-y"],
    )


def _flat(output: str) -> str:
    """Joins the output into one line, undoing Rich's line wrapping."""
    return " ".join(output.split())


def _owned_by(names: set[str], run_id: str) -> set[str]:
    """Returns the names of the objects of a run started with --run-id.

    CHS appends a random suffix to the run ID, so the objects of run
    `<run_id>-<suffix>` are named `chs-<run_id>-<suffix>[-...]`.
    """
    return {name for name in names if name.startswith(f"chs-{run_id}-")}


def _assert_run_left_nothing_behind(run_id: str) -> None:
    assert cluster_state.namespace_exists()
    leftovers = {
        "Jobs": _owned_by(cluster_state.job_names(), run_id),
        "Services": _owned_by(cluster_state.service_names(), run_id),
        "Secrets": _owned_by(cluster_state.secret_names(), run_id),
    }
    for kind, names in leftovers.items():
        assert not names, f"Run {run_id} left {kind}: {names}"


def _skip_counts_by_run(log_text: str) -> dict[str, set[int]]:
    """Returns the skipped node counts each other run caused, from warnings.

    A run may log its skip warning more than once, so every count it
    reported is kept.
    """
    skipped = {}
    for match in SKIP_WARNING_PATTERN.finditer(log_text):
        skipped.setdefault(match["run_id"], set()).add(int(match["count"]))
    return skipped


def _assert_skipped_for_other_run(log_text: str, nodes: set[str]) -> None:
    skipped = _skip_counts_by_run(log_text)
    assert skipped == {OTHER_RUN_ID: {len(nodes)}}, log_text


def _assert_no_nodes_left(exit_code: int, output: str, run_id: str) -> None:
    assert exit_code != 0, f"Run {run_id} ran on targeted nodes:\n{output}"
    assert NO_NODES_ERROR in _flat(output)
    _assert_run_left_nothing_behind(run_id)


def _assert_succeeded(exit_code: int, output: str, run_id: str) -> dict:
    assert exit_code == 0, f"Run {run_id} failed:\n{output}"
    boh, boh_file = test_utils.get_latest_boh()
    assert boh, f"Run {run_id} produced no Bill of Health."
    assert run_id in boh_file
    _assert_run_left_nothing_behind(run_id)
    return boh


def _tested_nodes(boh: dict) -> set[str]:
    return {
        resource["node_name"]
        for result in boh["test_results"]
        for resource in result.get("target_resources", [])
    }


def _assert_job_untouched(job_name: str, kind: str) -> None:
    assert job_name in cluster_state.job_names()
    if kind == "running":
        phases = cluster_state.pod_phases(job_name)
        assert phases and set(phases) == {"Running"}, phases


@pytest.fixture(scope="class")
def gpu_cluster():
    return test_utils.provision_kind_cluster(GPU_CONFIG)


@pytest.fixture(scope="class")
def tpu_cluster():
    return test_utils.provision_kind_cluster(TPU_CONFIG)


@pytest.fixture(scope="class")
def tpu_dynamic_slice_cluster():
    return test_utils.provision_kind_cluster(TPU_DYNAMIC_SLICE_CONFIG)


class _MockedChsProcess:
    """A GPU `chs run` in its own process, like another terminal or user."""

    def __init__(
        self,
        run_id: str,
        process: subprocess.Popen,
        app_data: ChsAppData,
        log_path: Path,
    ) -> None:
        self.run_id = run_id
        self.process = process
        self.app_data = app_data
        self.log_path = log_path

    @classmethod
    def start(
        cls, run_id: str, nodes: str, tmp_path: Path, data_home: Path
    ) -> "_MockedChsProcess":
        """Starts the run with CHS app data stored under data_home."""
        app_data = ChsAppData(data_dir=str(data_home / "chs"))
        app_data.set_storage_config(ChsAppData().get_storage_config())
        log_path = tmp_path / f"{run_id}.log"
        chs_args = [
            "run",
            "--tests",
            GPU_TEST,
            "--nodes",
            nodes,
            "--run-id",
            run_id,
            "-y",
        ]
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tests.integration.test_utils.mocked_chs_cli",
                    GPU_CONFIG,
                    "--",
                    *chs_args,
                ],
                cwd=REPO_ROOT,
                env={**os.environ, "XDG_DATA_HOME": str(data_home)},
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        return cls(run_id, process, app_data, log_path)

    def kill_if_running(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()

    def _recorded_run_id(self) -> str:
        """Returns the generated run ID that CHS recorded for this run."""
        recorded = [
            run["run_id"]
            for run in self.app_data.get_all_runs().values()
            if run.get("run_id", "").startswith(f"{self.run_id}-")
        ]
        assert (
            len(recorded) == 1
        ), f"Expected one runs.json entry for {self.run_id}, found {recorded}."
        return recorded[0]

    def output(self) -> str:
        return self.log_path.read_text()

    def assert_succeeded(self) -> None:
        self.process.wait(timeout=SUBPROCESS_TIMEOUT_SEC)
        assert (
            self.process.returncode == 0
        ), f"Run {self.run_id} failed:\n{self.output()}"
        recorded_run_id = self._recorded_run_id()
        boh, _ = test_utils.get_run_boh(self.app_data, recorded_run_id)
        assert boh, f"Run {self.run_id} produced no Bill of Health."
        _assert_run_left_nothing_behind(self.run_id)


def _wait_for_jobs_of(run: _MockedChsProcess) -> None:
    """Blocks until the run has submitted Jobs to the cluster."""
    deadline = time.monotonic() + SUBPROCESS_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if _owned_by(cluster_state.job_names(), run.run_id):
            return
        if run.process.poll() is not None:
            pytest.fail(
                f"Run {run.run_id} exited before submitting Jobs:\n"
                f"{run.output()}"
            )
        time.sleep(JOB_POLL_INTERVAL_SEC)
    pytest.fail(f"Run {run.run_id} submitted no Jobs in time.")


def _assert_staggered_disjoint_runs_succeed(
    tmp_path: Path, first_data_home: Path, second_data_home: Path
) -> None:
    """Starts a second run on disjoint nodes once the first has Jobs."""
    first = _MockedChsProcess.start(
        "in-flight", "node-1,node-2", tmp_path, first_data_home
    )
    runs = [first]
    try:
        _wait_for_jobs_of(first)
        runs.append(
            _MockedChsProcess.start(
                "concurrent", "node-3,node-4", tmp_path, second_data_home
            )
        )
        for run in runs:
            run.assert_succeeded()
    finally:
        for run in runs:
            run.kill_if_running()


@pytest.mark.gpu
class TestGpuNodeConflicts:
    """Node conflicts between CHS runs targeting GPU nodes with --nodes."""

    @pytest.mark.parametrize("kind", OTHER_RUN_JOB_FACTORIES)
    def test_partially_overlapping_run_skips_nodes_of_other_run(
        self, gpu_cluster, caplog, kind
    ):
        job_name = OTHER_RUN_JOB_FACTORIES[kind](
            OTHER_RUN_ID, ["node-1", "node-2"]
        )

        exit_code, output = _run_test(
            gpu_cluster, GPU_TEST, ["--nodes", "node-2,node-3"], "overlap"
        )

        boh = _assert_succeeded(exit_code, output, "overlap")
        assert _tested_nodes(boh) == {"node-3"}
        _assert_skipped_for_other_run(caplog.text, {"node-2"})
        _assert_job_untouched(job_name, kind)

    @pytest.mark.parametrize("kind", OTHER_RUN_JOB_FACTORIES)
    def test_fully_overlapping_run_finds_no_nodes(
        self, gpu_cluster, caplog, kind
    ):
        job_name = OTHER_RUN_JOB_FACTORIES[kind](
            OTHER_RUN_ID, ["node-1", "node-2"]
        )

        exit_code, output = _run_test(
            gpu_cluster, GPU_TEST, ["--nodes", "node-1,node-2"], "full-overlap"
        )

        _assert_no_nodes_left(exit_code, output, "full-overlap")
        _assert_skipped_for_other_run(caplog.text, {"node-1", "node-2"})
        _assert_job_untouched(job_name, kind)

    @pytest.mark.parametrize("kind", OTHER_RUN_JOB_FACTORIES)
    def test_disjoint_run_succeeds_and_keeps_other_run(
        self, gpu_cluster, caplog, kind
    ):
        job_name = OTHER_RUN_JOB_FACTORIES[kind](
            OTHER_RUN_ID, ["node-1", "node-2"]
        )
        service_name = cluster_state.create_worker_service(OTHER_RUN_ID)

        exit_code, output = _run_test(
            gpu_cluster, GPU_TEST, ["--nodes", "node-3,node-4"], "disjoint"
        )

        boh = _assert_succeeded(exit_code, output, "disjoint")
        assert _tested_nodes(boh) == {"node-3", "node-4"}
        assert not _skip_counts_by_run(caplog.text)
        _assert_job_untouched(job_name, kind)
        assert service_name in cluster_state.service_names()

    def test_completed_job_of_other_run_still_blocks(self, gpu_cluster, caplog):
        job_name = cluster_state.create_completed_job(
            OTHER_RUN_ID, ["node-1", "node-2"]
        )

        exit_code, output = _run_test(
            gpu_cluster, GPU_TEST, ["--nodes", "node-2,node-3"], "after-done"
        )

        boh = _assert_succeeded(exit_code, output, "after-done")
        assert _tested_nodes(boh) == {"node-3"}
        _assert_skipped_for_other_run(caplog.text, {"node-2"})
        assert job_name in cluster_state.job_names()

    def test_unlabeled_job_does_not_block(self, gpu_cluster, caplog):
        job_name = cluster_state.create_unlabeled_suspended_job(
            ["node-1", "node-2"]
        )

        exit_code, output = _run_test(
            gpu_cluster, GPU_TEST, ["--nodes", "node-1,node-2"], "unlabeled"
        )

        boh = _assert_succeeded(exit_code, output, "unlabeled")
        assert _tested_nodes(boh) == {"node-1", "node-2"}
        assert not _skip_counts_by_run(caplog.text)
        assert job_name in cluster_state.job_names()

    def test_job_without_managed_by_label_does_not_block(
        self, gpu_cluster, caplog
    ):
        job_name = cluster_state.create_instance_only_suspended_job(
            OTHER_RUN_ID, ["node-1", "node-2"]
        )

        exit_code, output = _run_test(
            gpu_cluster, GPU_TEST, ["--nodes", "node-1,node-2"], "foreign"
        )

        boh = _assert_succeeded(exit_code, output, "foreign")
        assert _tested_nodes(boh) == {"node-1", "node-2"}
        assert not _skip_counts_by_run(caplog.text)
        assert job_name in cluster_state.job_names()

    def test_cancel_deletes_only_that_runs_objects(self, gpu_cluster):
        other_jobs = {
            cluster_state.create_suspended_job(OTHER_RUN_ID, ["node-1"]),
            cluster_state.create_running_job(OTHER_RUN_ID, ["node-2"]),
        }
        other_service = cluster_state.create_worker_service(OTHER_RUN_ID)
        bystander_job = cluster_state.create_running_job(
            "bystander", ["node-3"]
        )
        bystander_service = cluster_state.create_worker_service("bystander")
        runs_before = ChsAppData().get_all_runs()

        exit_code, output = _run_chs(
            gpu_cluster, ["cancel", OTHER_RUN_ID, "--auto-approve"]
        )

        assert exit_code == 0, output
        assert f"Successfully canceled CHS run '{OTHER_RUN_ID}'." in _flat(
            output
        )
        assert not other_jobs & cluster_state.job_names()
        assert other_service not in cluster_state.service_names()
        _assert_job_untouched(bystander_job, "running")
        assert bystander_service in cluster_state.service_names()
        assert cluster_state.namespace_exists()
        assert ChsAppData().get_all_runs() == runs_before

    def test_cancel_unknown_run_warns_and_keeps_other_runs(self, gpu_cluster):
        other_job = cluster_state.create_running_job(OTHER_RUN_ID, ["node-1"])
        other_service = cluster_state.create_worker_service(OTHER_RUN_ID)

        exit_code, output = _run_chs(
            gpu_cluster, ["cancel", UNKNOWN_RUN_ID, "--auto-approve"]
        )

        assert exit_code == 0, output
        assert UNKNOWN_RUN_WARNING in _flat(output)
        assert "Successfully canceled" not in output
        _assert_job_untouched(other_job, "running")
        assert other_service in cluster_state.service_names()

    def test_cancel_spares_run_whose_id_extends_cancelled_id(self, gpu_cluster):
        cancelled, sibling = "x-ab12", "x-ab12-cd34"
        cancelled_job = cluster_state.create_running_job(cancelled, ["node-1"])
        sibling_job = cluster_state.create_running_job(sibling, ["node-2"])
        cancelled_objects = {
            "service": cluster_state.create_worker_service(cancelled),
            "secret": cluster_state.create_ssh_secret(cancelled, cancelled_job),
        }
        sibling_objects = {
            "service": cluster_state.create_worker_service(sibling),
            "secret": cluster_state.create_ssh_secret(sibling, sibling_job),
        }

        exit_code, output = _run_chs(
            gpu_cluster, ["cancel", cancelled, "--auto-approve"]
        )

        assert exit_code == 0, output
        assert cancelled_job not in cluster_state.job_names()
        assert cancelled_objects["service"] not in cluster_state.service_names()
        assert cancelled_objects["secret"] not in cluster_state.secret_names()
        _assert_job_untouched(sibling_job, "running")
        assert sibling_objects["service"] in cluster_state.service_names()
        assert sibling_objects["secret"] in cluster_state.secret_names()

    @pytest.mark.parametrize(
        "run_id",
        [
            "Upper-case",
            "trailing-dash-",
            "under_score",
            "x" * 64,
            "20261014-084121_cluster-lite-ab12",
        ],
    )
    def test_cancel_rejects_invalid_run_id(self, gpu_cluster, run_id):
        other_job = cluster_state.create_running_job(OTHER_RUN_ID, ["node-1"])

        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.integration.test_utils.mocked_chs_cli",
                GPU_CONFIG,
                "--",
                "cancel",
                run_id,
                "--auto-approve",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SEC,
        )

        output = f"{res.stdout}\n{res.stderr}"
        assert res.returncode == USAGE_ERROR_EXIT_CODE, output
        assert INVALID_RUN_ID_ERROR in _flat(output)
        _assert_job_untouched(other_job, "running")

    def test_run_started_while_disjoint_run_is_in_flight_succeeds(
        self, gpu_cluster, tmp_path
    ):
        _assert_staggered_disjoint_runs_succeed(
            tmp_path,
            first_data_home=tmp_path / "first-user",
            second_data_home=tmp_path / "second-user",
        )

    def test_same_user_staggered_disjoint_runs_both_recorded(
        self, gpu_cluster, tmp_path
    ):
        shared_data_home = tmp_path / "same-user"
        _assert_staggered_disjoint_runs_succeed(
            tmp_path,
            first_data_home=shared_data_home,
            second_data_home=shared_data_home,
        )


@pytest.mark.tpu
class TestTpuNodepoolConflicts:
    """Node conflicts between CHS runs targeting TPU nodepools."""

    @pytest.mark.parametrize("targeted", ["whole_pool", "one_node"])
    def test_targeted_nodepool_is_skipped_as_a_whole(
        self, tpu_cluster, caplog, targeted
    ):
        pool_nodes = resolve_node_mapping()["tpu-v7-pool-01"]
        targeted_nodes = (
            pool_nodes if targeted == "whole_pool" else pool_nodes[:1]
        )
        job_name = cluster_state.create_suspended_job(
            OTHER_RUN_ID, targeted_nodes
        )

        exit_code, output = _run_test(
            tpu_cluster,
            TPU_TEST,
            ["--nodepools", "tpu-v7-pool-01"],
            "tpu-overlap",
        )

        _assert_no_nodes_left(exit_code, output, "tpu-overlap")
        _assert_skipped_for_other_run(caplog.text, set(targeted_nodes))
        assert job_name in cluster_state.job_names()

    def test_run_proceeds_on_nodepool_not_targeted(self, tpu_cluster, caplog):
        node_mapping = resolve_node_mapping()
        job_name = cluster_state.create_suspended_job(
            OTHER_RUN_ID, node_mapping["tpu-v7-pool-02"][:1]
        )

        exit_code, output = _run_test(
            tpu_cluster,
            TPU_TEST,
            ["--nodepools", "tpu-v7-pool-01,tpu-v7-pool-02"],
            "tpu-partial",
        )

        boh = _assert_succeeded(exit_code, output, "tpu-partial")
        assert _tested_nodes(boh) == set(node_mapping["tpu-v7-pool-01"])
        _assert_skipped_for_other_run(
            caplog.text, set(node_mapping["tpu-v7-pool-02"][:1])
        )
        assert job_name in cluster_state.job_names()

    def test_disjoint_nodepool_succeeds_and_keeps_other_run(
        self, tpu_cluster, caplog
    ):
        other_pool_nodes = resolve_node_mapping()["tpu-v7-pool-02"]
        job_name = cluster_state.create_suspended_job(
            OTHER_RUN_ID, other_pool_nodes
        )

        exit_code, output = _run_test(
            tpu_cluster,
            TPU_TEST,
            ["--nodepools", "tpu-v7-pool-01"],
            "tpu-disjoint",
        )

        boh = _assert_succeeded(exit_code, output, "tpu-disjoint")
        assert _tested_nodes(boh) == set(
            resolve_node_mapping()["tpu-v7-pool-01"]
        )
        assert not _skip_counts_by_run(caplog.text)
        assert job_name in cluster_state.job_names()


@pytest.mark.tpu
class TestTpuDynamicSliceConflicts:
    """Node conflicts through dynamic TPU Slices of other CHS runs.

    Kind has no Slice controller, so a run that provisions its own Slice
    never sees it become active; only skipped pools are exercised here.
    """

    def test_pool_covered_by_slice_of_other_run_is_skipped(
        self, tpu_dynamic_slice_cluster, caplog
    ):
        slice_name = cluster_state.create_slice(
            OTHER_RUN_ID, "tpu-v7-dynamic-01", ["partition-01"]
        )
        pool_nodes = resolve_node_mapping()["tpu-v7-dynamic-01"]

        exit_code, output = _run_test(
            tpu_dynamic_slice_cluster,
            TPU_TEST,
            ["--nodepools", "tpu-v7-dynamic-01"],
            "slice-overlap",
        )

        _assert_no_nodes_left(exit_code, output, "slice-overlap")
        _assert_skipped_for_other_run(caplog.text, set(pool_nodes))
        assert cluster_state.slice_names() == {slice_name}
