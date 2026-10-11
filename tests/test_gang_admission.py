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

"""Unit tests for gang admission planner."""

from typing import Any
import unittest
from chs.schedulers.gang_admission import plan_admissions


def _make_job(
    name: str,
    nodes: list[str] | None = None,
    suspended: bool = True,
    finished: bool = False,
    creation_timestamp: str = "2026-10-07T00:00:00Z",
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "name": name,
        "creationTimestamp": creation_timestamp,
    }
    spec: dict[str, Any] = {"suspend": suspended}
    if nodes is not None:
        expr = {
            "key": "kubernetes.io/hostname",
            "operator": "In",
            "values": nodes,
        }
        term = {"matchExpressions": [expr]}
        node_affinity = {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [term]
            }
        }
        pod_spec = {"affinity": {"nodeAffinity": node_affinity}}
        spec["template"] = {"spec": pod_spec}
    job: dict[str, Any] = {
        "metadata": metadata,
        "spec": spec,
    }
    if finished:
        job["status"] = {"conditions": [{"type": "Complete", "status": "True"}]}
    return job


class TestGangAdmission(unittest.TestCase):
    """Tests for plan_admissions function."""

    def test_disjoint_single_node_jobs_all_start(self):
        job1 = _make_job("job1", ["node1"], suspended=True)
        job2 = _make_job("job2", ["node2"], suspended=True)
        job3 = _make_job("job3", ["node3"], suspended=True)
        candidates = ["job1", "job2", "job3"]
        admitted = plan_admissions([job1, job2, job3], candidates)
        self.assertEqual(admitted, ["job1", "job2", "job3"])

    def test_multi_node_job_waits_while_node_used(self):
        job1 = _make_job("job1", ["node1"], suspended=False)
        job2 = _make_job("job2", ["node1", "node2"], suspended=True)
        candidates = ["job2"]
        admitted = plan_admissions([job1, job2], candidates)
        self.assertEqual(admitted, [])

    def test_finished_jobs_free_nodes(self):
        job1 = _make_job("job1", ["node1"], suspended=False, finished=True)
        job2 = _make_job("job2", ["node1", "node2"], suspended=True)
        candidates = ["job2"]
        admitted = plan_admissions([job1, job2], candidates)
        self.assertEqual(admitted, ["job2"])

    def test_failed_jobs_free_nodes(self):
        job1 = _make_job("job1", ["node1"], suspended=False)
        job1["status"] = {
            "conditions": [
                {"type": "Failed", "status": "True", "reason": "BackoffLimit"}
            ]
        }
        job2 = _make_job("job2", ["node1"], suspended=True)
        self.assertEqual(plan_admissions([job1, job2], ["job2"]), ["job2"])

    def test_interim_job_conditions_do_not_free_nodes(self):
        """Only Complete/Failed mean a Job's pods are gone."""
        job1 = _make_job("job1", ["node1"], suspended=False)
        job1["status"] = {
            "conditions": [
                {"type": "SuccessCriteriaMet", "status": "True"},
                {"type": "FailureTarget", "status": "True"},
            ]
        }
        job2 = _make_job("job2", ["node1"], suspended=True)
        self.assertEqual(plan_admissions([job1, job2], ["job2"]), [])

    def test_two_jobs_on_same_node_start_in_queue_order(self):
        job1 = _make_job(
            "job1",
            ["node1"],
            suspended=True,
            creation_timestamp="2026-10-07T00:00:02Z",
        )
        job2 = _make_job(
            "job2",
            ["node1"],
            suspended=True,
            creation_timestamp="2026-10-07T00:00:01Z",
        )
        admitted = plan_admissions([job2, job1], ["job1", "job2"])
        self.assertEqual(admitted, ["job1"])

    def test_waiting_multi_node_job_reserves_nodes(self):
        job1 = _make_job("job1", ["node1"], suspended=False)
        job2 = _make_job(
            "job2",
            ["node1", "node2"],
            suspended=True,
            creation_timestamp="2026-10-07T00:00:01Z",
        )
        job3 = _make_job(
            "job3",
            ["node2"],
            suspended=True,
            creation_timestamp="2026-10-07T00:00:02Z",
        )
        candidates = ["job2", "job3"]
        admitted = plan_admissions([job1, job2, job3], candidates)
        self.assertEqual(admitted, [])

    def test_non_candidates_never_returned_but_started_occupy_nodes(self):
        job1 = _make_job("job1", ["node1"], suspended=False)
        job2 = _make_job("job2", ["node2"], suspended=True)
        job3 = _make_job("job3", ["node1"], suspended=True)
        job4 = _make_job("job4", ["node3"], suspended=True)
        candidates = ["job3", "job4"]
        admitted = plan_admissions([job1, job2, job3, job4], candidates)
        self.assertEqual(admitted, ["job4"])

    def test_job_with_no_hostname_affinity_needs_all_nodes(self):
        job_no_affinity = _make_job(
            "job-broad",
            nodes=None,
            suspended=True,
            creation_timestamp="2026-10-07T00:00:01Z",
        )
        job_node1 = _make_job(
            "job-pinned",
            ["node1"],
            suspended=True,
            creation_timestamp="2026-10-07T00:00:02Z",
        )
        admitted = plan_admissions(
            [job_no_affinity, job_node1], ["job-broad", "job-pinned"]
        )
        self.assertEqual(admitted, ["job-broad"])

        job_running = _make_job("job-running", ["node2"], suspended=False)
        admitted_with_running = plan_admissions(
            [job_running, job_no_affinity], ["job-broad"]
        )
        self.assertEqual(admitted_with_running, [])

        job_broad_running = _make_job(
            "job-broad-running", nodes=None, suspended=False
        )
        admitted_pinned = plan_admissions(
            [job_broad_running, job_node1], ["job-pinned"]
        )
        self.assertEqual(admitted_pinned, [])

    def test_job_without_pod_template_needs_all_nodes(self):
        job_without_template = {
            "metadata": {"name": "job-odd"},
            "spec": {"suspend": False, "template": {}},
        }
        job_pinned = _make_job("job-pinned", ["node1"], suspended=True)
        admitted = plan_admissions(
            [job_without_template, job_pinned], ["job-pinned"]
        )
        self.assertEqual(admitted, [])


if __name__ == "__main__":
    unittest.main()
