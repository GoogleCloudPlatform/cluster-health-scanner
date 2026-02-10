"""Unit tests for CHS Schedulers."""

import subprocess
import unittest
from unittest import mock
from chs.schedulers.default import DefaultScheduler
from chs.schedulers.slurm import SlurmScheduler
import click


class TestSlurmScheduler(unittest.TestCase):
  """Tests for SlurmScheduler."""

  def test_name(self):
    scheduler = SlurmScheduler()
    self.assertEqual(scheduler.name, "slurm")

  def test_get_nodes_under_test_nodes_arg(self):
    scheduler = SlurmScheduler(nodes="node1,node2")
    self.assertEqual(scheduler.get_nodes_under_test(), ["node1", "node2"])

  def test_get_nodes_under_test_multiple_args_error(self):
    scheduler = SlurmScheduler(nodes="n1", nodelist="nl1")
    with self.assertRaisesRegex(click.UsageError, "Only one of"):
      scheduler.get_nodes_under_test()

  @mock.patch("subprocess.run")
  def test_get_nodes_under_test_nodelist_arg(self, mock_run):
    scheduler = SlurmScheduler(nodelist="node-[1-2]")
    mock_run.return_value.stdout = "node-1\nnode-2"

    nodes = scheduler.get_nodes_under_test()

    self.assertEqual(nodes, ["node-1", "node-2"])
    mock_run.assert_called_once()
    self.assertIn("scontrol", mock_run.call_args[0][0])

  @mock.patch("subprocess.run")
  def test_get_nodes_under_test_partition_arg(self, mock_run):
    """Tests get_nodes_under_test with partition argument."""
    scheduler = SlurmScheduler(partition="p1")
    mock_run.side_effect = [
        mock.Mock(stdout="node-[1-2]"),
        mock.Mock(stdout="node-1\nnode-2"),
    ]

    nodes = scheduler.get_nodes_under_test()

    self.assertEqual(nodes, ["node-1", "node-2"])
    self.assertEqual(mock_run.call_count, 2)
    self.assertIn("sinfo", mock_run.call_args_list[0][0][0])
    self.assertIn("scontrol", mock_run.call_args_list[1][0][0])

  def test_parse_job_ids(self):
    scheduler = SlurmScheduler()
    output = "Submitted batch job 123\nSubmitted batch job 456"
    self.assertEqual(scheduler.parse_job_ids(output), ["123", "456"])

  @mock.patch("subprocess.run")
  def test_check_jobs_running_true(self, mock_run):
    scheduler = SlurmScheduler()
    mock_run.return_value.stdout = "123"
    self.assertTrue(scheduler.check_jobs_running(["123"]))

  @mock.patch("subprocess.run")
  def test_check_jobs_running_false(self, mock_run):
    scheduler = SlurmScheduler()
    mock_run.return_value.stdout = ""
    self.assertFalse(scheduler.check_jobs_running(["123"]))


class TestDefaultScheduler(unittest.TestCase):
  """Tests for DefaultScheduler."""

  def test_name(self):
    scheduler = DefaultScheduler("gke")
    self.assertEqual(scheduler.name, "gke")

  def test_get_nodes_under_test(self):
    scheduler = DefaultScheduler("gke")
    self.assertEqual(scheduler.get_nodes_under_test(), [])

  def test_check_jobs_running(self):
    scheduler = DefaultScheduler("gke")
    self.assertFalse(scheduler.check_jobs_running(["123"]))

  def test_parse_job_ids(self):
    scheduler = DefaultScheduler("gke")
    self.assertEqual(scheduler.parse_job_ids("output"), [])


if __name__ == "__main__":
  unittest.main()
