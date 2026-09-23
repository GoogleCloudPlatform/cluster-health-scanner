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

"""Unit tests for the shipped test packages.

Two kinds of test live here. The first checks what every package must declare,
so a new test cannot ship half-described. The second covers the packages that
carry real code of their own: the pairwise NCCL test, which chooses node pairs
from the cluster topology, and the TPU collectives tests, which pick their
benchmark input and mesh from the workload.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from chs.testkit import TestRequest, available_tests, load_plugin

# A two-subblock cluster. Small enough to write the expected pairs out by
# hand, large enough that the within-subblock and across-subblock passes both
# produce something.
TOPOLOGY = {
    "block-a": {
        "sub-1": ["node-1", "node-2"],
        "sub-2": ["node-3", "node-4"],
    }
}


def request(
    nodes=("node-1", "node-2"), workspace_root=None, configs=None, **overrides
):
    """Builds the smallest request a plugin will act on."""
    variables = {
        "scheduler": "kubernetes",
        "machine_type": "machine-x",
        "nodes": list(nodes),
    }
    variables.update(overrides)
    return TestRequest(
        variables=variables,
        base_prefix="chs-run",
        configs=configs,
        workspace_root=workspace_root,
    )


def plugin(test_name):
    return load_plugin(test_name)


class ShippedPackageTest(unittest.TestCase):
    """What every test package has to declare to be usable.

    These run over whatever is on disk, so a package added later is held to
    the same rules without this file being touched.
    """

    def setUp(self):
        self.tests = available_tests()

    def test_there_are_tests_to_check(self):
        """Guards the loops below against silently checking nothing."""
        self.assertGreater(len(self.tests), 10)

    def test_every_package_loads(self):
        for name in self.tests:
            with self.subTest(test=name):
                self.assertIsNotNone(plugin(name))

    def test_every_package_names_itself_after_its_directory(self):
        """The directory name is how a test is requested, so it must match."""
        for name in self.tests:
            with self.subTest(test=name):
                self.assertEqual(name, plugin(name).name)

    def test_every_package_renders_an_application(self):
        for name in self.tests:
            with self.subTest(test=name):
                self.assertTrue(plugin(name).application)

    def test_every_package_offers_a_workload(self):
        """A test with no workload would build no experiments."""
        for name in self.tests:
            with self.subTest(test=name):
                self.assertTrue(plugin(name).default_workloads)

    def test_every_package_describes_itself(self):
        """``chs list`` and the HTML report both read these."""
        for name in self.tests:
            with self.subTest(test=name):
                p = plugin(name)
                self.assertTrue(p.category, "No category.")
                self.assertTrue(p.description, "No description.")
                self.assertTrue(p.long_description, "No long description.")

    def test_a_workload_is_never_a_bare_string(self):
        """A string would iterate character by character further down."""
        for name in self.tests:
            with self.subTest(test=name):
                self.assertNotIsInstance(plugin(name).default_workloads, str)

    def test_a_test_that_hides_the_workload_runs_only_one(self):
        """Two workloads would produce two experiments with the same name."""
        for name in self.tests:
            p = plugin(name)
            if p.workload_in_experiment_name:
                continue
            with self.subTest(test=name):
                self.assertEqual(1, len(p.default_workloads))

    def test_experiment_names_are_unique_across_the_shipped_tests(self):
        """Two tests answering to one name would overwrite each other."""
        seen = {}
        for name in self.tests:
            p = plugin(name)
            for workload in p.default_workloads:
                experiment = p.experiment_name("chs-run", workload)
                with self.subTest(experiment=experiment):
                    self.assertNotIn(experiment, seen)
                seen[experiment] = name


class PairwisePairingTest(unittest.TestCase):
    """How the pairwise NCCL test chooses which nodes to measure."""

    def setUp(self):
        self.module = _pairwise_module()

    def test_nodes_in_a_subblock_are_paired_with_their_neighbour(self):
        pairs = self.module.generate_topological_pairs(
            {"block-a": {"sub-1": ["node-1", "node-2", "node-3", "node-4"]}}
        )

        self.assertEqual([["node-1", "node-2"], ["node-3", "node-4"]], pairs)

    def test_pairs_do_not_share_nodes(self):
        """A shared node would degrade several measurements at once."""
        pairs = self.module.generate_topological_pairs(
            {"block-a": {"sub-1": ["node-1", "node-2", "node-3", "node-4"]}}
        )

        used = [node for pair in pairs for node in pair]
        self.assertEqual(len(used), len(set(used)))

    def test_an_odd_group_still_measures_its_last_node(self):
        """At the cost of one node being measured twice."""
        pairs = self.module.generate_topological_pairs(
            {"block-a": {"sub-1": ["node-1", "node-2", "node-3"]}}
        )

        self.assertEqual([["node-1", "node-2"], ["node-2", "node-3"]], pairs)

    def test_a_subblock_of_one_produces_no_pair(self):
        pairs = self.module.generate_topological_pairs(
            {"block-a": {"sub-1": ["node-1"]}}
        )

        self.assertEqual([], pairs)

    def test_an_empty_subblock_contributes_nothing(self):
        """It has no node to stand for it in the cross-subblock pass."""
        pairs = self.module.generate_topological_pairs(
            {
                "block-a": {
                    "sub-0": [],
                    "sub-1": ["node-1", "node-2"],
                    "sub-2": ["node-3", "node-4"],
                }
            }
        )

        self.assertEqual(
            [
                ["node-1", "node-2"],
                ["node-3", "node-4"],
                ["node-1", "node-3"],
            ],
            pairs,
        )

    def test_an_empty_block_contributes_nothing(self):
        pairs = self.module.generate_topological_pairs(
            {
                "block-a": {},
                "block-b": {"sub-1": ["node-1", "node-2"]},
            }
        )

        self.assertEqual([["node-1", "node-2"]], pairs)

    def test_subblocks_are_linked_by_one_node_each(self):
        """The medium-distance pass, which covers links within a block."""
        pairs = self.module.generate_topological_pairs(TOPOLOGY)

        self.assertEqual(
            [
                ["node-1", "node-2"],
                ["node-3", "node-4"],
                ["node-1", "node-3"],
            ],
            pairs,
        )

    def test_blocks_are_linked_by_one_node_each(self):
        """The long-distance pass, which covers links between blocks."""
        pairs = self.module.generate_topological_pairs(
            {
                "block-a": {"sub-1": ["node-1", "node-2"]},
                "block-b": {"sub-1": ["node-3", "node-4"]},
            }
        )

        self.assertEqual(
            [
                ["node-1", "node-2"],
                ["node-3", "node-4"],
                ["node-1", "node-3"],
            ],
            pairs,
        )

    def test_an_empty_topology_produces_no_pairs(self):
        self.assertEqual([], self.module.generate_topological_pairs({}))

    def test_the_result_does_not_depend_on_dictionary_order(self):
        """Blocks and subblocks are sorted, so a run is reproducible."""
        forwards = self.module.generate_topological_pairs(
            {
                "block-a": {"sub-1": ["node-2", "node-1"]},
                "block-b": {"sub-1": ["node-4", "node-3"]},
            }
        )
        backwards = self.module.generate_topological_pairs(
            {
                "block-b": {"sub-1": ["node-3", "node-4"]},
                "block-a": {"sub-1": ["node-1", "node-2"]},
            }
        )

        self.assertEqual(forwards, backwards)


class PairwiseTopologyFileTest(unittest.TestCase):
    """Reading the topology file the CLI leaves in the workspace."""

    def setUp(self):
        self.module = _pairwise_module()
        self.root = tempfile.mkdtemp(prefix="chs-pairwise-")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def write(self, content):
        path = os.path.join(self.root, self.module.TOPOLOGY_FILENAME)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_a_written_topology_is_read_back(self):
        self.write(json.dumps(TOPOLOGY))

        self.assertEqual(TOPOLOGY, self.module.load_topology(self.root))

    def test_no_workspace_means_no_topology(self):
        self.assertIsNone(self.module.load_topology(None))

    def assert_warns_and_carries_on(self):
        """The warning is checked on a stub, not through the root logger.

        Several other test modules switch logging off globally and do not
        switch it back on, which would make assertLogs fail here depending on
        the order the suite runs in.
        """
        with mock.patch.object(self.module, "logger") as logger:
            self.assertIsNone(self.module.load_topology(self.root))

        logger.warning.assert_called_once()
        self.assertIn(
            "Pairing nodes arbitrarily", logger.warning.call_args[0][0]
        )

    def test_a_missing_file_is_reported_and_not_fatal(self):
        """The test still runs; it just cannot tell near pairs from far ones."""
        self.assert_warns_and_carries_on()

    def test_malformed_json_is_reported_and_not_fatal(self):
        self.write("{not json")

        self.assert_warns_and_carries_on()

    def test_an_unreadable_file_is_reported_and_not_fatal(self):
        self.write(json.dumps(TOPOLOGY))

        with mock.patch("builtins.open", side_effect=IOError("denied")):
            self.assert_warns_and_carries_on()


class PairwiseGroupingTest(unittest.TestCase):
    """Turning the topology into groups, names and expectations."""

    def setUp(self):
        self.plugin = plugin("nccl-pairwise")
        self.module = _pairwise_module()
        self.root = tempfile.mkdtemp(prefix="chs-pairwise-")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def with_topology(self):
        path = os.path.join(self.root, self.module.TOPOLOGY_FILENAME)
        with open(path, "w") as f:
            json.dump(TOPOLOGY, f)
        return self.root

    def test_one_node_is_too_few_to_measure_a_pair(self):
        self.assertEqual(2, self.plugin.minimum_nodes)
        self.assertEqual([], self.plugin.build(request(nodes=["node-1"])))

    def test_two_nodes_are_enough_for_one_pair(self):
        """A 2-node cluster has exactly one pair, and it is worth measuring."""
        specs = self.plugin.build(
            request(
                configs={
                    "machine-x": {
                        "variables": {},
                        "nccl-tests": {"variables": {}},
                    }
                }
            )
        )

        self.assertEqual(
            ["chs-run-nccl-pairwise-all-gather-node-1-node-2"],
            [spec.name for spec in specs if spec.workload == "all-gather"],
        )

    def test_the_topology_decides_the_pairs(self):
        grouping = self.plugin.grouping(
            request(
                nodes=["node-1", "node-2", "node-3", "node-4"],
                workspace_root=self.with_topology(),
            )
        )

        self.assertEqual(
            [
                ["node-1", "node-2"],
                ["node-3", "node-4"],
                ["node-1", "node-3"],
            ],
            grouping.groups,
        )
        self.assertEqual(
            {"node-1", "node-2", "node-3", "node-4"},
            set(grouping.node_locations),
        )

    def test_without_a_topology_the_allocation_is_one_flat_group(self):
        grouping = self.plugin.grouping(
            request(nodes=["node-1", "node-2", "node-3", "node-4"])
        )

        self.assertEqual(
            [["node-1", "node-2"], ["node-3", "node-4"]], grouping.groups
        )
        self.assertIsNone(grouping.node_locations)

    def test_an_experiment_is_named_after_the_pair_it_measures(self):
        self.assertEqual(
            "-node-1-node-2", self.plugin.group_suffix(["node-1", "node-2"])
        )

    def test_a_pair_in_one_subblock_is_judged_as_a_near_pair(self):
        grouping = self.plugin.grouping(
            request(
                nodes=["node-1", "node-2", "node-3", "node-4"],
                workspace_root=self.with_topology(),
            )
        )

        self.assertEqual(
            "intradomain",
            self.plugin.group_domain_type(["node-1", "node-2"], grouping),
        )

    def test_a_pair_spanning_subblocks_is_judged_as_a_far_pair(self):
        grouping = self.plugin.grouping(
            request(
                nodes=["node-1", "node-2", "node-3", "node-4"],
                workspace_root=self.with_topology(),
            )
        )

        self.assertEqual(
            "interdomain",
            self.plugin.group_domain_type(["node-1", "node-3"], grouping),
        )

    def test_an_unplaceable_pair_is_judged_as_a_far_pair(self):
        """A healthy link must not be called slow because of a missing map."""
        grouping = self.plugin.grouping(
            request(
                nodes=["node-1", "node-2", "node-3", "node-4"],
                workspace_root=self.with_topology(),
            )
        )

        self.assertEqual(
            "interdomain",
            self.plugin.group_domain_type(["node-1", "node-99"], grouping),
        )

    def test_without_a_topology_every_pair_is_judged_as_a_far_pair(self):
        grouping = self.plugin.grouping(request(nodes=["n1", "n2", "n3"]))

        self.assertEqual(
            "interdomain", self.plugin.group_domain_type(["n1", "n2"], grouping)
        )

    def test_a_group_that_is_not_a_pair_is_judged_as_a_far_pair(self):
        grouping = self.plugin.grouping(
            request(
                nodes=["node-1", "node-2", "node-3", "node-4"],
                workspace_root=self.with_topology(),
            )
        )

        self.assertEqual(
            "interdomain",
            self.plugin.group_domain_type(["node-1"], grouping),
        )


class SingleNodeNcclTest(unittest.TestCase):
    """The intra-node NCCL test."""

    def setUp(self):
        self.plugin = plugin("nccl-single-node")

    def test_every_node_is_measured_on_its_own(self):
        grouping = self.plugin.grouping(
            request(nodes=["node-1", "node-2", "node-3"])
        )

        self.assertEqual([["node-1"], ["node-2"], ["node-3"]], grouping.groups)

    def test_an_experiment_is_named_after_the_node_it_measures(self):
        self.assertEqual("-node-1", self.plugin.group_suffix(["node-1"]))

    def test_the_collective_is_kept_off_the_cluster_fabric(self):
        """Otherwise the intra-node measurement would include the fabric."""
        context = {}

        self.plugin.finalize_context(context, "all-gather", request())

        self.assertEqual("Socket", context["env_vars"]["set"]["NCCL_NET"])

    def test_existing_environment_variables_are_kept(self):
        context = {"env_vars": {"set": {"NCCL_DEBUG": "INFO"}}}

        self.plugin.finalize_context(context, "all-gather", request())

        self.assertEqual(
            {"NCCL_DEBUG": "INFO", "NCCL_NET": "Socket"},
            context["env_vars"]["set"],
        )


class TpuCollectivesTest(unittest.TestCase):
    """The multi-node TPU collectives test."""

    def setUp(self):
        self.plugin = plugin("tpu-collectives")

    def test_the_benchmark_input_follows_the_collective(self):
        variables = {}

        self.plugin.prepare_variables(variables, "all_reduce", request())

        self.assertEqual(
            "collectives_all_reduce.yaml", variables["benchmarks_yaml_file"]
        )

    def test_the_mesh_is_worked_out_once_the_slice_is_known(self):
        """The benchmark input ships as a template; the slice fills it in."""
        context = {
            "variables": {
                "execution_topology": "2x2x1",
                "cores_per_chip": 1,
                "benchmarks_yaml_escaped": "mesh: __MESH__ ici: __ICI__",
            }
        }

        self.plugin.finalize_context(context, "all_reduce", request())

        self.assertEqual(
            "mesh: 2x2x2 ici: 4",
            context["variables"]["benchmarks_yaml_escaped"],
        )

    def test_a_slice_of_unknown_shape_is_refused(self):
        """A guessed mesh would report bandwidth that means nothing."""
        context = {"variables": {"cores_per_chip": 1}}

        with self.assertRaises(ValueError):
            self.plugin.finalize_context(context, "all_reduce", request())

    def test_every_default_workload_builds_with_no_placeholders_left(self):
        req = request(
            slices={"slice-0": ["node-1", "node-2"]},
            tpu_topology="2x2x2",
            configs={
                "machine-x": {
                    "variables": {"cores_per_chip": 2},
                    "tpu-collectives": {"variables": {}},
                }
            },
        )

        specs = self.plugin.build(req)

        self.assertEqual(len(self.plugin.default_workloads), len(specs))
        for spec in specs:
            escaped = spec.context["variables"]["benchmarks_yaml_escaped"]
            self.assertTrue(escaped)
            for token in (
                "__MESH__",
                "__SHARDING__",
                "__ICI__",
                "__START__",
                "__END__",
                "__MULTIPLIER__",
            ):
                self.assertNotIn(token, escaped)


class TpuCollectivesSingleNodeTest(unittest.TestCase):
    """The same application, pinned to one node."""

    def setUp(self):
        self.plugin = plugin("tpu-collectives-single-node")

    def test_it_renders_the_multi_node_application(self):
        self.assertEqual("tpu-collectives", self.plugin.application)

    def test_its_settings_are_kept_apart_from_the_multi_node_ones(self):
        self.assertNotEqual(self.plugin.application, self.plugin.config_name)

    def test_the_collective_is_held_to_a_single_slice(self):
        """So the measurement never crosses the links between hosts."""
        variables = {}

        self.plugin.prepare_variables(variables, "all_reduce", request())

        self.assertEqual("2x2x1", variables["execution_topology"])
        self.assertEqual(
            "collectives_all_reduce.yaml", variables["benchmarks_yaml_file"]
        )

    def test_the_mesh_matches_the_single_host_topology(self):
        context = {
            "variables": {
                "execution_topology": "2x2x1",
                "cores_per_chip": 1,
                "benchmarks_yaml_escaped": "mesh: __MESH__",
            }
        }

        self.plugin.finalize_context(context, "all_gather", request())

        self.assertEqual(
            "mesh: 2x2x2", context["variables"]["benchmarks_yaml_escaped"]
        )


class GpuAndTpuPackageDeclarationsTest(unittest.TestCase):
    """Package-specific declarations on the remaining GPU and TPU plugins."""

    def _configs(self, app_key):
        return {
            "machine-x": {
                "variables": {"cores_per_chip": 2},
                app_key: {"variables": {"timeout": 600, "deadline": 600}},
            }
        }

    def test_dcgm_fans_out_over_node_and_diagnostic_level(self):
        p = plugin("dcgm")
        req = request(configs=self._configs(p.config_name))

        specs = p.build(req)

        self.assertEqual(1, len(specs))
        self.assertEqual(
            "chs-run-dcgm-level-{diag_level}-{node}", specs[0].name
        )
        self.assertEqual(["node", "diag_level"], specs[0].context["matrix"])

    def test_gemm_keeps_its_historical_experiment_label(self):
        p = plugin("gemm")
        req = request(configs=self._configs(p.config_name))

        specs = p.build(req)

        self.assertEqual(1, len(specs))
        self.assertEqual("chs-run-nvidia-gemm-check-{node}", specs[0].name)

    def test_stream_binds_one_gpu_per_rank_on_slurm(self):
        p = plugin("stream")
        req = request(
            scheduler="slurm",
            timeout=600,
            deadline=600,
            configs=self._configs(p.config_name),
        )

        specs = p.build(req)

        self.assertEqual(1, len(specs))
        self.assertIn(
            "#SBATCH --gpu-bind=single:1",
            specs[0].context["variables"]["extra_sbatch_headers"],
        )

    def test_nccl_cluster_skips_below_three_nodes_and_runs_at_three(self):
        p = plugin("nccl-cluster")
        two_nodes = request(
            nodes=["node-1", "node-2"], configs=self._configs(p.config_name)
        )
        self.assertEqual([], p.build(two_nodes))

        three_nodes = request(
            nodes=["node-1", "node-2", "node-3"],
            configs=self._configs(p.config_name),
        )
        specs = p.build(three_nodes)
        self.assertEqual(2, len(specs))
        self.assertEqual(3, specs[0].context["variables"]["n_nodes"])

    def test_tpu_packages_with_benchmark_inputs_inline_them(self):
        for test_name, expected_file in (
            ("tpu-gemm", "gemm_16384_bf16.yaml"),
            ("tpu-hbm", "hbm_copy_bf16.yaml"),
            ("tpu-host-device", "host_device.yaml"),
        ):
            p = plugin(test_name)
            req = request(
                slices={"slice-0": ["node-1"]},
                configs=self._configs(p.config_name),
            )
            with self.subTest(test=test_name):
                self.assertEqual(expected_file, p.benchmark_config)
                specs = p.build(req)
                self.assertEqual(1, len(specs))
                self.assertEqual(
                    expected_file,
                    specs[0].context["variables"]["benchmarks_yaml_file"],
                )
                self.assertTrue(
                    specs[0].context["variables"]["benchmarks_yaml_escaped"]
                )


def _pairwise_module():
    """Returns the pairwise plugin's module, for its module-level helpers."""
    return sys.modules[type(plugin("nccl-pairwise")).__module__]
