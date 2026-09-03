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

"""Unit tests for IntegrationValidator and composable pure function checkers."""

import glob
import os
import unittest
import yaml

from tests.integration.test_utils.validator import (
    IntegrationValidator,
    ValidationContext,
)


class TestIntegrationValidator(unittest.TestCase):
    def test_validation_context_defaults(self):
        ctx = ValidationContext(
            nodepool="pool-1", node_index=0, test_name="tpu-info"
        )
        self.assertEqual(ctx.nodepool, "pool-1")
        self.assertEqual(ctx.node_index, 0)
        self.assertEqual(ctx.actual_outcome, "PASSED")
        self.assertEqual(ctx.metrics, {})
        self.assertEqual(ctx.logs, "")
        self.assertEqual(ctx.stderr, "")
        self.assertIsNone(ctx.error_code)

    def test_outcome_checker(self):
        validator = IntegrationValidator()
        ctx_pass = ValidationContext(actual_outcome="PASSED")
        ctx_fail = ValidationContext(actual_outcome="FAILED")

        report1 = validator.validate(ctx_pass, {"outcome": "PASSED"})
        self.assertTrue(report1.passed)
        self.assertEqual(report1.results[0].check_name, "outcome")

        report2 = validator.validate(ctx_fail, {"outcome": "PASSED"})
        self.assertFalse(report2.passed)
        self.assertEqual(report2.results[0].expected, "PASSED")
        self.assertEqual(report2.results[0].actual, "FAILED")

    def test_unsuccessful_status_not_passed(self):
        ctx = ValidationContext.from_boh_result(
            boh_match={"status": "UNSUCCESSFUL"},
            target_scenario={"test": "tpu-info"},
        )
        self.assertEqual(ctx.actual_outcome, "UNSUCCESSFUL")
        self.assertNotEqual(ctx.actual_outcome, "PASSED")

    def test_none_error_code(self):
        ctx = ValidationContext.from_boh_result(
            boh_match={
                "status": "PASSED",
                "error_code": None,
                "metadata": None,
            },
            target_scenario={"test": "tpu-info"},
        )
        self.assertIsNone(ctx.error_code)

    def test_regex_error_code_safety(self):
        validator = IntegrationValidator()
        ctx = ValidationContext(error_code="ERR[404] (INVALID_CONFIG)")
        report_literal = validator.validate(
            ctx, {"error_code": "ERR[404] (INVALID_CONFIG)"}
        )
        self.assertTrue(report_literal.passed)

        report_regex = validator.validate(ctx, {"error_code": r"re:ERR\[\d+\]"})
        self.assertTrue(report_regex.passed)

    def test_index_metadata_key(self):
        validator = IntegrationValidator()
        ctx = ValidationContext(
            nodepool="pool-1",
            node_index=0,
            test_name="tpu-info",
            actual_outcome="PASSED",
        )
        scenario = {
            "nodepool": "pool-1",
            "index": 0,
            "test": "tpu-info",
            "outcome": "PASSED",
        }
        report = validator.validate(ctx, scenario)
        self.assertTrue(report.passed)

    def test_metrics_checker_exact_and_tolerance(self):
        validator = IntegrationValidator()
        ctx = ValidationContext(
            metrics={"Bandwidth (median)": 1200.0, "TFLOPS": 34.0, "count": 4}
        )

        report = validator.validate(
            ctx,
            {
                "metrics": {
                    "Bandwidth (median)": 1200.0,
                    "TFLOPS": 34.0,
                    "count": 4,
                }
            },
        )
        self.assertTrue(report.passed)

        # Missing metric test
        report_missing = validator.validate(
            ctx, {"metrics": {"missing_key": 10}}
        )
        self.assertFalse(report_missing.passed)
        self.assertIn(
            "Missing metric 'missing_key'", report_missing.results[0].message
        )

    def test_metrics_checker_expressions(self):
        validator = IntegrationValidator()
        ctx = ValidationContext(
            metrics={
                "iops": 1000.0,
                "latency": 0.5,
                "temp": -10.0,
                "bandwidth": 1200.0,
                "err_rate": 0.0,
            }
        )

        spec = {
            "metrics": {
                "iops": 1000.0,
                "latency": 0.5,
                "temp": -10.0,
                "bandwidth": 1200.0,
                "err_rate": 0.0,
            }
        }
        report = validator.validate(ctx, spec)
        self.assertTrue(report.passed)

    def test_error_code_and_stderr_checkers(self):
        validator = IntegrationValidator()
        ctx = ValidationContext(
            actual_outcome="FAILED",
            error_code="TPU_CHIP_INITIALIZATION_ERROR",
            stderr=(
                "RuntimeError: Unable to initialize TPU device 3: "
                "HbmTimeoutError"
            ),
        )

        spec = {
            "outcome": "FAILED",
            "error_code": "TPU_CHIP_INITIALIZATION_ERROR",
            "stderr": "Unable to initialize TPU device",
        }

        report = validator.validate(ctx, spec)
        self.assertTrue(report.passed)

    def test_logs_contain_and_exclude(self):
        validator = IntegrationValidator()
        ctx = ValidationContext(
            logs="HOST node-0: JAX Devices: 4\nHost Info: Linux x86_64"
        )

        spec_pass = {
            "logs_contain": ["JAX Devices: 4", "Linux x86_64"],
            "logs_exclude": ["Fatal Error", "Out of Memory"],
        }
        report_pass = validator.validate(ctx, spec_pass)
        self.assertTrue(report_pass.passed)

        spec_fail = {"logs_contain": ["Missing pattern"]}
        report_fail = validator.validate(ctx, spec_fail)
        self.assertFalse(report_fail.passed)

    def test_validate_all_blueprint_configs(self):
        """Validates all blueprint YAML configs parse and validate."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_pattern = os.path.join(base_dir, "integration/configs/**/*.yaml")
        config_paths = glob.glob(config_pattern, recursive=True)
        self.assertGreater(len(config_paths), 0)

        validator = IntegrationValidator()

        for config_path in config_paths:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}

            for scenario in data.get("test_scenarios", []):
                if not isinstance(scenario, dict):
                    continue

                ctx = ValidationContext(
                    nodepool=scenario.get("nodepool", ""),
                    node_index=scenario.get("node_index", 0),
                    test_name=scenario.get("test", ""),
                    actual_outcome=scenario.get("outcome", "PASSED"),
                    metrics=scenario.get("metrics", {}),
                    error_code=scenario.get("error_code"),
                    stderr=scenario.get("stderr", ""),
                    degraded_links=scenario.get("degraded_links", []),
                )

                report = validator.validate(ctx, scenario)
                summary = report.format_summary()
                self.assertTrue(
                    report.passed,
                    f"Failed scenario in {config_path}:\n{summary}",
                )

    def test_validate_boh_and_output_success_and_failure(self):
        validator = IntegrationValidator()

        blueprint_config = {
            "nodepools": [{"name": "tpu-v6e-pool-01", "count": 2}],
            "test_scenarios": [
                {
                    "nodepool": "tpu-v6e-pool-01",
                    "node_index": 0,
                    "test": "tpu-hbm",
                    "outcome": "DEGRADED",
                    "metrics": {"Bandwidth (median)": 34.0},
                }
            ],
        }

        mock_boh = {
            "results": [
                {
                    "test_name": "tpu-hbm",
                    "status": "DEGRADED",
                    "metrics": {"Bandwidth (median)": 34.0},
                }
            ]
        }

        # 1. Test clean execution matching boh
        report_ok = validator.validate_boh_and_output(
            blueprint_config,
            mock_boh,
            exit_code=0,
            stdout="Bill of Health generated successfully.",
        )
        self.assertTrue(report_ok.passed)

        # 2. Test execution with string nodepools
        blueprint_str_pools = {
            "nodepools": ["tpu-v6e-pool-01"],
            "test_scenarios": [
                {
                    "nodepool": "tpu-v6e-pool-01",
                    "node_index": 0,
                    "test": "tpu-hbm",
                    "outcome": "DEGRADED",
                    "metrics": {"Bandwidth (median)": 34.0},
                }
            ],
        }
        report_str_pools = validator.validate_boh_and_output(
            blueprint_str_pools,
            mock_boh,
            exit_code=0,
            stdout="Bill of Health generated successfully.",
        )
        self.assertTrue(report_str_pools.passed)

        # 3. Test execution with unexpected non-zero exit code
        report_err_code = validator.validate_boh_and_output(
            blueprint_config, mock_boh, exit_code=1, stderr="Fatal error"
        )
        self.assertFalse(report_err_code.passed)

        # 4. Test execution with missing boh
        report_no_boh = validator.validate_boh_and_output(
            blueprint_config, boh=None, exit_code=0, stdout="Success"
        )
        self.assertFalse(report_no_boh.passed)

    def test_from_boh_result_does_not_use_scenario_fallbacks(self):
        """Verifies missing/unknown status not auto-populated."""
        ctx = ValidationContext.from_boh_result(
            boh_match={"status": "UNKNOWN", "metrics": []},
            target_scenario={
                "outcome": "PASSED",
                "error_code": "ERR_SPEC",
                "stderr": "expected err",
            },
        )
        self.assertEqual(ctx.actual_outcome, "UNKNOWN")
        self.assertIsNone(ctx.error_code)
        self.assertEqual(ctx.stderr, "")

        # When evaluated, UNKNOWN actual vs PASSED expected outcome must fail
        validator = IntegrationValidator()
        report = validator.validate(ctx, {"outcome": "PASSED"})
        self.assertFalse(report.passed)

    def test_invalid_regex_error_code_returns_failure(self):
        """Verifies invalid regex in error_code returns failed CheckResult."""
        validator = IntegrationValidator()
        ctx = ValidationContext(error_code="SOME_ERROR")
        report = validator.validate(ctx, {"error_code": "re:ERR[*"})
        self.assertFalse(report.passed)
        self.assertIn("Invalid regex pattern", report.results[0].message)

    def test_degraded_links_validation(self):
        """Verifies that degraded_links checker evaluates link expectations."""
        validator = IntegrationValidator()
        ctx = ValidationContext(
            actual_outcome="PASSED",
            degraded_links=[
                {
                    "source": {"nodepool": "p1", "node_index": 0},
                    "target": {"nodepool": "p1", "node_index": 1},
                    "metrics": {"Bandwidth": 10.0},
                }
            ],
        )
        report = validator.validate(
            ctx,
            {
                "degraded_links": [
                    {
                        "source": {"nodepool": "p1", "node_index": 0},
                        "target": {"nodepool": "p1", "node_index": 1},
                        "metrics": {"Bandwidth": 15.0},
                    }
                ]
            },
        )
        self.assertTrue(report.passed)

    def test_metrics_checker_threshold_bounds(self):
        """Verifies check_metrics evaluates min, max, and exact bounds."""
        validator = IntegrationValidator()
        ctx_ok = ValidationContext(
            metrics={"bandwidth": 1200.0, "latency": 1.2}
        )
        report_ok = validator.validate(
            ctx_ok,
            {
                "metrics": {
                    "bandwidth": {"min": 1100.0},
                    "latency": {"max": 2.0},
                }
            },
        )
        self.assertTrue(report_ok.passed)

        ctx_fail = ValidationContext(metrics={"bandwidth": 900.0})
        report_fail = validator.validate(
            ctx_fail, {"metrics": {"bandwidth": {"min": 1000.0}}}
        )
        self.assertFalse(report_fail.passed)
        self.assertIn("expected >= 1000.0", report_fail.results[0].message)

    def test_default_thresholds_resolution(self):
        """Verifies automatic fallback to default thresholds."""
        custom_thresholds = {
            "ct6e-standard-4t": {
                "tpu-hbm": {
                    "type": "min",
                    "thresholds": {"bandwidth": 1100.0},
                }
            }
        }
        validator = IntegrationValidator(thresholds_data=custom_thresholds)
        resolved = validator.resolve_default_thresholds(
            "ct6e-standard-4t", "tpu-hbm"
        )
        self.assertEqual(resolved, {"bandwidth": {"min": 1100.0}})

        blueprint_no_metrics = {
            "nodepools": [
                {"name": "tpu-v6e-pool-01", "machine_type": "ct6e-standard-4t"}
            ],
            "test_scenarios": [
                {
                    "nodepool": "tpu-v6e-pool-01",
                    "node_index": 0,
                    "test": "tpu-hbm",
                    "outcome": "PASSED",
                }
            ],
        }

        mock_boh_pass = {
            "results": [
                {
                    "nodepool": "tpu-v6e-pool-01",
                    "node_index": 0,
                    "test_name": "tpu-hbm",
                    "status": "PASSED",
                    "metrics": [{"key": "bandwidth", "value": 1200.0}],
                }
            ]
        }
        report = validator.validate_boh_and_output(
            blueprint_no_metrics, mock_boh_pass, exit_code=0
        )
        self.assertTrue(report.passed)

        mock_boh_fail = {
            "results": [
                {
                    "nodepool": "tpu-v6e-pool-01",
                    "node_index": 0,
                    "test_name": "tpu-hbm",
                    "status": "PASSED",
                    "metrics": [{"key": "bandwidth", "value": 800.0}],
                }
            ]
        }
        report_fail = validator.validate_boh_and_output(
            blueprint_no_metrics, mock_boh_fail, exit_code=0
        )
        self.assertFalse(report_fail.passed)

    def test_falsy_status_0_not_evaluated_as_passed(self):
        """Verifies integer 0 or falsy status is not converted to PASSED."""
        ctx = ValidationContext.from_boh_result(
            boh_match={"status": 0, "test": "nccl-test"},
            target_scenario={"test": "nccl-test"},
        )
        self.assertNotEqual(ctx.actual_outcome, "PASSED")

        validator = IntegrationValidator()
        mock_boh = {"results": [{"status": 0, "test_name": "nccl-test"}]}
        report = validator.validate_boh_and_output({}, mock_boh, exit_code=0)
        self.assertFalse(report.passed)

    def test_empty_expected_code_does_not_match_non_empty_actual(self):
        """Verifies empty string expected_code fails when actual is present."""
        validator = IntegrationValidator()
        ctx = ValidationContext(error_code="ERR_OOM")
        report = validator.validate(ctx, {"error_code": ""})
        self.assertFalse(report.passed)

    def test_check_stderr_fails_when_pattern_missing_even_with_error_code(self):
        """Verifies stderr check is not bypassed when error_code is present."""
        validator = IntegrationValidator()
        ctx = ValidationContext(
            error_code="SOME_ERROR",
            stderr="A completely different error message",
        )
        report = validator.validate(
            ctx, {"stderr": "ExpectedNonExistentPattern"}
        )
        self.assertFalse(report.passed)
        self.assertIn("not found in stderr/logs", report.results[0].message)

    def test_check_metrics_non_numeric_threshold_bounds(self):
        """Verifies non-numeric min/max thresholds fail cleanly."""
        validator = IntegrationValidator()
        ctx = ValidationContext(metrics={"bandwidth": 100.0})
        report_min = validator.validate(
            ctx, {"metrics": {"bandwidth": {"min": "not-a-number"}}}
        )
        self.assertFalse(report_min.passed)
        self.assertIn(
            "non-numeric min threshold", report_min.results[0].message
        )

        report_max = validator.validate(
            ctx, {"metrics": {"bandwidth": {"max": "not-a-number"}}}
        )
        self.assertFalse(report_max.passed)
        self.assertIn(
            "non-numeric max threshold", report_max.results[0].message
        )

    def test_from_boh_result_captures_error_metric(self):
        """Verifies error metrics in BOH are captured in ctx.stderr/logs."""
        boh_match = {
            "test_name": "tpu-info",
            "status": "FAILED",
            "metrics": [
                {
                    "key": "TPU Hardware / Driver / Network Error Detected",
                    "value": "HOST node-4: RuntimeError: ICI link failure",
                }
            ],
        }
        ctx = ValidationContext.from_boh_result(
            boh_match=boh_match,
            target_scenario={"test": "tpu-info"},
        )
        self.assertIn("RuntimeError: ICI link failure", ctx.stderr)
        self.assertIn("RuntimeError: ICI link failure", ctx.logs)

        validator = IntegrationValidator()
        report = validator.validate(
            ctx, {"stderr": "RuntimeError: ICI link failure"}
        )
        self.assertTrue(report.passed)

    def test_multi_workload_candidate_selection(self):
        """Verifies multi-workload test selects degraded candidate when
        expecting DEGRADED.
        """
        validator = IntegrationValidator()
        blueprint_config = {
            "nodepools": [{"name": "gpu-a4x-pool-01", "count": 2}],
            "test_scenarios": [
                {
                    "test": "nccl-pairwise",
                    "outcome": "DEGRADED",
                    "degraded_links": [
                        {
                            "source": {
                                "nodepool": "gpu-a4x-pool-01",
                                "node_index": 0,
                            },
                            "target": {
                                "nodepool": "gpu-a4x-pool-01",
                                "node_index": 1,
                            },
                            "metrics": {"Bandwidth": 200.0},
                        }
                    ],
                }
            ],
        }
        mock_boh = {
            "results": [
                {
                    "test_name": "nccl-pairwise",
                    "status": "PASSED",
                    "target_resources": [
                        {"node_name": "node-1"},
                        {"node_name": "node-2"},
                    ],
                    "metrics": [{"key": "Avg. Bus Bandwidth", "value": 5500.0}],
                },
                {
                    "test_name": "nccl-pairwise",
                    "status": "FAILED",
                    "target_resources": [
                        {"node_name": "node-1"},
                        {"node_name": "node-2"},
                    ],
                    "metrics": [{"key": "Avg. Bus Bandwidth", "value": 200.0}],
                },
            ]
        }
        report = validator.validate_boh_and_output(
            blueprint_config, mock_boh, exit_code=0
        )
        self.assertTrue(
            report.passed, f"Report failed: {report.format_summary()}"
        )


if __name__ == "__main__":
    unittest.main()
