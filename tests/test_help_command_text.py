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

import click
from unittest import mock

from chs.cli import help_command_text


@mock.patch("chs.utils.suites.get_all_suites_details")
@mock.patch("chs.utils.tests.get_all_supported_machine_types")
@mock.patch("chs.utils.tests.get_tests_by_accelerator_type")
def test_generate_run_epilog(
    mock_get_tests, mock_get_machine_types, mock_get_suites
):
    mock_get_suites.return_value = [{"name": "suite_1"}, {"name": "suite_2"}]
    mock_get_machine_types.return_value = ["a3-highgpu-8g", "a2-highgpu-1g"]

    def get_tests_side_effect(accel_type):
        if accel_type == "gpu":
            return ["dummy_gpu_test1", "dummy_gpu_test2"]
        elif accel_type == "tpu":
            return ["dummy_tpu_test1"]
        return []

    mock_get_tests.side_effect = get_tests_side_effect

    epilog = help_command_text.generate_run_epilog()

    assert "Supported Suites:" in epilog
    assert "suite_1, suite_2" in epilog
    assert "Supported Tests:" in epilog
    assert "GPU: dummy_gpu_test1, dummy_gpu_test2" in epilog
    assert "TPU: dummy_tpu_test1" in epilog
    assert "Supported Machine Types:" in epilog
    assert "a3-highgpu-8g, a2-highgpu-1g" in epilog


@mock.patch("chs.utils.suites.get_all_suites_details")
def test_get_suite_help(mock_get_suites):
    # Test with multiple suites
    mock_get_suites.return_value = [
        {"name": "lite"},
        {"name": "standard"},
        {"name": "comprehensive"},
    ]
    help_text = help_command_text.get_suite_help()
    assert "Select one of: lite, standard, or comprehensive." in help_text

    # Test with two suites
    mock_get_suites.return_value = [{"name": "lite"}, {"name": "standard"}]
    help_text = help_command_text.get_suite_help()
    assert "Select one of: lite or standard." in help_text

    # Test with empty suites
    mock_get_suites.return_value = []
    help_text = help_command_text.get_suite_help()
    assert "No suites available." in help_text


@mock.patch("chs.utils.tests.get_tests_by_accelerator_type")
def test_get_tests_help(mock_get_tests):
    def get_tests_side_effect(accel_type):
        if accel_type == "gpu":
            return ["dummy_gpu_test"]
        elif accel_type == "tpu":
            return ["dummy_tpu_test"]
        return []

    mock_get_tests.side_effect = get_tests_side_effect

    help_text = help_command_text.get_tests_help()
    assert "Supported GPU benchmarks:\ndummy_gpu_test" in help_text
    assert "Supported TPU benchmarks:\ndummy_tpu_test" in help_text


def test_style_run_docstring():
    def dummy_func():
        """This is a test Note: testing."""
        pass

    styled_func = help_command_text.style_run_docstring(dummy_func)
    # The note should be styled with click.style
    expected = (
        "This is a test "
        + click.style("Note:", fg="yellow", bold=True)
        + " testing."
    )
    assert styled_func.__doc__ == expected

    # Test func without docstring
    def empty_func():
        pass

    styled_empty_func = help_command_text.style_run_docstring(empty_func)
    assert styled_empty_func.__doc__ is None
