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

from chs import utils


def generate_run_epilog() -> str:
    """Generates the epilog with available tests, suites, and machine types."""
    suites_list = utils.suites.get_all_suites_details()
    suites = ", ".join([s.get("name") for s in suites_list if s.get("name")])
    machine_types = ", ".join(utils.tests.get_all_supported_machine_types())
    gpu_tests = ", ".join(utils.tests.get_tests_by_accelerator_type("gpu"))
    tpu_tests = ", ".join(utils.tests.get_tests_by_accelerator_type("tpu"))

    suites_header = click.style("Supported Suites:", fg="cyan", bold=True)
    tests_header = click.style("Supported Tests:", fg="cyan", bold=True)
    machine_types_header = click.style(
        "Supported Machine Types:", fg="cyan", bold=True
    )
    note_header = click.style("Note:", fg="yellow", bold=True)

    return (
        "\b\n"
        f"{suites_header}\n"
        f"  {suites}\n"
        " \n"
        f"{tests_header}\n"
        f"  GPU: {gpu_tests}\n"
        f"  TPU: {tpu_tests}\n"
        " \n"
        f"{machine_types_header}\n"
        f"  {machine_types}\n"
        " \n"
        f"{note_header}\n"
        "  When selecting targets, ensure that the given nodes, nodepools, or\n"
        "  partitions consist of one of the Supported Machine Types "
        "listed above.\n"
        "  Mixed hardware targets within a single run are not currently "
        "supported."
    )


def get_suite_help() -> str:
    suites_list = utils.suites.get_all_suites_details()
    suites_names = [s.get("name") for s in suites_list if s.get("name")]
    if not suites_names:
        return "No suites available."

    if len(suites_names) > 2:
        suites = ", ".join(suites_names[:-1]) + ", or " + suites_names[-1]
    else:
        suites = " or ".join(suites_names)

    return (
        "Name of the pre-configured test suite to run. A suite is a package "
        "of tests pre-configured for specific use-cases (e.g., a quick "
        "health check vs a standard validation).\n\n"
        f"Select one of: {suites}.\n\n"
        "If you don't specify either a suite or specific tests, the 'lite' "
        "suite is run by default."
        "\n\n "
    )


def get_tests_help() -> str:
    gpu = ", ".join(utils.tests.get_tests_by_accelerator_type("gpu"))
    tpu = ", ".join(utils.tests.get_tests_by_accelerator_type("tpu"))
    return (
        "Comma-separated list (or multiple flags) of individual hardware or "
        "network benchmarks "
        "to run. Use this flag if you want to selectively run specific "
        "workloads rather than a full suite.\n\n"
        f"Supported GPU benchmarks:\n{gpu}\n\n"
        f"Supported TPU benchmarks:\n{tpu}"
        "\n\n "
    )


def style_run_docstring(func):
    """Decorator to style 'Note:' in the command docstring for help."""
    if func.__doc__:
        func.__doc__ = func.__doc__.replace(
            "Note:", click.style("Note:", fg="yellow", bold=True)
        )
    return func


def style_cli_docstring(func):
    """Decorator to style 'TL;DR:' in the cli docstring for help."""
    if func.__doc__:
        func.__doc__ = func.__doc__.replace(
            "TL;DR:", click.style("TL;DR:", fg="yellow", bold=True)
        )
    return func
