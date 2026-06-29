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

from collections import defaultdict
from typing import Any

from rich.console import Group
from rich.text import Text

from chs import utils
from chs.utils import theme
from chs.utils.suites import load_suite


def get_suites_renderable() -> Any:
    """Returns renderable for available suites."""
    suites_list = utils.suites.get_all_suites_details()
    if suites_list:
        table = theme.create_styled_table(
            "Name", "Est. Time", "Description", expand=False
        )
        for s in suites_list:
            name = s.get("name", "")
            est_time = s.get("est_time", "N/A")
            desc = s.get("description", "")
            table.add_row(name, est_time, desc)

        return Group(Text("Available Test Suites:", style="bold cyan"), table)
    return "No suites found."


def get_tests_renderables(
    suite: str | None,
    tests_flag: bool,
    machine_type: str | None = None,
    accelerator_type: str | None = None,
) -> list[Any]:
    """Returns renderables for available tests."""
    tests_details = utils.tests.get_all_supported_tests_details()

    if suite:
        suite_cfg = load_suite(suite)
        test_names = [
            t.get("name") if isinstance(t, dict) else t
            for t in suite_cfg.get("tests", [])
        ]
        if machine_type:
            supported_tests = utils.tests.get_tests_by_machine_type(
                machine_type
            )
            test_names = [t for t in test_names if t in supported_tests]
        elif accelerator_type:
            supported_tests = utils.tests.get_tests_by_accelerator_type(
                accelerator_type
            )
            test_names = [t for t in test_names if t in supported_tests]
    elif machine_type:
        test_names = utils.tests.get_tests_by_machine_type(machine_type)
    elif accelerator_type:
        test_names = utils.tests.get_tests_by_accelerator_type(accelerator_type)
    else:
        test_names = utils.tests.get_all_supported_tests()

    if test_names:
        renderables = []
        if suite:
            renderables.append(
                Text(
                    f"Available Tests for suite '{suite}':",
                    style="bold cyan",
                )
            )
        else:
            renderables.append(Text("Available Tests:", style="bold cyan"))

        if tests_flag:
            hints = []
            if not suite:
                hints.append("--suite <suite_name>")
            if not machine_type and not accelerator_type:
                hints.append("--machine-type <machine_type>")
            if hints:
                renderables.append(
                    Text(
                        f"  (Hint: Use {' or '.join(hints)} to filter tests)",
                        style="dim",
                    )
                )

        if machine_type:
            renderables.append(
                Text(
                    f"  (Filtered by machine type: {machine_type})", style="dim"
                )
            )
        elif accelerator_type:
            renderables.append(
                Text(
                    f"  (Filtered by accelerator: {accelerator_type.upper()})",
                    style="dim",
                )
            )

        filtered_tests = [
            t for t in tests_details if t.get("name") in test_names
        ]

        grouped = defaultdict(list)
        for t in filtered_tests:
            cat = t.get("category", "Other")
            grouped[cat].append(t)

        for cat in sorted(grouped.keys()):
            renderables.append(Text(f"\n{cat}", style="bold"))
            table = theme.create_styled_table(
                "Name", "Description", expand=False
            )
            for t in sorted(grouped[cat], key=lambda x: x.get("name", "")):
                name = t.get("name", "")
                desc = t.get("description", "")
                table.add_row(name, desc)
            renderables.append(table)

        return renderables
    return ["No tests found."]


def get_schedulers_renderable() -> Any:
    """Returns renderable for available schedulers."""
    schedulers_list = utils.tests.get_all_supported_schedulers()
    if schedulers_list:
        table = theme.create_styled_table("Name", expand=False)
        for s in schedulers_list:
            table.add_row(s)
        return Group(Text("Supported Schedulers:", style="bold cyan"), table)
    return "No schedulers found."


def get_machine_types_renderable() -> Any:
    """Returns renderable for available machine types."""
    machine_types_list = utils.tests.get_all_supported_machine_types()
    if machine_types_list:
        table = theme.create_styled_table("Name", expand=False)
        for m in machine_types_list:
            table.add_row(m)
        return Group(Text("Supported Machine Types:", style="bold cyan"), table)
    return "No machine types found."
