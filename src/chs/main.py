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

"""Entry point for the Cluster Health Scanner (CHS) command-line interface.

This script initializes logging and delegates argument parsing and command
execution to the CHS application library.
"""

import logging
import sys

from rich.logging import RichHandler

from chs import cli
from chs import telemetry
from chs.telemetry import reporting as telemetry_reporting
from chs.utils.app_data import ChsAppData


def _normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    new_argv = [argv[0]]
    for token in argv[1:]:
        if token.startswith("-"):
            new_argv.append(token)
            continue
        previous = new_argv[-1]
        if not previous.startswith("-") and (
            previous.endswith(",") or token.startswith(",")
        ):
            new_argv[-1] = previous + token
        else:
            new_argv.append(token)
    return new_argv


def main():
    normalized_argv = _normalize_argv(sys.argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    args = normalized_argv[1:]
    if telemetry_reporting.is_command_supported(args) and (
        ChsAppData().is_telemetry_enabled()
    ):
        telemetry.enable()
    cli.main(args=args)


if __name__ == "__main__":
    main()
