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

import sys
import time
import logging
import threading
from contextlib import contextmanager
from rich import box
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
from chs.utils.cluster import is_in_cluster

CHS_HEADER_STYLE = "bold cyan"
CHS_WARNING_STYLE = "yellow"
CHS_TITLE_STYLE = "bold"
CHS_SUCCESS_STYLE = "green"
CHS_FAILURE_STYLE = "red"
CHS_NEUTRAL_STYLE = "dim"


def create_styled_table(
    *columns: str,
    header_style: str = CHS_HEADER_STYLE,
    expand: bool = True,
    show_lines: bool = False,
) -> Table:
    """Creates a Table with CHS standard styling.

    If columns are provided, they are added to the table.
    If no columns are provided, an empty styled table is returned,
    allowing caller to add columns with custom styles.
    """
    table = Table(
        box=box.SIMPLE,
        show_lines=show_lines,
        header_style=header_style,
        expand=expand,
    )
    for col in columns:
        table.add_column(col, overflow="fold")
    return table


class LogSpinner:
    """A progress logger for non-interactive environments (e.g. cluster pods).

    Instead of rendering a rich spinner, it logs the progress description
    at a steady interval using a background thread.
    """

    def __init__(self, description: str, show_time: bool = True):
        self.logger = logging.getLogger("chs.theme.progress")
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.propagate = False

        self.show_time = show_time
        self.start_time = time.time()
        self.clean_description = self._format_description(description)

        self.logger.info(self.clean_description)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat)
        self._thread.daemon = True
        self._thread.start()

    def _format_description(self, description: str) -> str:
        return Text.from_markup(description).plain

    def _heartbeat(self):
        interval = 10.0 if self.show_time else 30.0
        while not self._stop_event.wait(interval):
            if self.show_time:
                elapsed = int(time.time() - self.start_time)
                self.logger.info(
                    f"{self.clean_description} (Elapsed: {elapsed}s)"
                )
            else:
                self.logger.info(self.clean_description)

    def update(
        self, task_id, description: str, force_print: bool = False, **kwargs
    ):
        self.clean_description = self._format_description(description)
        if force_print:
            self.logger.info(self.clean_description)

    def stop(self):
        self._stop_event.set()


@contextmanager
def get_progress_spinner(description: str, show_time: bool = True):
    """Context manager for a standard CHS progress spinner with elapsed time."""
    if not sys.stdout.isatty() or is_in_cluster():
        log_spinner = LogSpinner(description, show_time=show_time)
        try:
            yield log_spinner, "log_task"
        finally:
            log_spinner.stop()
            elapsed = int(time.time() - log_spinner.start_time)
            desc = log_spinner.clean_description.splitlines()[0]
            log_spinner.logger.info(f"Finished: {desc} (Elapsed: {elapsed}s)")
        return

    columns = [
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
    ]
    if show_time:
        columns.append(TimeElapsedColumn())

    with Progress(*columns, transient=True) as progress:
        task_id = progress.add_task(description=description, total=None)
        yield progress, task_id
