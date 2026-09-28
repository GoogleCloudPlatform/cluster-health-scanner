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

import logging
import subprocess
import threading

from chs import RambleManager
from chs import exceptions as ex
from chs.utils import theme

logger = logging.getLogger(__name__)


class BackgroundCleanup:
    """Context manager that runs `RambleManager.cancel_experiments` on a
    background thread and waits for it to finish on exit.
    """

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def __enter__(self) -> "BackgroundCleanup":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._thread is not None:
            self._wait()
            self._handle_outcome(body_failed=exc_type is not None)
        return False

    def start(self, ramble_manager: RambleManager) -> None:
        """Starts cleaning up scheduler resources in the background."""
        if self._thread is not None:
            return
        scheduler_name = ramble_manager.get_scheduler_type()
        thread = threading.Thread(
            target=self._cancel_experiments,
            args=(ramble_manager, scheduler_name),
            name="chs-cleanup",
            daemon=True,
        )
        self._thread = thread
        try:
            thread.start()
        except BaseException:
            self._thread = None
            raise

    def _cancel_experiments(
        self, ramble_manager: RambleManager, scheduler_name: str
    ) -> None:
        try:
            ramble_manager.cancel_experiments(scheduler_name=scheduler_name)
        except BaseException as e:  # Handled on the calling thread.
            self._error = e

    def _wait(self) -> None:
        if self._thread is None:
            return
        try:
            if self._thread.is_alive():
                with theme.get_progress_spinner("Cleaning up CHS resources..."):
                    self._thread.join()
            else:
                self._thread.join()
        except KeyboardInterrupt:
            logger.warning(
                "Interrupted while cleaning up CHS resources for run %s; "
                "cleanup may be incomplete. Run 'chs cancel' to finish "
                "cleaning up.",
                self._run_id,
            )
            raise

    def _handle_outcome(self, *, body_failed: bool) -> None:
        error, self._error = self._error, None
        if error is None:
            logger.debug("Cleaned up CHS resources for run %s", self._run_id)
            return
        if isinstance(
            error, (ex.ChsException, subprocess.SubprocessError, OSError)
        ):
            logger.warning("Failed to clean CHS resources. Error: %s", error)
            return
        if body_failed:
            logger.warning("Failed to clean CHS resources. Error: %s", error)
            logger.debug("Background cleanup failed.", exc_info=error)
            return
        raise error
