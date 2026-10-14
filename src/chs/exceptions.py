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
import click

logger = logging.getLogger(__name__)


def format_subprocess_error(e: Exception) -> str:
    """Formats an exception into a clean, detailed error message for CLI output.

    Combines command context with stderr/stdout for CalledProcessError,
    or falls back to str(e).
    """
    if isinstance(e, subprocess.CalledProcessError):
        raw_output = e.stderr if e.stderr is not None else e.stdout
        if isinstance(raw_output, bytes):
            raw_output = raw_output.decode("utf-8", errors="replace")
        output = (raw_output or "").strip()
        if output:
            cmd_str = str(e).rstrip(".")
            return f"{cmd_str}: {output}"
        return str(e)
    return str(e)


class ChsException(click.ClickException):
    """Base exception class for all CHS-related errors."""

    def __init__(self, message: str = ""):
        super().__init__(message)

    def show(self, file=None):
        logger.error(self.format_message())


class ChsValidationError(ChsException):
    """Raised when configuration or arguments fail validation."""


class ChsPreflightException(ChsException):
    """Raised when a preflight check fails."""


class NoPreviousRunsFoundException(ChsException):
    """Raised when no previous CHS runs are found."""


class WorkspaceCreationException(ChsException):
    """Raised when the workspaces directory cannot be created."""


class WorkspaceNotFoundException(ChsException):
    """Raised when a specified workspace is not found."""


class ChsExecutionException(ChsException):
    """Raised for errors during test suite execution."""


class ChsAnalysisException(ChsException):
    """Raised for errors during results analysis."""


class BillOfHealthLoadException(ChsException):
    """Raised when the bill_of_health.json file cannot be loaded or parsed."""


class StorageObjectNotFoundException(ChsException):
    """Raised when a storage object is not found in the remote provider."""


class BillOfHealthNotFoundException(ChsException):
    """Raised when the bill_of_health.json file is not found."""


class JobSubmissionFailureException(ChsException):
    """Raised when job submission fails."""


class ChsCancellationException(ChsException):
    """Raised for errors during job cancellation."""


class ChsRunInProgressException(ChsException):
    """Raised when a new run is initiated while another is still active."""


class NoNodesProvidedException(ChsException):
    """Raised when no nodes are specified for a run."""


class UnsupportedSchedulerException(ChsException):
    """Raised when an unsupported scheduler is specified."""


class ChsSchedulerException(ChsException):
    """Raised for general scheduler-specific issues like argument errors."""


class SuiteNotFoundException(ChsException):
    """Raised when a specific test suite is not found."""


class ChsArchiveException(ChsException):
    """Raised for errors during workspace archival."""


class NetworkDiscoveryException(ChsException):
    """Raised when network discovery fails to find required networks."""


class ClusterDiscoveryException(ChsException):
    """Raised when cluster discovery/retrieval fails."""


class MachineTypeDiscoveryException(ChsException):
    """Raised when machine type discovery fails."""


class InvalidTestConfigException(ChsException):
    """Raised test config defined in suites.yaml is invalid"""

    pass
