"""Custom exceptions for the Cluster Health Scanner."""


class ChsException(Exception):
    """Base exception class for all CHS-related errors."""


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


class BillOfHealthNotFoundException(ChsException):
    """Raised when the bill_of_health.json file is not found."""


class ChsCancellationException(ChsException):
    """Raised for errors during job cancellation."""


class ChsRunInProgressException(ChsException):
    """Raised when a new run is initiated while another is still active."""


class NoNodesProvidedException(ChsException):
    """Raised when no nodes are specified for a run."""


class UnsupportedSchedulerException(ChsException):
    """Raised when an unsupported scheduler is specified."""


class SuiteNotFoundException(ChsException):
    """Raised when a specific test suite is not found."""
