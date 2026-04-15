"""Lifecycle Manager for CHS Jobs."""

from enum import Enum


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    DEADLINE = "DEADLINE"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, status: str) -> str:
        """Converts a string into a Lifecycle Status safely.

        Fallback to UNKNOWN.
        """
        mappings = {
            cls.PENDING.value: ["PENDING"],
            cls.RUNNING.value: ["RUNNING"],
            cls.COMPLETED.value: ["COMPLETED"],
            cls.TIMEOUT.value: ["TIMEOUT"],
            cls.CANCELLED.value: ["CANCELLED"],
            cls.DEADLINE.value: ["DEADLINE"],
        }
        for target_status, values in mappings.items():
            if status.upper() in values:
                return target_status
        return cls.UNKNOWN.value

    @classmethod
    def is_terminal(cls, status: str) -> bool:
        """Returns True if the status represents a terminal state."""
        return status in [
            cls.COMPLETED.value,
            cls.TIMEOUT.value,
            cls.CANCELLED.value,
            cls.DEADLINE.value,
            cls.UNKNOWN.value,
        ]
