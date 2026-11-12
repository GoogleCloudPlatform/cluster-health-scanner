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

"""Lifecycle Manager for CHS Jobs."""

from dataclasses import dataclass
from enum import StrEnum

from chs.constants import K8S_JOB_STARTUP_BUDGET_SEC
from chs.protos import bill_of_health_pb2 as pb


class JobStatus(StrEnum):
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


class ReasonCode(StrEnum):
    """Why a job did not finish."""

    SLOW_STARTUP = "SLOW_STARTUP"
    IMAGE_PULL_FAILED = "IMAGE_PULL_FAILED"
    NOT_SCHEDULED = "NOT_SCHEDULED"
    NODE_BUSY = "NODE_BUSY"
    STATUS_UNAVAILABLE = "STATUS_UNAVAILABLE"


@dataclass(frozen=True)
class JobState:
    """A job status and, if it did not finish, the reason code."""

    status: str
    reason: ReasonCode | None = None


TIME_EXCEEDED_PB_STATUSES = {
    JobStatus.TIMEOUT.value: pb.RESULT_STATUS_TIMEOUT,
    JobStatus.DEADLINE.value: pb.RESULT_STATUS_DEADLINE,
}


_REASON_LABELS = {
    ReasonCode.SLOW_STARTUP: (
        f"Pod didn't start within {K8S_JOB_STARTUP_BUDGET_SEC // 60} min"
    ),
    ReasonCode.IMAGE_PULL_FAILED: "Image pull failed",
    ReasonCode.NOT_SCHEDULED: "Not scheduled",
    ReasonCode.NODE_BUSY: "Node busy",
    ReasonCode.STATUS_UNAVAILABLE: "Status unavailable",
}


def incomplete_label(test_result: pb.TestResult) -> str:
    """Returns the tooltip cause for a reason code, or ""."""
    code = test_result.metadata.fields.get("reason_code")
    return _REASON_LABELS.get(code.string_value, "") if code else ""
