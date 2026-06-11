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

from enum import Enum
from chs.protos import bill_of_health_pb2 as pb


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


TIME_EXCEEDED_PB_STATUSES = {
    JobStatus.TIMEOUT.value: pb.RESULT_STATUS_TIMEOUT,
    JobStatus.DEADLINE.value: pb.RESULT_STATUS_DEADLINE,
}
