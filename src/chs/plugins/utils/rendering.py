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

from typing import Any
from chs.protos import bill_of_health_pb2 as pb


def format_nodes(target_resource_ids: list[str]) -> str:
    """Formats target resource IDs into a comma-separated string."""
    return ", ".join(target_resource_ids) if target_resource_ids else ""


def format_context_prefix(metric: pb.Metric) -> str:
    """Formats the context of a metric into a prefix string."""
    ctx_name = (
        metric.metadata.fields["context"].string_value
        if "context" in metric.metadata.fields
        else ""
    )
    return f"[{ctx_name}] " if ctx_name and ctx_name != "unknown" else ""


def parse_value(value_proto) -> Any:
    """Extracts the raw value from a google.protobuf.Value message.

    Converts floats to int if they are integers.
    """
    if value_proto.HasField("number_value"):
        val = value_proto.number_value
        if val.is_integer():
            return int(val)
        return val
    elif value_proto.HasField("bool_value"):
        return value_proto.bool_value
    return value_proto.string_value
