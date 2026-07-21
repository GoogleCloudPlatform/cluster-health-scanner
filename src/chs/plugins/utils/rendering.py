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

from typing import Any, Iterable
from chs.protos import bill_of_health_pb2 as pb
from rich.markup import escape


def extract_node_names(
    target_resources: Iterable[pb.TargetResource],
) -> list[str]:
    """Extracts raw node names from target resources."""
    if not target_resources:
        return []
    names = []
    for r in target_resources:
        if hasattr(r, "node_name"):
            names.append(r.node_name)
        elif isinstance(r, dict) and "node_name" in r:
            names.append(r["node_name"])
        else:
            names.append(str(r))
    return names


def format_nodes(
    target_resources: Iterable[pb.TargetResource], join_str: str = ", "
) -> str:
    if not target_resources:
        return ""
    formatted = []
    for r in target_resources:
        if hasattr(r, "node_name"):
            if getattr(r, "instance_id", None):
                formatted.append(f"{r.node_name} ({r.instance_id})")
            else:
                formatted.append(r.node_name)
        elif isinstance(r, dict) and "node_name" in r:
            if r.get("instance_id"):
                formatted.append(f"{r['node_name']} ({r['instance_id']})")
            else:
                formatted.append(r["node_name"])
        else:
            formatted.append(str(r))
    return join_str.join(formatted)


def format_context_prefix(metric: pb.Metric) -> str:
    """Formats the context of a metric into a prefix string."""
    ctx_name = (
        metric.metadata.fields["context"].string_value
        if "context" in metric.metadata.fields
        else ""
    )
    return (
        escape(f"[{ctx_name}] ") if ctx_name and ctx_name != "unknown" else ""
    )


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


def format_val(
    val: float | int | None, precision: int = 2, default: str = "N/A"
) -> str:
    """Formats numeric value with precision, or returns default if None."""
    if val is None:
        return default
    return f"{val:.{precision}f}"
