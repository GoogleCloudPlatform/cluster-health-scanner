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

import os
from typing import Any, Iterable
import urllib.parse
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


GCS_CONSOLE_URL_PREFIX = "https://storage.cloud.google.com/"
_URL_SAFE_CHARS = ":/?#@!$&'()*+,;=-._~%"


def get_web_href(path_or_uri: str | None) -> str | None:
    """Returns a browser-openable URL for a remote URI, or None otherwise.

    gs:// URIs map to the Cloud Storage console; http(s):// URLs are returned
    as-is apart from URL-encoding unsafe characters. Local paths return None:
    OSC 8 terminal hyperlinks to file:// (or relative) targets fail with
    "No application found to open URL" on environments without a registered
    file-scheme handler (e.g. Cloudtop, SSH and browser-based terminals).
    """
    if not path_or_uri:
        return None
    if path_or_uri.startswith("gs://"):
        href = path_or_uri.replace("gs://", GCS_CONSOLE_URL_PREFIX, 1)
    elif path_or_uri.startswith(("http://", "https://")):
        href = path_or_uri
    else:
        return None
    return urllib.parse.quote(href, safe=_URL_SAFE_CHARS)


def format_location(path_or_uri: str | os.PathLike[str]) -> str:
    """Formats an artifact location as Rich markup.

    The location text is markup-escaped so paths containing brackets render
    verbatim. Remote URIs are wrapped in a clickable hyperlink; local paths are
    left as plain text so terminals can detect, open or copy them natively.
    """
    location = str(path_or_uri)
    text = escape(location)
    href = get_web_href(location)
    if href:
        return f"[link={href}]{text}[/link]"
    return text
