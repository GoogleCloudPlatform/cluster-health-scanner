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

"""Formatting utilities for CHS."""

import datetime


def format_relative_time(timestamp_iso: str) -> str:
    """Calculates and formats the relative time from an ISO timestamp.

    Args:
        timestamp_iso: An ISO 8601 formatted timestamp string.

    Returns:
        A formatted relative time string (e.g., '10m ago', '2d ago').
    """
    if not timestamp_iso:
        return "Unknown"
    try:
        # Replace 'Z' with '+00:00' to support standard isoformat parsing
        sub_time = datetime.datetime.fromisoformat(
            timestamp_iso.replace("Z", "+00:00")
        )
        if sub_time.tzinfo is None:
            sub_time = sub_time.replace(tzinfo=datetime.UTC)

        now = datetime.datetime.now(datetime.UTC)
        diff = now - sub_time
        seconds = max(0, diff.total_seconds())

        if seconds < 60:
            return f"{int(seconds)}s ago"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m ago"
        elif seconds < 86400:
            return f"{int(seconds // 3600)}h ago"
        return f"{int(seconds // 86400)}d ago"
    except (ValueError, TypeError):
        return "Unknown"


def format_names(names: list[str], max_display: int = 3) -> str:
    """Joins the first names and counts the rest.

    Args:
        names: The names to list.
        max_display: How many names to show before counting the rest.

    Returns:
        The names joined by commas, e.g. 'a, b, c', or the first
        max_display of them and a count of the rest, e.g. 'a, b and 3 others'.
    """
    shown = ", ".join(names[:max_display])
    hidden = len(names) - max_display
    if hidden <= 0:
        return shown
    return f"{shown} and {hidden} {'other' if hidden == 1 else 'others'}"


def format_storage_path(storage_dict: dict | None) -> str:
    """Formats a storage configuration dictionary into a concise string.

    Args:
        storage_dict: The storage dictionary from the run metadata.

    Returns:
        A concise path representation (e.g. 'gs://bucket/prefix' or local path),
        or 'Local' if no remote storage is configured.
    """
    if not storage_dict:
        return "Local"
    storage_type = storage_dict.get("type", "unknown").upper()
    if storage_type == "GCS":
        gcs_config = storage_dict.get("gcs", {})
        bucket = gcs_config.get("bucket_name", "").strip("/")
        prefix = gcs_config.get("prefix", "").strip("/")
        if prefix:
            return f"gs://{bucket}/{prefix}"
        return f"gs://{bucket}"
    elif storage_type == "PATH":
        return storage_dict.get("path", {}).get("path", "")
    return "Unknown"
