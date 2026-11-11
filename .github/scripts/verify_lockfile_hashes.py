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

"""Checks that lockfile hashes belong to the pinned package versions.

`uv pip compile` keeps the hashes already in the output file, so a
version edited by hand (`pip==26.2.1` -> `pip==21.0`) keeps the old
version's hashes and the pip-compile hook still passes. The Docker
build only fails later, at `pip install --require-hashes`.

For every `name==version` entry, this checks that each listed sha256
is a file PyPI publishes for that exact version.

Usage:
  verify_lockfile_hashes.py requirements.txt [requirements-system.txt ...]
"""

import concurrent.futures
import json
import re
import sys
import time
import urllib.error
import urllib.request

_ENTRY = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")
_HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})")
_PYPI_URL = "https://pypi.org/pypi/{name}/{version}/json"


def parse_lockfile(text: str) -> list[tuple[str, str, set[str]]]:
    """Returns (name, version, sha256 hashes) for each pinned entry."""
    entries = []
    for line in text.splitlines():
        entry = _ENTRY.match(line)
        if entry:
            entries.append((entry.group(1), entry.group(2), set()))
        if entries:
            entries[-1][2].update(_HASH.findall(line))
    return entries


def fetch_pypi_hashes(name: str, version: str, attempts: int = 3) -> set[str]:
    """Returns the sha256 digests of all files PyPI has for name==version."""
    url = _PYPI_URL.format(name=name, version=version)
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.load(response)
            return {f["digests"]["sha256"] for f in data.get("urls", [])}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return set()
            if attempt == attempts - 1:
                raise
        except urllib.error.URLError:
            if attempt == attempts - 1:
                raise
        time.sleep(2**attempt)
    return set()


def check_entries(entries, fetch=fetch_pypi_hashes) -> list[str]:
    """Returns an error for each entry whose hashes do not match PyPI."""
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = {
            pool.submit(fetch, name, version): (name, version, hashes)
            for name, version, hashes in entries
        }
        for future in concurrent.futures.as_completed(futures):
            name, version, hashes = futures[future]
            if not hashes:
                errors.append(f"{name}=={version}: no --hash entries")
                continue
            published = future.result()
            if not published:
                errors.append(f"{name}=={version}: not found on PyPI")
                continue
            stale = sorted(hashes - published)
            if stale:
                errors.append(
                    f"{name}=={version}: {len(stale)} of {len(hashes)}"
                    f" hashes are not files of this version"
                    f" (first: {stale[0][:16]}...)"
                )
    return sorted(errors)


def main(argv: list[str]) -> int:
    failed = False
    for path in argv:
        with open(path, encoding="utf-8") as f:
            entries = parse_lockfile(f.read())
        for error in check_entries(entries):
            print(f"{path}: {error}")
            failed = True
    if failed:
        print(
            "\nRegenerate the lockfile instead of editing it by hand: run the"
            " `uv pip compile` command from its header, with"
            " `--upgrade-package <name>` to change a version."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
