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

"""Tests for the JAX and libtpu versions that CHS TPU tests run with."""

import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from chs import tpu_runtime_versions
from chs.tpu_runtime_versions import TpuRuntimeVersions

_REQUIREMENTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "requirements.txt",
)


def _installed(packages: dict[str, str]):
    """Returns a fake importlib.metadata.version for the given packages."""

    def version(name: str) -> str:
        if name not in packages:
            raise importlib.metadata.PackageNotFoundError(name)
        return packages[name]

    return version


_LOCK = """\
jax==0.12.0 ; python_full_version < '3.13' and platform_machine == 'x86_64' \\
    --hash=sha256:aaaa \\
    --hash=sha256:bbbb
    # via
    #   accelerator-microbenchmarks
jaxlib==0.12.1 ; python_full_version < '3.13' \\
    --hash=sha256:cccc
libtpu==0.0.50 ; python_full_version < '3.13' and sys_platform == 'linux' \\
    --hash=sha256:dddd
libtpu-nightly==0.1.dev1 \\
    --hash=sha256:eeee
pyyaml==6.0.3 \\
    --hash=sha256:ffff
"""


class TestTpuRuntimeVersions(unittest.TestCase):
    def setUp(self):
        tpu_runtime_versions.load.cache_clear()
        self.addCleanup(tpu_runtime_versions.load.cache_clear)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.manifest = os.path.join(tmp.name, "tpu_runtime_versions.json")
        self.requirements = os.path.join(tmp.name, "requirements.txt")
        with open(self.requirements, "w", encoding="utf-8") as f:
            f.write(_LOCK)

    def test_parse_requirements_reads_only_jax_and_libtpu_pins(self):
        self.assertEqual(
            tpu_runtime_versions.parse_requirements(_LOCK),
            {"jax": "0.12.0", "libtpu": "0.0.50"},
        )

    def test_parse_requirements_rejects_conflicting_pins(self):
        with self.assertRaisesRegex(
            ValueError, "jax is pinned to both 0.12.0 and 0.13.0"
        ):
            tpu_runtime_versions.parse_requirements(
                _LOCK + "jax==0.13.0 ; python_full_version >= '3.13'\n"
            )

    def test_parse_requirements_reads_the_repo_lock(self):
        """The lock that the CHS image is built from pins both packages."""
        with open(_REQUIREMENTS, encoding="utf-8") as f:
            versions = tpu_runtime_versions.parse_requirements(f.read())

        self.assertEqual(sorted(versions), ["jax", "libtpu"])
        self.assertTrue(all(versions.values()))

    def test_load_returns_the_versions_written_from_requirements(self):
        tpu_runtime_versions.write_manifest(self.requirements, self.manifest)

        with mock.patch.object(
            tpu_runtime_versions, "MANIFEST_PATH", self.manifest
        ):
            versions = tpu_runtime_versions.load()

        self.assertEqual(
            versions, TpuRuntimeVersions(jax="0.12.0", libtpu="0.0.50")
        )

    def test_load_returns_none_without_a_valid_manifest(self):
        for content in (
            None,
            "not json",
            "[]",
            '{"jax": "0.12.0"}',
            '{"jax": "0.12.0", "libtpu": ""}',
            '{"jax": 12, "libtpu": "0.0.50"}',
        ):
            with self.subTest(content=content):
                tpu_runtime_versions.load.cache_clear()
                if content is not None:
                    with open(self.manifest, "w", encoding="utf-8") as f:
                        f.write(content)
                with mock.patch.object(
                    tpu_runtime_versions, "MANIFEST_PATH", self.manifest
                ):
                    self.assertIsNone(tpu_runtime_versions.load())

    def test_module_writes_the_manifest_as_an_isolated_script(self):
        """setup.py runs the module before CHS dependencies are installed."""
        subprocess.run(
            [
                sys.executable,
                "-I",
                tpu_runtime_versions.__file__,
                self.requirements,
                self.manifest,
            ],
            check=True,
        )

        with open(self.manifest, encoding="utf-8") as f:
            self.assertEqual(
                json.load(f), {"jax": "0.12.0", "libtpu": "0.0.50"}
            )

    def test_module_fails_cleanly_as_a_script_without_writing(self):
        """setup.py skips the versions when the script exits with an error."""
        with open(self.requirements, "a", encoding="utf-8") as f:
            f.write("jax==0.13.0 ; python_full_version >= '3.13'\n")

        result = subprocess.run(
            [
                sys.executable,
                "-I",
                tpu_runtime_versions.__file__,
                self.requirements,
                self.manifest,
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 1)
        self.assertEqual(
            result.stderr.strip(),
            "Could not save the TPU runtime versions: jax is pinned to both "
            "0.12.0 and 0.13.0.",
        )
        self.assertFalse(os.path.exists(self.manifest))

    def test_installed_reads_the_packages_installed_with_chs(self):
        for packages, expected in (
            (
                {"jax": "0.12.0", "libtpu": "0.0.50"},
                TpuRuntimeVersions(jax="0.12.0", libtpu="0.0.50"),
            ),
            (
                {"jax": "0.12.0", "libtpu-nightly": "0.1.dev1"},
                TpuRuntimeVersions(jax="0.12.0", libtpu="0.1.dev1"),
            ),
            (
                # Checked in the same order as tpu-info's inventory script.
                {"jax": "0.12.0", "libtpu": "0.0.50", "libtpu-nightly": "0.1"},
                TpuRuntimeVersions(jax="0.12.0", libtpu="0.1"),
            ),
            ({"libtpu": "0.0.50"}, None),
            ({"jax": "0.12.0"}, None),
        ):
            with (
                self.subTest(packages=packages),
                mock.patch.object(
                    importlib.metadata, "version", _installed(packages)
                ),
            ):
                self.assertEqual(tpu_runtime_versions.installed(), expected)


if __name__ == "__main__":
    unittest.main()
