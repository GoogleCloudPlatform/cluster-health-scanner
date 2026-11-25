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
import pathlib
import subprocess
import sys
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyCommand(build_py):
    """Custom build_py command to generate protobuf files and the TPU runtime
    versions."""

    def run(self):
        source_dir = "src"
        proto_file = os.path.join(
            source_dir, "chs", "protos", "bill_of_health.proto"
        )

        if os.path.exists(proto_file):
            # TODO: Replace print statements with proper logging statements
            print(f"Compiling {proto_file}...")
            command = [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"-I{source_dir}",
                f"--python_out={source_dir}",
                proto_file,
            ]

            try:
                subprocess.check_call(command)
                # TODO: Replace print statements with proper logging statements
                print("Protobuf compilation successful.")
            except subprocess.CalledProcessError as e:
                print(f"Protobuf compilation failed: {e}")
                sys.exit(1)
        else:
            print(f"Proto file not found: {proto_file}")

        self._save_tpu_runtime_versions(source_dir)
        super().run()

    def _save_tpu_runtime_versions(self, source_dir):
        """Saves the jax and libtpu pins of requirements.txt in the package.

        The published CHS image is built from requirements.txt, so the Bill
        of Health can report the versions that TPU pods run without querying
        them. Never fails the build: without the file, the Bill of Health
        leaves the versions out.
        """
        package_dir = os.path.join(source_dir, "chs")
        manifest = os.path.join(package_dir, "tpu_runtime_versions.json")
        # A file left by an earlier build must not outlive a failed update.
        pathlib.Path(manifest).unlink(missing_ok=True)
        requirements = "requirements.txt"
        if not os.path.exists(requirements):
            print(f"{requirements} not found, TPU runtime versions not saved.")
            return
        command = [
            sys.executable,
            "-I",
            os.path.join(package_dir, "tpu_runtime_versions.py"),
            requirements,
            manifest,
        ]
        try:
            subprocess.check_call(command)
        except subprocess.CalledProcessError as e:
            print(f"TPU runtime versions not saved: {e}")


setup(
    cmdclass={
        "build_py": BuildPyCommand,
    },
)
