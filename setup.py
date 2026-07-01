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
import subprocess
import sys
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyCommand(build_py):
    """Custom build_py command to generate protobuf files."""

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

        super().run()


setup(
    cmdclass={
        "build_py": BuildPyCommand,
    },
)
