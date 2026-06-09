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

from ramble.modkit import *


class NcclTestCompiler(BasicModifier):
    """Modifier to clone and build nccl-tests"""

    name = "nccl-test-compiler"

    tags("compile", "nccl")

    maintainers("bytetwin")

    mode(
        "standard",
        description="Standard execution mode for building nccl-tests",
    )
    default_mode("standard")

    modifier_variable(
        "nccl_tests_url",
        default="https://github.com/NVIDIA/nccl-tests.git",
        description="URL for the nccl-tests repository",
        modes=["standard"],
    )

    modifier_variable(
        "nccl_tests_ref",
        default="master",
        description="Git reference (branch, tag, or commit) to checkout",
        modes=["standard"],
    )

    modifier_variable(
        "nccl_tests_build_dir",
        default="{workspace_shared}/nccl-tests-{nccl_tests_ref}",
        description="Directory to clone and build nccl-tests",
        modes=["standard"],
    )

    modifier_variable(
        "nccl_tests_make_args",
        default="MPI=1 CC=mpicc CXX=mpicxx -j",
        description="Arguments for the make command",
        modes=["standard"],
    )

    modifier_variable(
        "nccl_tests_build_command",
        default="[ ! -d {nccl_tests_build_dir} ] && git clone {nccl_tests_url} {nccl_tests_build_dir}; cd {nccl_tests_build_dir} && git checkout {nccl_tests_ref} && make {nccl_tests_make_args}",
        description="The core commands to clone and build nccl-tests",
        modes=["standard"],
    )

    modifier_variable(
        "nccl_tests_build_wrapper",
        default='srun -N 1 --ntasks=1 --container-mounts="{workspace_shared}:{workspace_shared}" --container-image="{container_path}" bash -c "{nccl_tests_build_command}"',
        description="Execution wrapper for the build command (defaults to srun with container mounts)",
        modes=["standard"],
    )

    build_script = """
LOCK_DIR="{nccl_tests_build_dir}.lock"
SUCCESS_FILE="{nccl_tests_build_dir}/.success"
FAIL_FILE="{nccl_tests_build_dir}.failed"

if mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Acquired build lock. Starting compilation..."
  if [ ! -f "$SUCCESS_FILE" ]; then
    {nccl_tests_build_wrapper}
    BUILD_STATUS=$?
    if [ $BUILD_STATUS -eq 0 ]; then
      echo "Compilation successful."
      touch "$SUCCESS_FILE"
    else
      echo "Compilation failed with exit code $BUILD_STATUS!"
      touch "$FAIL_FILE"
      rmdir "$LOCK_DIR"
      exit $BUILD_STATUS
    fi
  fi
  rmdir "$LOCK_DIR"
else
  echo "Waiting for another node to finish compiling..."
  wait_time=0
  max_wait=1800
  while [ ! -f "$SUCCESS_FILE" ]; do
    if [ -f "$FAIL_FILE" ]; then
      echo "Another node failed to compile NCCL tests. Aborting."
      exit 1
    fi
    if [ $wait_time -ge $max_wait ]; then
      echo "Timeout waiting for compile lock. Aborting."
      exit 1
    fi
    sleep 5
    wait_time=$((wait_time + 5))
  done
  echo "Compilation finished by another node. Proceeding."
fi
"""

    modifier_variable(
        "build_nccl_tests_script",
        default=build_script.strip(),
        description="Bash script to clone and build nccl-tests",
        modes=["standard"],
    )

    executable_modifier("build_nccl_tests", usage_filter="once")

    def build_nccl_tests(self, executable_name, executable, app_inst=None):
        from ramble.util.executable import CommandExecutable

        pre_exec = []
        post_exec = []

        pre_exec.append(
            CommandExecutable(
                "build-nccl-tests",
                template=["{build_nccl_tests_script}"],
            )
        )

        return pre_exec, post_exec
