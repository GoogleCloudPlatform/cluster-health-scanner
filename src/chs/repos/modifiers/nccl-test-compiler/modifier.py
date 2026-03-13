# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

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
        default="{workspace_inputs}/nccl-tests-{nccl_tests_ref}",
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
        default="git clone {nccl_tests_url} {nccl_tests_build_dir} && cd {nccl_tests_build_dir} && git checkout {nccl_tests_ref} && make {nccl_tests_make_args}",
        description="The core commands to clone and build nccl-tests",
        modes=["standard"],
    )

    modifier_variable(
        "nccl_tests_build_wrapper",
        default='srun -N 1 --ntasks=1 --container-mounts="{workspace_inputs}:{workspace_inputs}" --container-image="{container_path}" bash -c "{nccl_tests_build_command}"',
        description="Execution wrapper for the build command (defaults to srun with container mounts)",
        modes=["standard"],
    )

    build_script = """
if [ ! -d "{nccl_tests_build_dir}/build" ]; then
  {nccl_tests_build_wrapper}
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
