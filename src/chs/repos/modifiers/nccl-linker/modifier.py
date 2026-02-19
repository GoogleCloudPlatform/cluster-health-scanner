# Copyright 2022-2026 The Ramble Authors
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

from ramble.modkit import *


class NcclLinker(BasicModifier):
    """Modifier to ensure NCCL tests are run using gIB or TCPX(O)"""

    name = "nccl-linker"

    tags("gpu")

    maintainers("bytetwin")

    mode(
        "auto", description="Auto detected shell based on config:shell setting"
    )
    default_mode("auto")

    executable_modifier("override_nccltest_executables")

    def override_nccltest_executables(
        self, executable_name, executable, app_inst=None
    ):
        prepend_execs = []
        append_execs = []

        machine_type = self.expander.expand_var_name("machine_type")
        if machine_type == "a3-high":
            if executable_name == "all-to-all-execute":
                executable.template = [
                    'bash -c "export'
                    " LD_LIBRARY_PATH=/var/lib/tcpx/lib64:${LD_LIBRARY_PATH};"
                    ' ./alltoall_perf {additional_args}"'
                ]
            if executable_name == "all-reduce-execute":
                executable.template = [
                    'bash -c "export'
                    " LD_LIBRARY_PATH=/var/lib/tcpx/lib64:${LD_LIBRARY_PATH};"
                    ' ./all_reduce_perf {additional_args}"'
                ]
            if executable_name == "all-gather-execute":
                executable.template = [
                    'bash -c "export'
                    " LD_LIBRARY_PATH=/var/lib/tcpx/lib64:${LD_LIBRARY_PATH};"
                    ' ./all_gather_perf {additional_args}"'
                ]
            if executable_name == "reduce-scatter-execute":
                executable.template = [
                    'bash -c "export'
                    " LD_LIBRARY_PATH=/var/lib/tcpx/lib64:${LD_LIBRARY_PATH};"
                    ' ./reduce_scatter_perf {additional_args}"'
                ]
            if executable_name == "send-recv-execute":
                executable.template = [
                    'bash -c "export'
                    " LD_LIBRARY_PATH=/var/lib/tcpx/lib64:${LD_LIBRARY_PATH};"
                    ' ./sendrecv_perf {additional_args}"'
                ]
        if machine_type == "a3-ultra":
            if executable_name == "all-to-all-execute":
                executable.template = [
                    'bash -c "export'
                    " LD_LIBRARY_PATH=/usr/local/gib/lib64:${LD_LIBRARY_PATH};"
                    ' ./alltoall_perf {additional_args}"'
                ]
            if executable_name == "all-reduce-execute":
                executable.template = [
                    'bash -c "export'
                    " LD_LIBRARY_PATH=/usr/local/gib/lib64:${LD_LIBRARY_PATH};"
                    ' ./all_reduce_perf {additional_args}"'
                ]
            if executable_name == "all-gather-execute":
                executable.template = [
                    'bash -c "export'
                    " LD_LIBRARY_PATH=/usr/local/gib/lib64:${LD_LIBRARY_PATH};"
                    ' ./all_gather_perf {additional_args}"'
                ]
            if executable_name == "reduce-scatter-execute":
                executable.template = [
                    'bash -c "export'
                    " LD_LIBRARY_PATH=/usr/local/gib/lib64:${LD_LIBRARY_PATH};"
                    ' ./reduce_scatter_perf {additional_args}"'
                ]
            if executable_name == "send-recv-execute":
                executable.template = [
                    'bash -c "export'
                    " LD_LIBRARY_PATH=/usr/local/gib/lib64:${LD_LIBRARY_PATH};"
                    ' ./sendrecv_perf {additional_args}"'
                ]

        return prepend_execs, append_execs
