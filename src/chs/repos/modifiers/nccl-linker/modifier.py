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
        if machine_type == "a3-highgpu-8g":
            if executable_name == "all-to-all-execute":
                executable.template = [
                    'bash -eo pipefail -c "export'
                    " LD_LIBRARY_PATH=/var/lib/tcpx/lib64:${LD_LIBRARY_PATH};"
                    ' ./alltoall_perf {additional_args}"'
                ]
            if executable_name == "all-reduce-execute":
                executable.template = [
                    'bash -eo pipefail -c "export'
                    " LD_LIBRARY_PATH=/var/lib/tcpx/lib64:${LD_LIBRARY_PATH};"
                    ' ./all_reduce_perf {additional_args}"'
                ]
            if executable_name == "all-gather-execute":
                executable.template = [
                    'bash -eo pipefail -c "export'
                    " LD_LIBRARY_PATH=/var/lib/tcpx/lib64:${LD_LIBRARY_PATH};"
                    ' ./all_gather_perf {additional_args}"'
                ]
            if executable_name == "reduce-scatter-execute":
                executable.template = [
                    'bash -eo pipefail -c "export'
                    " LD_LIBRARY_PATH=/var/lib/tcpx/lib64:${LD_LIBRARY_PATH};"
                    ' ./reduce_scatter_perf {additional_args}"'
                ]
            if executable_name == "send-recv-execute":
                executable.template = [
                    'bash -eo pipefail -c "export'
                    " LD_LIBRARY_PATH=/var/lib/tcpx/lib64:${LD_LIBRARY_PATH};"
                    ' ./sendrecv_perf {additional_args}"'
                ]
        if machine_type == "a3-ultragpu-8g" or machine_type == "a4-highgpu-8g":
            if executable_name == "all-to-all-execute":
                executable.template = [
                    'bash -eo pipefail -c "export'
                    " LD_LIBRARY_PATH=/usr/local/gib/lib64:${LD_LIBRARY_PATH};"
                    ' ./alltoall_perf {additional_args}"'
                ]
            if executable_name == "all-reduce-execute":
                executable.template = [
                    'bash -eo pipefail -c "export'
                    " LD_LIBRARY_PATH=/usr/local/gib/lib64:${LD_LIBRARY_PATH};"
                    ' ./all_reduce_perf {additional_args}"'
                ]
            if executable_name == "all-gather-execute":
                executable.template = [
                    'bash -eo pipefail -c "export'
                    " LD_LIBRARY_PATH=/usr/local/gib/lib64:${LD_LIBRARY_PATH};"
                    ' ./all_gather_perf {additional_args}"'
                ]
            if executable_name == "reduce-scatter-execute":
                executable.template = [
                    'bash -eo pipefail -c "export'
                    " LD_LIBRARY_PATH=/usr/local/gib/lib64:${LD_LIBRARY_PATH};"
                    ' ./reduce_scatter_perf {additional_args}"'
                ]
            if executable_name == "send-recv-execute":
                executable.template = [
                    'bash -eo pipefail -c "export'
                    " LD_LIBRARY_PATH=/usr/local/gib/lib64:${LD_LIBRARY_PATH};"
                    ' ./sendrecv_perf {additional_args}"'
                ]
        if machine_type == "a3-megagpu-8g":
            if executable_name == "all-to-all-execute":
                executable.template = [
                    'bash -eo pipefail -c "export LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:/usr/lib/x86_64-linux-gnu:\\$LD_LIBRARY_PATH; ./alltoall_perf {additional_args}"'
                ]
            if executable_name == "all-reduce-execute":
                executable.template = [
                    'bash -eo pipefail -c "export LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:/usr/lib/x86_64-linux-gnu:\\$LD_LIBRARY_PATH; ./all_reduce_perf {additional_args}"'
                ]
            if executable_name == "all-gather-execute":
                executable.template = [
                    'bash -eo pipefail -c "export LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:/usr/lib/x86_64-linux-gnu:\\$LD_LIBRARY_PATH; ./all_gather_perf {additional_args}"'
                ]
            if executable_name == "reduce-scatter-execute":
                executable.template = [
                    'bash -eo pipefail -c "export LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:/usr/lib/x86_64-linux-gnu:\\$LD_LIBRARY_PATH; ./reduce_scatter_perf {additional_args}"'
                ]
            if executable_name == "send-recv-execute":
                executable.template = [
                    'bash -eo pipefail -c "export LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:/usr/lib/x86_64-linux-gnu:\\$LD_LIBRARY_PATH; ./sendrecv_perf {additional_args}"'
                ]
        return prepend_execs, append_execs
