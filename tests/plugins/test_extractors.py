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

"""Unit tests for plugin extractor classes."""

from chs.plugins.dcgm.extractor import DcgmExtractor
from chs.plugins.gemm.extractor import GemmExtractor
from chs.plugins.nvidia_smi.extractor import NvidiaSmiExtractor
from chs.plugins.tpu_collectives.extractor import TpuCollectivesExtractor
from chs.plugins.tpu_gemm.extractor import TpuGemmExtractor
from chs.plugins.tpu_hbm.extractor import TpuHbmExtractor
from chs.plugins.tpu_host_device.extractor import TpuHostDeviceExtractor
from chs.plugins.tpu_network_stress.extractor import (
    TpuNetworkStressExtractor,
)
from chs.protos import bill_of_health_pb2 as pb


def test_dcgm_extractor():
    extractor = DcgmExtractor()
    meta = extractor.parse_context_metadata("GPU-0")
    assert meta["gpu_id"] == "GPU-0"
    meta_group = extractor.parse_context_metadata("level-1")
    assert meta_group["test_group"] == "level-1"
    assert (
        extractor.get_description()
        == "NVIDIA Data Center GPU Manager Diagnostics"
    )


def test_gemm_extractor():
    extractor = GemmExtractor()
    ctxs = [{"name": "ctx1"}]
    assert extractor.get_required_contexts(ctxs) == ctxs


def test_nvidia_smi_extractor():
    extractor = NvidiaSmiExtractor()
    ctxs = [{"name": "ctx1"}]
    assert extractor.get_required_contexts(ctxs) == ctxs


def test_tpu_gemm_extractor():
    extractor = TpuGemmExtractor()
    meta = extractor.parse_context_metadata("node-1,1024x1024x1024,bfloat16")
    assert meta["node"] == "node-1"
    assert meta["matrix_shape"] == "1024x1024x1024"
    assert meta["dtype"] == "bfloat16"
    assert meta["sweep_param"] == "1024x1024x1024,bfloat16"


def test_tpu_hbm_extractor():
    extractor = TpuHbmExtractor()
    meta = extractor.parse_context_metadata("node-1,134217728,float32")
    assert meta["node"] == "node-1"
    assert meta["num_elements"] == 134217728
    assert meta["dtype"] == "float32"
    ctxs = [{"name": "ctx1"}]
    assert extractor.get_required_contexts(ctxs) == ctxs


def test_tpu_host_device_extractor():
    extractor = TpuHostDeviceExtractor()
    ctxs = [{"name": "ctx1"}]
    assert extractor.get_required_contexts(ctxs) == ctxs


def test_tpu_collectives_extractor():
    extractor = TpuCollectivesExtractor()
    ctxs = [{"name": "ctx1"}]
    assert extractor.get_required_contexts(ctxs) == ctxs


def test_tpu_network_stress_extractor():
    extractor = TpuNetworkStressExtractor()
    assert extractor.get_fom_aliases() == {"bandwidth": "Bandwidth"}

    res = pb.TestResult(test_name="tpu-network-stress")
    m1 = res.metrics.add(key="Bandwidth")
    m1.value.number_value = 100.0
    m1.metadata.fields["context"].string_value = "node-1 Axis-0"
    m2 = res.metrics.add(key="Bandwidth")
    m2.value.number_value = 80.0
    m2.metadata.fields["context"].string_value = "node-1 Axis-1"

    extractor.post_process_test_result(res)
    keys = [m.key for m in res.metrics]
    assert "Axis Bandwidth" in keys
    assert "Bandwidth" in keys
    min_m = next(m for m in res.metrics if m.key == "Bandwidth")
    assert min_m.value.number_value == 80.0
