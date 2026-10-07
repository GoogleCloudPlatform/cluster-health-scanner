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
from chs.plugins.nccl_base_extractor import NcclBaseExtractor
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


def _nccl_contexts(alg_ip, alg_oop, bus_ip, bus_oop):
    """Two-size NCCL sweep shaped like a real ramble_results_cache.json."""
    return [
        {
            "name": "null",
            "foms": [
                {"name": "Out of bounds values", "value": 0},
                {"name": "Avg. Bus Bandwidth", "value": 1.0},
            ],
        },
        {
            "name": "4294967296",
            "foms": [
                {"name": "In Place Alg Bandwidth", "value": 1.0},
                {"name": "In Place Bus Bandwidth", "value": 1.0},
                {"name": "Out of Place Alg Bandwidth", "value": 1.0},
                {"name": "Out of Place Bus Bandwidth", "value": 1.0},
            ],
        },
        {
            "name": "8589934592",
            "foms": [
                {"name": "In Place Alg Bandwidth", "value": alg_ip},
                {"name": "In Place Bus Bandwidth", "value": bus_ip},
                {"name": "Out of Place Alg Bandwidth", "value": alg_oop},
                {"name": "Out of Place Bus Bandwidth", "value": bus_oop},
            ],
        },
    ]


def _peak(contexts):
    got = NcclBaseExtractor().get_required_contexts(contexts)
    return {
        c["name"]: next(
            (
                f["value"]
                for f in c["foms"]
                if f["name"] == "Peak Detected Bandwidth"
            ),
            None,
        )
        for c in got
    }


def test_nccl_peak_ignores_alg_bandwidth_for_all_gather():
    """All-gather has busbw = algbw * (n-1)/n, so algbw is the larger number.

    The peak must still be the bus value; picking the max over every FOM
    containing "Bandwidth" would wrongly select algbw here.
    """
    peaks = _peak(_nccl_contexts(147.35, 147.23, 128.93, 128.83))
    assert peaks["8589934592"] == 128.93
    assert peaks["null"] is None


def test_nccl_peak_is_bus_bandwidth_for_all_reduce():
    """All-reduce has busbw = algbw * 2(n-1)/n, so busbw is already the max.

    This case must be unaffected by the bus-only filter.
    """
    peaks = _peak(_nccl_contexts(189.08, 189.10, 330.62, 330.93))
    assert peaks["8589934592"] == 330.93


def test_nccl_peak_uses_only_the_largest_message_size():
    contexts = _nccl_contexts(147.35, 147.23, 128.93, 128.83)
    got = NcclBaseExtractor().get_required_contexts(contexts)
    assert [c["name"] for c in got] == ["null", "8589934592"]


def test_nccl_peak_falls_back_to_null_context_when_alone():
    """Legacy Ramble results only had the null context with
    Avg. Bus Bandwidth.
    """
    contexts = [
        {
            "name": "null",
            "foms": [{"name": "Avg. Bus Bandwidth", "value": 127.02}],
        }
    ]
    got = NcclBaseExtractor().get_required_contexts(contexts)
    assert len(got) == 1
    assert _peak(contexts)["null"] == 127.02


def test_nccl_peak_absent_when_no_bus_bandwidth_reported():
    contexts = [
        {
            "name": "null",
            "foms": [{"name": "Out of bounds values", "value": 0}],
        },
        {
            "name": "8589934592",
            "foms": [{"name": "In Place Alg Bandwidth", "value": 500.0}],
        },
    ]
    got = NcclBaseExtractor().get_required_contexts(contexts)
    assert all(
        f["name"] != "Peak Detected Bandwidth" for c in got for f in c["foms"]
    )


def test_nccl_peak_skips_unparseable_values():
    contexts = _nccl_contexts(1.0, 1.0, "n/a", 42.0)
    assert _peak(contexts)["8589934592"] == 42.0


def test_nccl_utils_get_peak_metrics_ignores_alg_bandwidth():
    """get_peak_metrics must only consider 'Bus Bandwidth' metrics."""
    from chs.plugins.utils.nccl_utils import get_peak_metrics

    res = pb.TestResult()
    # All-gather shape: alg_bw > bus_bw
    m_alg = res.metrics.add(key="In Place Alg Bandwidth")
    m_alg.value.number_value = 414.74
    m_alg.metadata.fields["context"].string_value = str(8 * 1024**3)

    m_bus = res.metrics.add(key="In Place Bus Bandwidth")
    m_bus.value.number_value = 362.90
    m_bus.metadata.fields["context"].string_value = str(8 * 1024**3)

    peak = get_peak_metrics(res.metrics)
    assert peak.max_bw == 362.90
    assert peak.max_size == "8.0"
