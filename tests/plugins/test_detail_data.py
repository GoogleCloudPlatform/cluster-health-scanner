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

"""Unit tests for plugin renderer get_detail_data() methods."""

from chs.plugins.dcgm.renderer import DcgmRenderer
from chs.plugins.gemm.renderer import GemmRenderer
from chs.plugins.nccl_cluster.renderer import NcclClusterRenderer
from chs.plugins.nccl_pairwise.renderer import NcclPairwiseRenderer
from chs.plugins.nccl_single_node.renderer import NcclSingleNodeRenderer
from chs.plugins.nvidia_smi.renderer import NvidiaSmiRenderer
from chs.plugins.stream.renderer import StreamRenderer
from chs.plugins.tpu_collectives.renderer import TpuCollectivesRenderer
from chs.plugins.tpu_gemm.renderer import TpuGemmRenderer
from chs.plugins.tpu_hbm.renderer import TpuHbmRenderer
from chs.plugins.tpu_host_device.renderer import TpuHostDeviceRenderer
from chs.plugins.tpu_network_diagnostics.renderer import (
    TpuNetworkDiagnosticsRenderer,
)
from chs.protos import bill_of_health_pb2 as pb


def test_dcgm_renderer_get_detail_data():
    renderer = DcgmRenderer()
    assert renderer.get_detail_data([]) is None

    res = pb.TestResult(test_name="dcgm", test_level="1")
    res.target_resources.add(node_name="node-1")
    m = res.metrics.add(key="Pass", unit="")
    m.value.string_value = "PASS"
    m.metadata.fields["context"].string_value = "level-1"
    m.metadata.fields["sub_context"].string_value = "GPU0"

    report = renderer.get_detail_data([res])
    assert report is not None
    assert report.test_name == "dcgm"
    assert len(report.tables) == 1
    assert len(report.tables[0].rows) == 1
    assert report.tables[0].rows[0]["node"] == "node-1"


def test_nvidia_smi_renderer_get_detail_data():
    renderer = NvidiaSmiRenderer()
    assert renderer.get_detail_data([]) is None

    res = pb.TestResult(test_name="nvidia-smi", test_level="local")
    res.target_resources.add(node_name="node-1")

    m1 = res.metrics.add(key="Driver Version")
    m1.value.string_value = "535.104.12"
    m2 = res.metrics.add(key="CUDA Version")
    m2.value.string_value = "12.2"
    m3 = res.metrics.add(key="GPU Count")
    m3.value.number_value = 8

    report = renderer.get_detail_data([res])
    assert report is not None
    assert report.test_name == "nvidia-smi"
    assert len(report.tables) >= 1
    summary_table = report.tables[0]
    assert summary_table.title == "NVIDIA SMI Summary"
    assert summary_table.rows[0]["driver_version"] == "535.104.12"


def test_gemm_renderer_get_detail_data():
    renderer = GemmRenderer()
    assert renderer.get_detail_data([]) is None

    res = pb.TestResult(test_name="gemm", test_level="standard")
    res.target_resources.add(node_name="node-1")
    m = res.metrics.add(key="Value")
    m.value.number_value = 120500.0
    m.metadata.fields["context"].string_value = "MEDIAN"

    report = renderer.get_detail_data([res])
    assert report is not None
    assert report.test_name == "gemm"
    assert len(report.tables) == 1
    assert report.tables[0].rows[0]["median_tflops"] == "120.50"


def test_stream_renderer_get_detail_data():
    renderer = StreamRenderer()
    assert renderer.get_detail_data([]) is None

    res = pb.TestResult(test_name="stream", test_level="standard")
    res.target_resources.add(node_name="node-1")

    m1 = res.metrics.add(key="Rate")
    m1.value.number_value = 950500.0
    m1.metadata.fields["context"].string_value = "Device 0 Copy"

    report = renderer.get_detail_data([res])
    assert report is not None
    assert report.test_name == "stream"
    assert len(report.tables) == 1
    assert report.tables[0].rows[0]["copy_rate"] == "950.50"


def test_tpu_gemm_renderer_get_detail_data():
    renderer = TpuGemmRenderer()
    assert renderer.get_detail_data([]) is None

    res = pb.TestResult(test_name="tpu-gemm", test_level="check")
    res.target_resources.add(node_name="node-1")

    m1 = res.metrics.add(key="Throughput")
    m1.value.number_value = 120.5
    m1.metadata.fields[
        "context"
    ].string_value = "node-1,1024x1024x1024,bfloat16"

    report = renderer.get_detail_data([res])
    assert report is not None
    assert report.test_name == "tpu-gemm"
    assert len(report.tables) == 1
    assert report.tables[0].rows[0]["throughput"] == "120.50"


def test_tpu_hbm_renderer_get_detail_data():
    renderer = TpuHbmRenderer()
    assert renderer.get_detail_data([]) is None

    res = pb.TestResult(test_name="tpu-hbm", test_level="check")
    m = res.metrics.add(key="Bandwidth (median)")
    m.value.number_value = 280.5
    m.metadata.fields["context"].string_value = "node-1,134217728,float32"

    report = renderer.get_detail_data([res])
    assert report is not None
    assert report.test_name == "tpu-hbm"
    assert len(report.tables) == 1
    assert report.tables[0].rows[0]["bandwidth"] == "280.50"


def test_tpu_host_device_renderer_get_detail_data():
    renderer = TpuHostDeviceRenderer()
    assert renderer.get_detail_data([]) is None

    res = pb.TestResult(test_name="tpu-host-device", test_level="check")
    m = res.metrics.add(key="H2D Bandwidth")
    m.value.number_value = 15.6
    m.metadata.fields["context"].string_value = "node-1,1024MiB"

    report = renderer.get_detail_data([res])
    assert report is not None
    assert report.test_name == "tpu-host-device"
    assert len(report.tables) == 1
    assert report.tables[0].rows[0]["h2d"] == "15.60"


def test_tpu_network_diagnostics_renderer_get_detail_data():
    renderer = TpuNetworkDiagnosticsRenderer()
    assert renderer.get_detail_data([]) is None

    res = pb.TestResult(test_name="tpu-network-diagnostics", test_level="check")
    m = res.metrics.add(key="Bandwidth")
    m.value.number_value = 80.0
    m.metadata.fields["context"].string_value = "node-1 Axis-0"

    report = renderer.get_detail_data([res])
    assert report is not None
    assert report.test_name == "tpu-network-diagnostics"
    assert len(report.tables) == 1
    assert report.tables[0].rows[0]["node"] == "node-1"
    assert report.tables[0].rows[0]["bandwidth"] == "80.00"


def test_tpu_collectives_renderer_get_detail_data():
    renderer = TpuCollectivesRenderer()
    assert renderer.get_detail_data([]) is None

    res = pb.TestResult(test_name="tpu-collectives", test_level="all-reduce")
    res.target_resources.add(node_name="node-1")
    m = res.metrics.add(key="Bandwidth (median)")
    m.value.number_value = 111.01
    m.metadata.fields["context"].string_value = "all-reduce,1024,float32"

    report = renderer.get_detail_data([res])
    assert report is not None
    assert report.test_name == "tpu-collectives"
    assert len(report.tables) == 1
    assert report.tables[0].rows[0]["bandwidth"] == "111.01"


def test_nccl_renderers_get_detail_data():
    cluster_renderer = NcclClusterRenderer()
    pairwise_renderer = NcclPairwiseRenderer()
    single_node_renderer = NcclSingleNodeRenderer()

    res = pb.TestResult(test_name="nccl-cluster", test_level="all-reduce")
    res.target_resources.add(node_name="node-1")
    m = res.metrics.add(key="Bandwidth")
    m.value.number_value = 250.0
    m.metadata.fields["context"].string_value = "1073741824"

    cluster_report = cluster_renderer.get_detail_data([res])
    assert cluster_report is not None
    assert cluster_report.test_name == "nccl-cluster"
    assert cluster_report.tables[0].rows[0]["peak_bw"] == "250.00"

    res_pw = pb.TestResult(test_name="nccl-pairwise", test_level="all-reduce")
    res_pw.target_resources.add(node_name="node-1")
    res_pw.target_resources.add(node_name="node-2")
    m_pw = res_pw.metrics.add(key="Bandwidth")
    m_pw.value.number_value = 240.0
    m_pw.metadata.fields["context"].string_value = "1073741824"

    pairwise_report = pairwise_renderer.get_detail_data([res_pw])
    assert pairwise_report is not None
    assert pairwise_report.tables[0].rows[0]["peak_bw"] == "240.00"

    res_sn = pb.TestResult(
        test_name="nccl-single-node", test_level="all-reduce"
    )
    res_sn.target_resources.add(node_name="node-1")
    m_sn = res_sn.metrics.add(key="Bandwidth")
    m_sn.value.number_value = 260.0
    m_sn.metadata.fields["context"].string_value = "1073741824"

    single_node_report = single_node_renderer.get_detail_data([res_sn])
    assert single_node_report is not None
    assert single_node_report.tables[0].rows[0]["peak_bw"] == "260.00"
