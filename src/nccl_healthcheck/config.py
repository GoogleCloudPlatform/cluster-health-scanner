# Copyright 2024 Google LLC
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

"""Contains a series of config objects to use when using A-Series VMs."""

import config_pb2


def _create_a3_config():
  return config_pb2.ASeriesConfig(
      instance_type="a3-highgpu-8g",
      second_pass_yaml_path="a3/nccl_secondpass.yaml",
      nccl_test_command_template=(
          "bash /scripts/run-nccl-combined-plugins.sh gpudirect {benchmark}"
          " {ld_library_path} 8 eth1,eth2,eth3,eth4"
          " {start_message_size} {end_message_size} {nhosts} 1 {iterations}"
      ),
      default_threshold=60,
      ld_library_path="/usr/local/tcpx/lib64:/usr/local/nvidia/lib64/",
  )


def _create_a3plus_config():
  return config_pb2.ASeriesConfig(
      instance_type="a3-megagpu-8g",
      second_pass_yaml_path="a3plus/nccl_secondpass.yaml",
      nccl_test_command_template=(
          "bash /scripts/run-nccl-combined-plugins.sh fastrak {benchmark}"
          " {ld_library_path} 8 eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8"
          " {start_message_size} {end_message_size} {nhosts} 3 {iterations}"
      ),
      default_threshold=120,
      ld_library_path="/usr/local/tcpxo/lib64:/usr/local/nvidia/lib64/",
  )


def _create_a3plus_debian_config():
  return config_pb2.ASeriesConfig(
      instance_type="a3-megagpu-8g-debian",
      second_pass_yaml_path="a3plus/nccl_secondpass.yaml",
      nccl_test_command_template=(
          "bash /scripts/run-nccl-combined-plugins.sh fastrak_debian"
          " {benchmark} {ld_library_path} 8"
          " enp134s0,enp135s0,enp13s0,enp14s0,enp141s0,enp142s0,enp6s0,enp7s0"
          " {start_message_size} {end_message_size} {nhosts} 3 {iterations}"
      ),
      default_threshold=120,
      ld_library_path="/usr/local/cuda-12.2/lib64:/usr/local/nvidia/lib64/",
  )


def _create_a3ultra_config():
  return config_pb2.ASeriesConfig(
      instance_type="a3-ultragpu-8g",
      second_pass_yaml_path="a3ultra/nccl_secondpass.yaml",
      nccl_test_command_template=(
          "bash /scripts/run-nccl-combined-plugins.sh rdma {benchmark}"
          " {ld_library_path} 8 eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8"
          " {start_message_size} {end_message_size} {nhosts} 3 {iterations}"
      ),
      default_threshold=120,
      ld_library_path="/usr/local/gib/lib64:/usr/local/nvidia/lib64/",
  )


def _create_a4_config():
  return config_pb2.ASeriesConfig(
      instance_type="a4-highgpu-8g",
      second_pass_yaml_path="a4/nccl_secondpass.yaml",
      nccl_test_command_template=(
          "bash /scripts/run-nccl-combined-plugins.sh rdma {benchmark}"
          " {ld_library_path} 8 eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8"
          " {start_message_size} {end_message_size} {nhosts} 3 {iterations}"
      ),
      default_threshold=120,
      ld_library_path="/usr/local/gib/lib64:/usr/local/nvidia/lib64/",
  )


def get_config(instance_type: str) -> config_pb2.ASeriesConfig:
  """Returns the config for the given instance type.

  Args:
    instance_type: The instance type to get the config for.

  Returns:
    The config for the given instance type.

  Raises:
    ValueError: If the instance type is not supported.
  """
  if instance_type == "a3-highgpu-8g":
    return _create_a3_config()
  elif instance_type == "a3-megagpu-8g":
    return _create_a3plus_config()
  elif instance_type == "a3-megagpu-8g-debian":
    return _create_a3plus_debian_config()
  elif instance_type == "a3-ultragpu-8g":
    return _create_a3ultra_config()
  elif instance_type == "a4-highgpu-8g":
    return _create_a4_config()
  else:
    raise ValueError(f"Unsupported instance type: {instance_type}")
