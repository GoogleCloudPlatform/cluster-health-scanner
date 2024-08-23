"""Contains a series of config objects to use when using A-Series VMs."""
import config_pb2

def _create_a3_config():
  return config_pb2.ASeriesConfig(
      instance_type="a3-highgpu-8g",
      second_pass_yaml_path="a3/nccl_secondpass.yaml",
      nccl_test_command_template=(
          "/scripts/run-nccl-combined-plugins.sh gpudirect all_gather_perf"
          " {ld_library_path} 8 eth1,eth2,eth3,eth4"
          " {start_message_size} {end_message_size} {nhosts} 1"
      ),
      default_threshold=60,
      ld_library_path="/usr/local/tcpx/lib64:/usr/local/nvidia/lib64/",
  )


def _create_a3plus_config():
  return config_pb2.ASeriesConfig(
      instance_type="a3-megagpu-8g",
      second_pass_yaml_path="a3plus/nccl_secondpass.yaml",
      nccl_test_command_template=(
          "/scripts/run-nccl-combined-plugins.sh fastrak all_gather_perf"
          " {ld_library_path} 8 eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8"
          " {start_message_size} {end_message_size} {nhosts} 3"
      ),
      default_threshold=120,
      ld_library_path="/usr/local/fastrak/lib64:/usr/local/nvidia/lib64/"
      )


def get_config(instance_type: str) -> config_pb2.ASeriesConfig:
  if instance_type == "a3-highgpu-8g":
    return _create_a3_config()
  elif instance_type == "a3-megagpu-8g":
    return _create_a3plus_config()
  else:
    raise ValueError(f"Unsupported instance type: {instance_type}")
