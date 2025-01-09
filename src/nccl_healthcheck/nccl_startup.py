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

"""Run the NCCL test.

This module controls execution of pairwise NCCL test.
"""

import os
import re
import time

import checker_common
import config

JOB_NAME = os.environ.get("JOB_NAME")
SERVICE_NAME = os.environ.get("SERVICE_NAME")
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE")
KUBECTL = os.environ.get("KUBECTL_PATH", "/scripts/kubectl")
JOB_INDEX = int(os.getenv("JOB_COMPLETION_INDEX", "-1"))

_NCCL_PRE_RESULT_KEY = "aiinfra/nccl-healthcheck-pre-result"
TAINT_KEY = "aiinfra/nccl-healthcheck"
TAINT_VALUE_SUSPECT = "suspect"
TAINT_VALUE_FAILED = "failed"
TAINT_EFFECT_NOSCHEDULE = "NoSchedule"
TAINT_EFFECT_PREFERNOSCHEDULE = "PreferNoSchedule"

HEALTHCHECK_TIME_LABEL_KEY = "aiinfra/nccl-healthcheck-runtime-sec"
_NCCL_BANDWIDTH_RESULT_KEY = "aiinfra/nccl-healthcheck-bandwidth"

K_ADD_LABEL_FORMAT = "{k} label node %s %s=%s --overwrite".format(k=KUBECTL)
K_TAINT_NODE_FORMAT = "{k} taint node %s %s=%s:%s".format(k=KUBECTL)
K_REMOVE_LABEL_FORMAT = "{k} label node %s %s-".format(k=KUBECTL)
K_REMOVE_TAINT_NODE_FORMAT = "{k} taint node %s %s-".format(k=KUBECTL)
K_DELETE_SERVICE_FORMAT = "{k} delete svc %s".format(k=KUBECTL)

WORKLOAD_TERMINATE_FILE = "/usr/share/nemo/workload_terminated"


def ensure_env_variables() -> None:
  """Ensure necessary environment variables are set."""
  required_envs = [
      "NODE_NAME",
      "NHOSTS",
      "nr",
      "JOB_COMPLETION_INDEX",
      "BANDWIDTH_THRESHOLD",
      "START_MESSAGE_SIZE",
      "END_MESSAGE_SIZE",
      "DRY_RUN",
      "INSTANCE_TYPE",
  ]
  for env in required_envs:
    if env not in os.environ:
      raise ValueError(f"Must set {env}")


def apply_namespace_resolution() -> None:
  """Mitigate AR timeout issue, b/318412074.

  Systemd adds a route to DNS server for avery DHCP NIC even if the NIC
  doesn't have connectivity to it. This removes the duplicate routes.
  """
  for eth_idx in range(9):
    checker_common.run_command(
        "sudo route del -net 169.254.169.254 gw 0.0.0.0"
        " netmask 255.255.255.255 dev eth"
        f" {eth_idx}"
    )


def configure_ssh() -> None:
  """Configure SSH settings."""
  checker_common.run_command("service ssh start")
  with open("/root/.ssh/config", "a") as f:
    f.write("""
Host *
StrictHostKeyChecking no
User root
IdentityFile /root/.ssh/google_compute_engine
Port 222""")
  checker_common.run_command(
      "sed -i 's/#Port 22/Port 222/g' /etc/ssh/sshd_config", check=False
  )


def get_host_list(
    nhosts: int,
) -> list[str]:
  """Generate a hostfile based on host names from pods."""

  hosts = []
  # If there is only one node, assume it is the host and return it.
  if nhosts == 1:
    hosts.append(os.environ["NODE_NAME"])
    return hosts

  for i in range(nhosts):
    pod_name = f"{JOB_NAME}-{i}.{SERVICE_NAME}"
    host_name = get_host_name(pod_name)
    if host_name:
      hosts.append(host_name)
      print(f"Got host information from pod: {pod_name} on host {host_name}")

  return hosts


def create_hostfile(
    hosts: list[str],
    nr: str,
) -> None:
  """Create a hostfile for the NCCL test.

  Args:
    hosts (list[str]): The list of hosts to include in the hostfile.
    nr (str): The number of ranks to use in the hostfile.
  """
  nhosts = len(hosts)
  os.makedirs(f"hostfiles{nhosts}", exist_ok=True)

  nranks = [1, 2, 4, 8]
  for nrank in nranks:
    hostfile = f"hostfiles{nhosts}/hostfile{nrank}"
    with open(hostfile, "w") as f:
      for host in hosts:
        f.write(f"{host} port=222 slots={nr}\n")


def run_nccl_test(
    hosts: list[str],
) -> None:
  """Run the NCCL test."""
  nhosts = len(hosts)
  config_obj = config.get_config(INSTANCE_TYPE)
  if JOB_INDEX == 0:  # master node
    print(f"I am a master that will run nccl test on {nhosts} nodes")
    start_message_size = os.environ["START_MESSAGE_SIZE"] or "2G"
    end_message_size = os.environ["END_MESSAGE_SIZE"] or "8G"
    # Number of times NCCL operation will run in a test
    # TODO - Rename 'ITERATIONS' to be more descriptive & clear
    nccl_operation_iterations = int(os.getenv("ITERATIONS", "300"))
    # Number of times the test will run (a bandwidth for each)
    test_iterations = int(os.getenv("TEST_ITERATIONS", "5"))
    benchmark = os.environ.get("BENCHMARK", "all_gather_perf")
    ld_library_path = config_obj.ld_library_path

    print("Sleeping for 30 seconds to let rxdm spin up...")
    time.sleep(30)

    bandwidths = []
    # Run the test 'iter' amount of times and average the performance.
    for _ in range(test_iterations):
      test_result = checker_common.run_command(
          config_obj.nccl_test_command_template.format(
              ld_library_path=ld_library_path,
              start_message_size=start_message_size,
              end_message_size=end_message_size,
              nhosts=nhosts,
              iterations=nccl_operation_iterations,
              benchmark=benchmark,
          )
      )
      bandwidths.append(get_bandwidth(test_result.stdout))
    process_test_result(
        bandwidths=bandwidths,
        nodes=hosts,
        bandwidth_threshold=int(os.environ["BANDWIDTH_THRESHOLD"]),
    )
  else:  # secondary nodes
    while not os.path.exists("/master.done"):
      print("waiting for master pod...")
      time.sleep(10)
  # Create file to let tcpxo daemon to terminate, this only applies to A3
  # and A3+ machines which use rxdm.
  if INSTANCE_TYPE in (
      "a3-highgpu-8g",
      "a3-megagpu-8g",
      "a3-megagpu-8g-debian",
  ):
    with open(WORKLOAD_TERMINATE_FILE, "w") as _:
      pass


def process_test_result(
    bandwidths: list[int],
    nodes: list[str],
    bandwidth_threshold: int,
    acceptable_failure_rate: float = 0.5,
) -> None:
  """Process test results. Add node taints and labels."""
  second_pass = os.environ.get("SECOND_PASS", "").lower() == "true"

  # Filter for only valid bandwidths
  valid_bandwidths: tuple[int, ...] = tuple(bw for bw in bandwidths if bw != -1)
  iteration_failure_rate: float = 1.0 - len(valid_bandwidths) / len(bandwidths)

  # Average bandwidth is -1 if there are no valid bandwidths
  # Average bandwidth is rounded up to nearest integer
  avg_bandwidth: int = (
      sum(valid_bandwidths) // len(valid_bandwidths) if valid_bandwidths else -1
  )
  has_sufficient_bandwidth: bool = avg_bandwidth >= bandwidth_threshold
  has_acceptable_failure_rate: bool = (
      iteration_failure_rate <= acceptable_failure_rate
  )

  passed: bool = has_sufficient_bandwidth and has_acceptable_failure_rate

  test_name = os.environ.get("TEST_NAME", "nccl")
  for node in nodes:
    add_healthcheck_time_label(node)
    mark_node_bandwidth(node, avg_bandwidth)

    # Either it passed or a second pass is needed
    terminal = second_pass or passed
    checker_common.log_results(
        test_name=test_name,
        passed=passed,
        node_name=node,
        workflow_id=os.environ.get("WORKFLOW_ID"),
        result_data={
            "avg_bus_bandwidth": avg_bandwidth,
            "num_nodes": len(nodes),
            "all_nodes": sorted(nodes),
            "terminal_test": terminal,
        },
    )
    # After reaching end of its set of test sweeps (such as second pass)
    if terminal:
      result = "fail"
      if passed:
        result = "pass"
      elif avg_bandwidth == -1:
        result = "crash"

      # Pre-result label is used to determine if this run met criteria
      checker_common.add_label(
          node,
          _NCCL_PRE_RESULT_KEY,
          result,
          K_ADD_LABEL_FORMAT,
      )


def mark_failed_node(
    node: str,
    nodes: list[str],
    taint_value: str,
    taint_effect: str,
) -> None:
  """Mark a node as failed."""
  other_nodes = ",".join([n for n in nodes if n != node])
  taint_node(node, TAINT_KEY, taint_value, taint_effect)
  checker_common.add_label(
      node, "aiinfra/nccl-healthcheck", other_nodes, K_ADD_LABEL_FORMAT
  )


def get_bandwidth(
    test_result: str,
) -> int:
  """Extract the bandwidth from the test result.

  Args:
    test_result (str): The test result to extract the bandwidth from.

  Returns:
    int: The bandwidth (GB/s)extracted from the test result. -1 if not found.
  """
  # Search for the line of interest using regex
  match = re.search(r"# Avg bus bandwidth\s*:\s*(\d+)", test_result)

  # Extract the number if the pattern was found
  if match:
    bandwidth = int(match.group(1))
    print(f"Found bandwidth: {bandwidth}")
    return bandwidth
  else:
    print("Bandwidth not found in log.")
    return -1


def get_host_name(
    pod_name: str,
) -> str:
  """Retrieve the host name where the specified pod is running.

  Args:
    pod_name (str): The name of the pod for which the host name is to be
      retrieved.

  Returns:
  str: The host name where the pod is running.
  """
  start_time = int(time.time())

  while timeout_check(start_time, pod_name):
    result = checker_common.run_command(
        f"ssh {pod_name} -p 222 -- cat /host.name",
        check=False,
    )
    if result.returncode == 0:
      return result.stdout
    time.sleep(1)

  return ""


def timeout_check(
    start_time: float,
    pod_name: str,
    timeout_minutes: int = 10,
) -> bool:
  """Check if we exceed allocated timeout to get host name from that pod_name.

  Args:
    start_time (int): Time when we start checking the pod.
    pod_name (str): The name of the pod to be checked.
    timeout_minutes (int): The timeout in minutes.

  Returns:
    None

  Raises:
    TimeoutError: If the pod has been running longer than the allocated timeout.
  """
  elapsed_time = time.time() - start_time
  if elapsed_time >= 60 * timeout_minutes:  # in seconds
    error_message: str = (
        f"{timeout_minutes}min Timeout reached"
        f" while trying to ssh to pod {pod_name}"
    )
    print(f"Timeout Error: {error_message}")
    print(f"node calling from is: {os.environ.get('NODE_NAME')}")
    raise TimeoutError(error_message)
  return True


def taint_node(
    node_name: str,
    key: str,
    value: str,
    effect: str,
) -> None:
  """Apply a taint to a specified node with given key, value, and effect.

  Args:
    node_name (str): The name of the node to be tainted.
    key (str): The taint key to be set.
    value (str): The taint value to be set.
    effect (str): The effect of the taint (e.g., "NoExecute", "NoSchedule").
  """
  print("adding taint %s=%s to node %s" % (key, value, node_name))
  if os.environ.get("DRY_RUN") != "true":
    checker_common.run_command(
        K_TAINT_NODE_FORMAT % (node_name, key, value, effect)
    )


def remove_node_taint(
    node_name: str,
    taint_key: str,
) -> None:
  print("removing taint %s from node %s" % (taint_key, node_name))
  checker_common.run_command(
      K_REMOVE_TAINT_NODE_FORMAT % (node_name, taint_key)
  )


def add_healthcheck_time_label(
    node_name: str,
    node_label: str = HEALTHCHECK_TIME_LABEL_KEY,
    time_value: int | None = None,
) -> None:
  """Add healthcheck time label to node."""
  if time_value is None:
    time_value = int(time.time())
  checker_common.add_label(
      node_name,
      node_label,
      f"{time_value}",
      K_ADD_LABEL_FORMAT,
  )


def mark_node_bandwidth(
    node: str,
    bandwidth: int,
) -> None:
  """Mark the node bandwidth observed during the test.

  Args:
    node (str): The name of the node to be labeled.
    bandwidth (int): The bandwidth seen in the test to use as the label value.
      Bandwidth will be set to 'None' if the test fails.
  """
  if bandwidth == -1:
    bandwidth = None
  checker_common.add_label(
      node,
      _NCCL_BANDWIDTH_RESULT_KEY,
      f"{str(bandwidth):>02}",
      K_ADD_LABEL_FORMAT,
  )


def remove_label(
    node_name: str,
    label: str,
) -> None:
  print("removing label %s from node %s" % (label, node_name))
  checker_common.run_command(K_REMOVE_LABEL_FORMAT % (node_name, label))


def cleanup(
    hosts: list[str],
) -> None:
  """Clean up any additional resources deployed to the cluster."""
  print("Running cleanup commands.")
  if JOB_INDEX == 0:
    for i in range(len(hosts)):
      checker_common.run_command(
          f"ssh {JOB_NAME}-{i}.{SERVICE_NAME} -p 222 -- touch /master.done",
      )
    checker_common.run_command(K_DELETE_SERVICE_FORMAT % (SERVICE_NAME))


def main() -> None:
  """Main function."""
  ensure_env_variables()
  apply_namespace_resolution()
  configure_ssh()

  nhosts = int(os.environ["NHOSTS"])
  nr = os.environ["nr"]
  node_name = os.environ["NODE_NAME"]
  with open("/host.name", "w") as f:
    f.write(node_name)
  hosts = get_host_list(nhosts)
  create_hostfile(hosts, nr)
  run_nccl_test(hosts)
  cleanup(hosts)

  print("my job is done")


if __name__ == "__main__":
  main()
