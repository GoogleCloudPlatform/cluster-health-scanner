"""Run the NCCL test.

This module controls execution of pairwise NCCL test.
"""

import json
import os
import re
import time
from typing import List

import checker_common
import metrics
import config

JOB_NAME = os.environ.get("JOB_NAME")
SERVICE_NAME = os.environ.get("SERVICE_NAME")
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE")
KUBECTL = os.environ.get("KUBECTL_PATH", "/scripts/kubectl")
JOB_INDEX = int(os.getenv("JOB_COMPLETION_INDEX", "-1"))

_RESULT_LABEL_KEY = "aiinfra/nccl-healthcheck-result"
TAINT_KEY = "aiinfra/nccl-healthcheck"
TAINT_VALUE_SUSPECT = "suspect"
TAINT_VALUE_FAILED = "failed"
TAINT_EFFECT_NOSCHEDULE = "NoSchedule"
TAINT_EFFECT_PREFERNOSCHEDULE = "PreferNoSchedule"

HEALTHCHECK_TIME_LABEL_KEY = "aiinfra/nccl-healthcheck-valid-till-sec"
HEALTHCHECK_SECOND_PASS_NEEDED_LABEL_KEY = (
    "aiinfra/nccl-healthcheck-second-pass-needed"
)

K_ADD_LABEL_FORMAT = "{k} label node %s %s=%s --overwrite".format(k=KUBECTL)
K_TAINT_NODE_FORMAT = "{k} taint node %s %s=%s:%s".format(k=KUBECTL)
K_REMOVE_LABEL_FORMAT = "{k} label node %s %s-".format(k=KUBECTL)
K_REMOVE_TAINT_NODE_FORMAT = "{k} taint node %s %s-".format(k=KUBECTL)
K_DELETE_SERVICE_FORMAT = "{k} delete svc %s".format(k=KUBECTL)


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
      "JOB_NAME",
      "SERVICE_NAME",
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


def get_host_list(nhosts: int) -> List[str]:
  """Generate a hostfile based on host names from pods."""

  hosts = []
  for i in range(nhosts):
    pod_name = f"{JOB_NAME}-{i}.{SERVICE_NAME}"
    host_name = get_host_name(pod_name)
    if host_name:
      hosts.append(host_name)
      print(f"Got host information from pod: {pod_name} on host {host_name}")

  return hosts


def create_hostfile(hosts: List[str], nr: str) -> None:
  nhosts = len(hosts)
  os.makedirs(f"hostfiles{nhosts}", exist_ok=True)

  nranks = [1, 2, 4, 8]
  for nrank in nranks:
    hostfile = f"hostfiles{nhosts}/hostfile{nrank}"
    with open(hostfile, "w") as f:
      for host in hosts:
        f.write(f"{host} port=222 slots={nr}\n")


def run_nccl_test(hosts: List[str]) -> None:
  """Run the NCCL test."""
  nhosts = len(hosts)
  config_obj = config.get_config(INSTANCE_TYPE)
  if JOB_INDEX == 0:  # master node
    print(f"I am a master that will run nccl test on {nhosts} nodes")
    start_message_size = os.environ["START_MESSAGE_SIZE"] or "2G"
    end_message_size = os.environ["END_MESSAGE_SIZE"] or "8G"
    ld_library_path = config_obj.ld_library_path

    print("Sleeping for 30 seconds to let rxdm spin up...")
    time.sleep(30)

    bandwidths = []
    iterations = int(os.getenv("ITERATIONS", "5"))
    for _ in range(iterations):
      # Run the test 'iter' amount of times and average the performance.
      test_result = checker_common.run_command(
          config_obj.nccl_test_command_template.format(
              ld_library_path=ld_library_path,
              start_message_size=start_message_size,
              end_message_size=end_message_size,
              nhosts=nhosts,
          )
      )
      bandwidths.append(get_bandwidth(test_result.stdout))
    process_test_result(bandwidths, hosts)
  else:  # secondary nodes
    while not os.path.exists("/master.done"):
      print("waiting for master pod...")
      time.sleep(10)
  # Create file to let tcpd daemon to terminate
  with open("/usr/share/nemo/workload_terminated", "w") as _:
    pass


def process_test_result(bandwidths: List[int], nodes: List[str]) -> None:
  """Process test results. Add node taints and labels."""
  threshold = int(os.environ["BANDWIDTH_THRESHOLD"])
  enable_two_pass = (
      os.environ.get("ENABLE_TWO_PASS_STRATEGY", "").lower() == "true"
  )
  second_pass = os.environ.get("SECOND_PASS", "").lower() == "true"
  failures = 0
  total_bandwidth = 0
  for bandwidth in bandwidths:
    if bandwidth == -1:
      failures += 1
      continue
    total_bandwidth += bandwidth

  iterations = len(bandwidths) - failures
  avg_bandwidth = int(total_bandwidth / iterations) if iterations > 0 else -1

  # If more than half of the iterations failed or the average bandwidth is
  # below the threshold, the test is considered failed.
  passed = (
      failures <= iterations / 2
      and avg_bandwidth >= threshold
  )

  if passed:
    print("nccl test passed. Removing node taints...")
    taint = None
    # removing healthcheck label and taint
    for node in nodes:
      remove_node_taint(node, TAINT_KEY)
      remove_label(node, "aiinfra/nccl-healthcheck")
      if second_pass:
        remove_label(node, HEALTHCHECK_SECOND_PASS_NEEDED_LABEL_KEY)
  elif enable_two_pass:
    print("ENABLE_TWO_PASS_STRATEGY is set.")
    if second_pass:
      # failed with two pass enabled
      print("nccl test failed final pass. Adding node taint to Master Node...")
      taint = TAINT_VALUE_FAILED
      node = os.environ.get("NODE_NAME")
      mark_failed_node(
          node,
          nodes,
          TAINT_VALUE_FAILED,
          TAINT_EFFECT_NOSCHEDULE,
      )
      remove_label(node, HEALTHCHECK_SECOND_PASS_NEEDED_LABEL_KEY)
    else:
      # failed with two pass disabled
      print("nccl test failed first pass. Adding node taints...")
      taint = TAINT_VALUE_SUSPECT
      for node in nodes:
        mark_failed_node(
            node,
            nodes,
            TAINT_VALUE_SUSPECT,
            TAINT_EFFECT_PREFERNOSCHEDULE,
        )
        time.sleep(2)  # Sleep to force a different check-time
        deploy_second_pass(node)
  else:
    # failed and two pass disabled
    print("ENABLE_TWO_PASS_STRATEGY is not set.")
    print("nccl test failed. Adding node taint to nodes...")
    taint = TAINT_VALUE_FAILED
    for node in nodes:
      mark_failed_node(
          node,
          nodes,
          TAINT_VALUE_FAILED,
          TAINT_EFFECT_NOSCHEDULE,
      )

  for node in nodes:
    add_healthcheck_time_label(node)
    mark_node_bandwidth(node, avg_bandwidth)

    if second_pass and node != os.environ.get("NODE_NAME"):
      # Do not log metric if we're testing against a known-good node
      continue
    terminal = taint != TAINT_VALUE_SUSPECT
    log = metrics.log_dict(
        test_name="nccl",
        did_pass=passed,
        node_name=node,
        result_data={
            "avg_bus_bandwidth": avg_bandwidth,
            "num_nodes": len(nodes),
            "all_nodes": sorted(nodes),
            "taint_applied": taint,
            "terminal_test": terminal,
        },
    )
    if terminal:
      checker_common.add_label(
          node,
          _RESULT_LABEL_KEY,
          "pass" if passed else "fail",
          K_ADD_LABEL_FORMAT,
      )

    print(json.dumps(log))


def mark_failed_node(
    node: str,
    nodes: List[str],
    taint_value: str,
    taint_effect: str,
) -> None:
  """Mark a node as failed."""
  other_nodes = ",".join([n for n in nodes if n != node])
  taint_node(node, TAINT_KEY, taint_value, taint_effect)
  checker_common.add_label(
      node, "aiinfra/nccl-healthcheck", other_nodes, K_ADD_LABEL_FORMAT
  )


def deploy_second_pass(node: str) -> None:
  print("deploying second pass...")
  config_obj = config.get_config(INSTANCE_TYPE)
  checker_common.add_label(
      node, HEALTHCHECK_SECOND_PASS_NEEDED_LABEL_KEY, "true", K_ADD_LABEL_FORMAT
  )
  yaml_path = os.path.join("/scripts", config_obj.second_pass_yaml_path)
  checker_common.create_k8s_objects(yaml_path, KUBECTL)


def get_bandwidth(test_result: str) -> int:
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


def get_host_name(pod_name: str) -> str:
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


def timeout_check(start_time: float, pod_name: str) -> bool:
  """Check if we exceed allocated timeout to get host name from that pod_name.

  Args:
    start_time (int): Time when we start checking the pod.
    pod_name (str): The name of the pod to be checked.

  Returns:
    None

  Raises:
    TimeoutError: If the pod has been running longer than the allocated timeout.
  """
  elapsed_time = time.time() - start_time
  if elapsed_time >= 10 * 60:  # 10 minutes
    raise TimeoutError(
        f"10min Timeout reached while trying to ssh to pod {pod_name}"
    )
  return True


def taint_node(node_name: str, key: str, value: str, effect: str) -> None:
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


def remove_node_taint(node_name: str, taint_key: str) -> None:
  print("removing taint %s from node %s" % (taint_key, node_name))
  checker_common.run_command(
      K_REMOVE_TAINT_NODE_FORMAT % (node_name, taint_key)
  )


def add_healthcheck_time_label(node_name: str) -> None:
  """Add healthcheck time label to node."""
  # Add timestampt as number of seconds
  # since epoch time January 1, 1970, 00:00:00 (UTC) + 5 (default) hours
  # health validity.
  health_validity = (
      int(time.time())
      + int(os.environ.get("HEALTH_VALIDITY_HOURS", "5")) * 60 * 60
  )
  checker_common.add_label(
      node_name,
      HEALTHCHECK_TIME_LABEL_KEY,
      f"{health_validity}",
      K_ADD_LABEL_FORMAT,
  )


def mark_node_bandwidth(node: str, bandwidth: int) -> None:
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
      "aiinfra/nccl-healthcheck-bandwidth",
      f"{str(bandwidth):>02}",
      K_ADD_LABEL_FORMAT,
  )


def remove_label(node_name: str, label: str) -> None:
  print("removing label %s from node %s" % (label, node_name))
  checker_common.run_command(K_REMOVE_LABEL_FORMAT % (node_name, label))


def cleanup(hosts: List[str]) -> None:
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
