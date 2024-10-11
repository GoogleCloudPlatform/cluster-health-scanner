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

"""Runs the NCCL test.

This module control execution of pairwise NCCL test.
"""

import json
import os
import re
import time
from typing import Callable, Dict, List

import checker_common
import metrics

JOB_NAME = os.getenv("JOB_NAME")
SERVICE_NAME = os.getenv("SERVICE_NAME")
POD_NAME = os.getenv("POD_NAME")

_RESULT_LABEL_KEY = "aiinfra/neper-healthcheck-result"
TAINT_KEY = "aiinfra/neper-healthcheck"
TAINT_EFFECT = "NoSchedule"

HEALTHCHECK_TIME_LABEL_KEY = "aiinfra/neper-healthcheck-runtime-sec"

K_ADD_LABEL_FORMAT = "/scripts/kubectl label node %s %s=%s --overwrite"
K_TAINT_NODE_FORMAT = "/scripts/kubectl taint node %s %s=%s:%s"
K_REMOVE_LABEL_FORMAT = "/scripts/kubectl label node %s %s-"
K_REMOVE_TAINT_NODE_FORMAT = "/scripts/kubectl taint node %s %s-"


def ensure_env_variables() -> None:
  """Ensure necessary environment variables are set."""
  required_envs = [
      "NODE_NAME",
      "NODE_IP",
      "GOOD_THROUGHPUT",
      "HEALTH_VALIDITY_HOURS",
      "POD_NAME",
      "JOB_NAME",
      "SERVICE_NAME",
      "DRY_RUN",
  ]
  for env in required_envs:
    if env not in os.environ:
      raise ValueError(f"Must set {env}")
    print("ENV %s=%s" % (env, os.environ[env]))


def configure_ssh() -> None:
  """Configures SSH settings."""
  checker_common.run_command(
      "sed -i 's/#Port 22/Port 222/g' /etc/ssh/sshd_config", check=False
  )
  checker_common.run_command("service ssh restart")
  with open("/root/.ssh/config", "a") as f:
    f.write("""
Host *
StrictHostKeyChecking no
User root
IdentityFile /root/.ssh/google_compute_engine
Port 222""")


def get_host_to_ips() -> Dict[str, List[str]]:
  """Generates a hostfile based on host names from pods."""

  hosts = {}
  pod_name = f"{JOB_NAME}-1.{SERVICE_NAME}"
  host_name = get_host_name(pod_name)
  raw_ip_addresses = get_ip_addresses(pod_name)

  ip_addresses = raw_ip_addresses.strip().split("\n")

  if host_name and ip_addresses:
    hosts[host_name] = ip_addresses
    print(f"Got host information from pod: {pod_name} on host {host_name}")
    print(f"Got ip information from pod: {pod_name} on ip {ip_addresses}")

  return hosts


def run_neper_test(
    hosts_to_ips: Dict[str, List[str]]
) -> List[Callable[[], None]]:
  """Runs the Neper test."""

  def cleanup_delete_temp_files(regex: str) -> Callable[[], None]:
    def delete_file() -> None:
      print(f"Deleting temporary files with path: {regex}")
      checker_common.run_command(f"rm {regex}")

    return delete_file

  cleanup_functions = []

  if f"{JOB_NAME}-0" in POD_NAME:  # master node
    print("I am a master that will run neper test on 2 nodes")

    self_host = checker_common.run_command("cat /host.name").stdout
    log_files = []

    for host, ips in hosts_to_ips.items():
      print(f"Host: {host}")
      count = 0
      for dst_ip in ips:
        count += 1
        log_file = f"/tmp/{self_host}_{host}_eth{count}.log"
        log_files.append(log_file)
        checker_common.run_command(
            f"taskset -c 17-32 /scripts/tcp_stream -rw --client -H '{dst_ip}' "
            "--skip-rx-copy --num-threads=16 --num-flows=200 "
            f"--suicide-length=600 --test-length=30 > '{log_file}' 2>&1"
        )
        cleanup_functions.append(cleanup_delete_temp_files(log_file))

        checker_common.run_command(
            f"ssh {JOB_NAME}-1.{SERVICE_NAME} -p 222 -- touch"
            f" /master{count}.done",
        )
      process_test_result(log_files, self_host, host)
  else:  # secondary nodes
    for _, ips in hosts_to_ips.items():
      count = 0
      for dst_ip in ips:
        count += 1
        print(f"spinning up neper server for {dst_ip}...")
        checker_common.run_command(
            "taskset -c 17-32 /scripts/tcp_stream -rw --skip-rx-copy "
            "--num-threads=16 --num-flows=200 --suicide-length=600 "
            "--test-length=30 &"
        )
        while not os.path.exists(f"/master{count}.done"):
          print(f"test for {dst_ip} not done")
          time.sleep(10)
        print(f"test for {dst_ip} is done")
        cleanup_functions.append(
            cleanup_delete_temp_files(f"/master{count}.done")
        )

  return cleanup_functions


def process_test_result(
    log_files: List[str], local_host: str, remote_host: str
) -> None:
  """Analyze the log files and add taints to the nodes that yield bad throughput."""

  threshold = int(os.environ["GOOD_THROUGHPUT"])
  count = 0
  local_test_failed = False
  remote_test_failed = False
  local_throughput_by_eth = {}
  remote_throughput_by_eth = {}
  for log_file in log_files:
    count += 1
    local_throughput = get_throughput(log_file, local=True)
    remote_throughput = get_throughput(log_file, local=False)

    local_throughput_by_eth[f"eth{count}"] = local_throughput
    remote_throughput_by_eth[f"eth{count}"] = remote_throughput

    if local_throughput < threshold:
      local_test_failed = True
      print(
          f"local host {local_host} failed the neper test at eth{count} with"
          f" throughput {local_throughput}. Adding node taints..."
      )
      checker_common.add_label(
          local_host,
          f"{TAINT_KEY}_eth{count}",
          f"{local_throughput}",
          K_ADD_LABEL_FORMAT,
      )
    else:
      remove_label(local_host, f"{TAINT_KEY}_eth{count}")

    if remote_throughput < threshold:
      remote_test_failed = True
      print(
          f"remote host {remote_host} failed the neper test at eth{count} with"
          f" throughput {remote_throughput}. Adding node taints..."
      )
      checker_common.add_label(
          remote_host,
          f"{TAINT_KEY}_eth{count}",
          f"{remote_throughput}",
          K_ADD_LABEL_FORMAT,
      )
    else:
      # Removing taints and labels.
      remove_label(remote_host, f"{TAINT_KEY}_eth{count}")

  apply_fail_label(local_test_failed, local_host, remote_host)
  apply_fail_label(remote_test_failed, remote_host, local_host)

  add_healthcheck_time_label(local_host)
  add_healthcheck_time_label(remote_host)

  server_client = {"server": local_host, "client": remote_host}
  local_result = metrics.log_dict(
      test_name="neper",
      did_pass=not local_test_failed,
      node_name=local_host,
      result_data={
          "throughput_by_eth": local_throughput_by_eth,
          "server_client": server_client,
      },
  )
  print(json.dumps(local_result))
  remote_result = metrics.log_dict(
      test_name="neper",
      did_pass=not remote_test_failed,
      node_name=remote_host,
      result_data={
          "throughput_by_eth": remote_throughput_by_eth,
          "server_client": server_client,
      },
  )
  print(json.dumps(remote_result))
  checker_common.add_label(
      local_host,
      _RESULT_LABEL_KEY,
      "pass" if not local_test_failed and not remote_test_failed else "fail",
      K_ADD_LABEL_FORMAT,
  )


def get_throughput(log_file: str, local: bool) -> int:
  """Get local/remote throughput number from a log file."""

  with open(log_file, "r") as f:
    log_output = f.read()

    remote_throughput_match = re.search(r"remote_throughput=(\d+)", log_output)
    local_throughput_match = re.search(r"local_throughput=(\d+)", log_output)

    if local and local_throughput_match:
      local_throughput = int(local_throughput_match.group(1))
      return local_throughput
    if not local and remote_throughput_match:
      remote_throughput = int(remote_throughput_match.group(1))
      return remote_throughput

    return -1


def get_ip_addresses(pod_name: str) -> str:
  """Retrieve the host name where the specified pod is running.

  Args:
    pod_name (str): The name of the pod for which the host name is to be
      retrieved.

  Returns:
  str: The host name where the pod is running.
  """
  start_time = time.time()

  while timeout_check(start_time, pod_name):
    result = checker_common.run_command(
        f"ssh {pod_name} -p 222 -- cat /tmp/ip_addrs",
        check=False,
    )
    if result.returncode == 0:
      return result.stdout
    time.sleep(1)

  return ""


def get_host_name(pod_name: str) -> str:
  """Retrieve the host name where the specified pod is running.

  Args:
    pod_name (str): The name of the pod for which the host name is to be
      retrieved.

  Returns:
  str: The host name where the pod is running.
  """
  start_time = time.time()
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
    start_time (float): Time when we start checking the pod.
    pod_name (str): The name of the pod to be checked.

  Returns:
    bool

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
  if os.environ.get("DRY_RUN") != "true":
    print("adding taint %s=%s to node %s" % (key, value, node_name))
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
  checker_common.add_label(
      node_name,
      HEALTHCHECK_TIME_LABEL_KEY,
      f"{int(time.time())}",
      K_ADD_LABEL_FORMAT,
  )


def apply_fail_label(check_failed: bool, node_name: str, value: str) -> None:
  if check_failed:
    taint_node(node_name, TAINT_KEY, "failed", TAINT_EFFECT)
    checker_common.add_label(node_name, TAINT_KEY, value, K_ADD_LABEL_FORMAT)
  else:
    remove_node_taint(node_name, TAINT_KEY)
    remove_label(node_name, TAINT_KEY)


def remove_label(node_name: str, label: str) -> None:
  print("removing label %s from node %s" % (label, node_name))
  checker_common.run_command(K_REMOVE_LABEL_FORMAT % (node_name, label))


def main() -> None:
  """Main function."""
  ensure_env_variables()
  configure_ssh()

  node_name = os.environ["NODE_NAME"]
  with open("/host.name", "w") as f:
    f.write(node_name)

  host_to_ips = get_host_to_ips()
  cleanup_funcs = run_neper_test(host_to_ips)

  print("my job is done, running cleanups...")

  for cleanup in cleanup_funcs:
    cleanup()
  print("cleanups are done... exiting...")


if __name__ == "__main__":
  main()
