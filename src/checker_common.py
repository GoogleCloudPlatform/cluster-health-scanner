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

"""Common functions shared between health checkers."""

from collections.abc import Callable
import os
import string
import subprocess
import tempfile
import time
import uuid


K_APPLY_FORMAT = "%s apply -f %s"
K_DELETE_FORMAT = "%s delete -f %s"


def add_label(
    node_name: str, label: str, value: str, label_format: str
) -> None:
  """Adds a label to a node.

  Args:
    node_name (str): Name of the node.
    label (str): label being set.
    value (str): value being set.
    label_format (str): Interpolation format accepting node_name, label, value.

  Returns:
    None.
  """
  print("adding label %s=%s to node %s" % (label, value, node_name))
  run_command(label_format % (node_name, label, value))


def run_command(
    command: str, check: bool = False
) -> subprocess.CompletedProcess[str]:
  """Execute a shell command using subprocess.

  Args:
    command (str): The shell command to be executed.
    check (bool, optional): If True, raises CalledProcessError if the command
      returns a non-zero exit status. Defaults to True.

  Returns:
    subprocess.CompletedProcess: The result object containing information about
    the completed process.
  """

  print("running: %s" % command)
  start_time = time.time()
  diag = subprocess.run(
      command,
      shell=True,
      text=True,
      check=check,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
  )
  print(
      "took: %s seconds\nout: %s, err: %s"
      % (time.time() - start_time, diag.stdout, diag.stderr)
  )
  return diag


def create_k8s_objects(
    yaml_path: str, kubectl_path: str
) -> list[Callable[[], subprocess.CompletedProcess[str]]]:
  """Expands provided yaml file and runs `kubectl apply -f` on the contents."""

  cleanup_functions = []

  expanded_yaml_content = expand_template(yaml_path)
  with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
    file_name = f.name
    f.write(expanded_yaml_content)

  cleanup_functions.append(apply_yaml_file(file_name, kubectl_path))
  return cleanup_functions


def apply_yaml_file(
    yaml_path: str, kubectl_path: str
) -> Callable[[], subprocess.CompletedProcess[str]]:
  """Applies YAML file.

  Args:
    yaml_path (str): Relative filesystem path to the yaml file to apply.
    kubectl_path (str): Relative filesystem path to the kubectl binary.

  Returns:
    Callable((), subprocess.CompletedProcess(str)): A function that will run
    `kubectl delete -f` on the yaml_path provided for easy cleanup of temporary
    resources.
  """
  run_command(K_APPLY_FORMAT % (kubectl_path, yaml_path))

  def delete_yaml_file():
    return run_command(K_DELETE_FORMAT % (kubectl_path, yaml_path))

  return delete_yaml_file


def expand_template(yaml_template: str) -> str:
  """Expands YAML template."""
  with open(yaml_template, "r") as f:
    t = string.Template(f.read())
    return t.safe_substitute({
        "CHECK_TIME_EPOCH_SEC": int(time.time()),
        "DRY_RUN": os.environ.get("DRY_RUN"),
        "ORIG_CHECK_TIME_EPOCH_SEC": os.environ.get("CHECK_TIME_EPOCH_SEC"),
        "R_LEVEL": os.environ.get("R_LEVEL"),
        "IMAGE_TAG": os.environ.get("IMAGE_TAG", "latest"),
        "SHORT_GUID": os.environ.get("SHORT_GUID", str(uuid.uuid4())[:4]),
        "ITERATIONS": os.environ.get("ITERATIONS", 5),
    })
