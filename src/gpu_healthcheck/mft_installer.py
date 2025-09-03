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

"""Installs Mellanox Firmware Tools (MFT) on A4/A4x machines.

This module handles the process of downloading and installing the necessary
MFT kernel modules and userspace tools from GCS and Mellanox's website.
"""

import glob
import os
import re
import shutil
import subprocess

import google.cloud.storage
from packaging import version
import requests

import checker_common

# --- MFT Installation Constants ---
METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/"
METADATA_HEADERS = {"Metadata-Flavor": "Google"}
LSB_RELEASE_PATH = "/etc_host/lsb-release"
LOCAL_MODULE_PATH = "/tmp/mft_kernel_modules"
MFT_INSTALL_EXTRACT_DIR = "/tmp/mft_install"


def is_a4_or_a4x_machine(node_name: str) -> bool:
  """Check if the node is an A4 or A4x machine type using a kubectl jsonpath query."""
  print(f"Checking machine type for node: {node_name}")
  try:
    # Command to get the machine family from the node name.
    cmd = (
        f"/app/kubectl get node {node_name} -o "
        "jsonpath='{.metadata.labels.cloud\\.google\\.com/machine-family}'"
    )

    result = checker_common.run_command(cmd)
    machine_family = result.stdout.strip()
    print(f"Detected machine family: '{machine_family}'")

    return machine_family in ["a4", "a4x"]

  except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
    print(f"Could not determine machine type for {node_name}: {e}")
    return False


def _get_gcs_artifact_path() -> tuple[str, str]:
  """Determines the regional GCS artifact path from the VM's metadata and lsb-release file.

  Returns:
      A tuple containing the (bucket_name, bucket_path).

  Raises:
      requests.exceptions.RequestException: If there is an issue fetching
          metadata.
      FileNotFoundError: If the LSB_RELEASE_PATH file does not exist.
      ValueError: If the expected artifact location key is not found in the
          lsb-release file.
  """
  try:
    response = requests.get(
        METADATA_URL + "instance/zone", headers=METADATA_HEADERS
    )
    response.raise_for_status()
    zone = response.text.split("/")[-1]
  except requests.exceptions.RequestException as e:
    print(f"Error getting instance zone: {e}")
    raise

  if zone.startswith("asia"):
    key = "ARTIFACTS_LOCATION_ASIA"
  elif zone.startswith("europe"):
    key = "ARTIFACTS_LOCATION_EU"
  else:
    key = "ARTIFACTS_LOCATION_US"
  print(f"Using GCS location key for your region: '{key}'")

  if not os.path.exists(LSB_RELEASE_PATH):
    raise FileNotFoundError(f"{LSB_RELEASE_PATH} not found.")

  with open(LSB_RELEASE_PATH, "r") as f:
    lsb_content = f.read()

  match = re.search(f"^{key}=(.*)$", lsb_content, re.MULTILINE)
  if not match:
    raise ValueError(f"Key {key} not found in {LSB_RELEASE_PATH}")

  gcs_path = match.group(1).replace("gs://", "")
  bucket_name = gcs_path.split("/")[0]
  bucket_path = "/".join(gcs_path.split("/")[1:])
  return bucket_name, bucket_path


def _find_latest_mft_blob(
    bucket_name: str, bucket_path: str
) -> google.cloud.storage.Blob:
  """Finds the MFT kernel module blob with the highest version number in GCS."""
  storage_client = google.cloud.storage.Client()
  bucket = storage_client.bucket(bucket_name)
  search_prefix = f"{bucket_path}/mft-kernel-modules-"
  all_blobs = list(bucket.list_blobs(prefix=search_prefix))

  if not all_blobs:
    raise FileNotFoundError(
        f"No MFT kernel modules found at gs://{bucket_name}/{search_prefix}"
    )

  version_regex = re.compile(r"mft-kernel-modules-([\d.]+-\d+)")

  def version_key(filename):
    match = version_regex.search(filename)
    return version.parse(match.group(1)) if match else version.parse("0")

  latest_mft_blob = max(all_blobs, key=lambda blob: version_key(blob.name))
  return latest_mft_blob


def _download_and_extract_blob(blob: google.cloud.storage.Blob):
  """Downloads a GCS blob to the local module path and extracts it."""
  os.makedirs(LOCAL_MODULE_PATH, exist_ok=True)
  mft_tgz_filename = os.path.basename(blob.name)
  local_tgz_path = os.path.join(LOCAL_MODULE_PATH, mft_tgz_filename)

  print(
      f"Downloading gs://{blob.bucket.name}/{blob.name} to {local_tgz_path}..."
  )
  blob.download_to_filename(local_tgz_path)

  if not os.path.exists(local_tgz_path):
    raise FileNotFoundError(f"Downloaded file not found at {local_tgz_path}")

  checker_common.run_command(
      f"tar -xzvf {local_tgz_path} -C {LOCAL_MODULE_PATH} --strip-components=2"
  )


def download_mft_kernel_modules():
  """Finds, downloads, and extracts the latest MFT kernel modules from GCS."""
  print("--- [Step 2/5] Finding and Downloading Kernel Modules ---")

  # 1. Find the path
  bucket_name, bucket_path = _get_gcs_artifact_path()
  print(f"Searching for modules in gs://{bucket_name}/{bucket_path}...")

  # 2. Find the latest versioned file in that path
  latest_mft_blob = _find_latest_mft_blob(bucket_name, bucket_path)
  print(f"Success: Found latest MFT package: {latest_mft_blob.name}")

  # 3. Download and extract it
  _download_and_extract_blob(latest_mft_blob)

  print("SUCCESS: Kernel modules downloaded and extracted.")
  checker_common.run_command(f"ls -l {LOCAL_MODULE_PATH}")


def load_mft_kernel_modules():
  """Loads the MFT kernel modules."""
  print("--- [Step 3/5] Inserting Kernel Modules ---")
  if not os.path.exists(os.path.join(LOCAL_MODULE_PATH, "mst_pci.ko")):
    print("!!! FAILURE !!! Cannot find mst_pci.ko.")
    raise FileNotFoundError("mst_pci.ko not found")

  result = checker_common.run_command("lsmod")
  stdout = result.stdout
  if "mst_pci" not in stdout:
    print("Loading mst_pci...")
    checker_common.run_command(f"sudo insmod {LOCAL_MODULE_PATH}/mst_pci.ko")
  else:
    print("mst_pci module is already loaded.")

  if "mst_pciconf" not in stdout:
    print("Loading mst_pciconf...")
    checker_common.run_command(
        f"sudo insmod {LOCAL_MODULE_PATH}/mst_pciconf.ko debug=1"
    )
  else:
    print("mst_pciconf module is already loaded.")

  print("SUCCESS: Kernel modules should be loaded.")
  checker_common.run_command("lsmod | grep mst")


def _get_mft_userspace_download_info():
  """Determines the download URL and local path for MFT userspace tools."""
  mft_tgz_files = glob.glob(f"{LOCAL_MODULE_PATH}/mft-kernel-modules-*.tgz")
  if not mft_tgz_files:
    raise FileNotFoundError("MFT kernel module tgz not found")
  mft_tgz_filename = mft_tgz_files[0]

  version_string = (
      os.path.basename(mft_tgz_filename)
      .replace("mft-kernel-modules-", "")
      .replace(".tgz", "")
  )
  parts = version_string.split("-")
  if len(parts) < 3:
    print(f"!!! FAILURE !!! Could not parse version from {mft_tgz_filename}")
    raise ValueError("Could not parse version from filename")
  mft_ver = parts[0]
  build_ver = parts[1]
  print(f"Success: Detected MFT version {mft_ver}-{build_ver}")

  arch = os.uname().machine
  suffix = "deb.tgz"
  userspace_filename = f"mft-{mft_ver}-{build_ver}-{arch}-{suffix}"
  download_url = f"https://www.mellanox.com/downloads/MFT/{userspace_filename}"
  download_path = f"/tmp/{userspace_filename}"
  return download_url, download_path


def _download_mft_package(download_url: str, download_path: str):
  """Downloads the MFT package from the URL."""
  print(f"Downloading MFT userspace package from {download_url}...")
  try:
    response = requests.get(download_url, stream=True, timeout=120)
    response.raise_for_status()
    with open(download_path, "wb") as f:
      for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
    print("Download complete.")
  except requests.exceptions.RequestException as e:
    print(f"Error downloading {download_url}: {e}")
    raise


def _install_mft_from_extracted(download_path: str):
  """Installs MFT from the downloaded and extracted package."""
  print("Extracting package...")
  shutil.rmtree(MFT_INSTALL_EXTRACT_DIR, ignore_errors=True)
  os.makedirs(MFT_INSTALL_EXTRACT_DIR, exist_ok=True)
  checker_common.run_command(
      f"tar -xzvf {download_path} -C {MFT_INSTALL_EXTRACT_DIR}"
  )

  uname_r = os.uname().release
  kernel_header_makefile = f"/usr/src/linux-headers-{uname_r}/Makefile"

  install_script_dirs = glob.glob(f"{MFT_INSTALL_EXTRACT_DIR}/mft-*")
  if not install_script_dirs:
    print("!!! FAILURE !!! Could not find install script directory.")
    raise FileNotFoundError("MFT install script directory not found")
  install_script_dir = install_script_dirs[0]
  install_script = os.path.join(install_script_dir, "install.sh")

  if os.path.exists(kernel_header_makefile):
    print("Kernel headers found. Proceeding with standard installation.")
    checker_common.run_command(f"yes | sudo {install_script}")
  else:
    print(
        "!!! WARNING: Kernel headers not found. Falling back to manual package"
        " installation."
    )
    mft_deb_files = glob.glob(f"{MFT_INSTALL_EXTRACT_DIR}/*/DEBS/mft_*.deb")
    if not mft_deb_files:
      print("!!! CRITICAL FAILURE: Could not find the MFT .deb file.")
      print("Attempting to install dependencies (gcc, make)...")
      checker_common.run_command("sudo apt-get update")
      checker_common.run_command("sudo apt-get install -y gcc make")
      raise FileNotFoundError("MFT .deb file not found")
    mft_deb_file = mft_deb_files[0]
    print(f"Manually installing package: {mft_deb_file}")
    checker_common.run_command(f"sudo dpkg -i {mft_deb_file}")
    print("Fixing broken dependencies if any...")
    checker_common.run_command("sudo apt-get install -f -y")


def install_mft_userspace():
  """Downloads and installs MFT userspace tools."""
  print("--- [Step 4/5] Downloading and Installing MFT Userspace Tools ---")
  download_url, download_path = _get_mft_userspace_download_info()
  _download_mft_package(download_url, download_path)
  _install_mft_from_extracted(download_path)

  print("--- Verifying MFT installation ---")
  checker_common.run_command("sudo mst start")
  checker_common.run_command("sudo mst status")
  print("SUCCESS: MFT userspace tools should now be installed and running.")


def install_mft_if_needed(node_name: str) -> bool:
  """Installs Mellanox Firmware Tools (MFT) if necessary."""
  if not is_a4_or_a4x_machine(node_name):
    print(f"Node {node_name} is not an A4 machine, skipping MFT installation.")
    return False

  print(f"A4 machine detected, proceeding with MFT installation on {node_name}")
  try:
    download_mft_kernel_modules()
    load_mft_kernel_modules()
    install_mft_userspace()
    print("MFT installation process completed successfully.")
    return True
  except (
      FileNotFoundError,
      ValueError,
      requests.exceptions.RequestException,
      subprocess.CalledProcessError,
      subprocess.TimeoutExpired,
      OSError,
  ) as e:
    print(f"MFT installation failed: {e}")
    return False
