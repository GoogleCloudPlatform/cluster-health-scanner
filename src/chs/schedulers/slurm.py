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

"""Slurm scheduler implementation for Cluster Health Scanner."""

import concurrent.futures
import importlib.resources as pkg_resources
import logging
import os
import re
import subprocess
import yaml
from google.cloud import compute_v1
import google.auth
from google.api_core.exceptions import GoogleAPICallError

from chs.exceptions import NoActiveJobsException

from typing import Any


from chs.schedulers.base import BaseScheduler
from chs.utils.lifecycle import JobStatus
import click

from chs import exceptions as ex

_MAX_CONCURRENT_API_REQUESTS = 10


def boolean_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:bool", str(data))


yaml.Dumper.add_representer(bool, boolean_representer)

logger = logging.getLogger(__name__)


class SlurmScheduler(BaseScheduler):
    """Slurm scheduler implementation."""

    @property
    def _enroot_paths(self) -> dict[str, str]:
        """Dictionary of Enroot paths, allowing environment overrides."""
        # TODO: Find a more robust way to manage enroot paths.
        base = os.getenv(
            "ENROOT_BASE", os.path.join(os.path.expanduser("~"), ".enroot")
        )
        return {
            "ENROOT_RUNTIME_PATH": os.getenv(
                "ENROOT_RUNTIME_PATH", os.path.join(base, "run")
            ),
            "ENROOT_CACHE_PATH": os.getenv(
                "ENROOT_CACHE_PATH", os.path.join(base, "cache")
            ),
            "ENROOT_CONFIG_CREDENTIALS": os.getenv(
                "ENROOT_CONFIG_CREDENTIALS", os.path.join(base, ".credentials")
            ),
        }

    def __init__(
        self,
        nodes: str | None = None,
        nodelists: str | None = None,
        partitions: str | None = None,
        nodepools: str | None = None,
    ):
        super().__init__()

        if nodepools:
            raise ex.ChsValidationError(
                "The --nodepools flag is Kubernetes-specific and cannot be "
                "used with the Slurm scheduler. Please use --nodes, "
                "--nodelists, or --partitions instead."
            )

        self.nodes_arg = nodes
        self.nodelists_arg = nodelists
        self.partitions_arg = partitions
        self._resolved_nodes: list[str] | None = None
        self._resolved_raw_nodes: list[str] | None = None

    def _resolve_project_id(self, node: str) -> str:
        """Resolves GCP project ID from GCP application default credentials."""
        try:
            _, project = google.auth.default()
        except Exception as e:
            raise ex.ChsSchedulerException(
                "Failed to get GCP credentials."
            ) from e
        if not project:
            raise ex.ChsSchedulerException(
                "Could not determine GCP project from credentials. Please set"
                " it using 'gcloud auth application-default login --project"
                " <PROJECT>' or the GOOGLE_CLOUD_PROJECT environment variable."
            )
        return project

    @property
    def name(self) -> str:
        return "slurm"

    def get_cluster_name(self) -> str | None:
        """Retrieves cluster name from Slurm scontrol.

        Failsafe: All subprocess execution queries are wrapped in strict timeout
        blocks.
        """
        try:
            cmd = ["scontrol", "show", "config"]
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5, check=False
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if "ClusterName" in line:
                        parts = line.split("=")
                        if len(parts) == 2:
                            val = parts[1].strip()
                            logger.debug(
                                "Found cluster name from scontrol: %s", val
                            )
                            return val
        except Exception as e:
            logger.debug("scontrol query failed: %s", e)

        return None

    def _run_srun(self, args: list[str], timeout: int = 60) -> str:
        """Runs a command via srun and returns filtered output."""
        cmd = ["srun", "-N1", "-n1"]
        if self.partitions_arg:
            cmd.extend(["-p", self.partitions_arg])
        cmd.extend(args)
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=True,
            timeout=timeout,
        )
        return result.stdout.strip()

    def _discover_network_interfaces(self) -> list[str]:
        """
        Discovers physical network interfaces from the routing table.
        Uses the default route interface as an anchor to find siblings.
        """
        try:
            default_out = self._run_srun(["ip", "route", "show", "default"])
            anchor_match = re.search(r"dev\s+(\S+)", default_out)

            prefix = ""
            if anchor_match and (
                m := re.match(r"^[a-zA-Z]+", anchor_match.group(1))
            ):
                prefix = m.group(0)
                logger.debug("Anchor interface prefix: %s", prefix)

            route_out = self._run_srun(["ip", "route", "show"])
            all_ifaces = set(re.findall(r"dev\s+(\S+)", route_out))
            excluded = ("docker", "veth", "br-", "lo")
            interfaces = sorted(
                [
                    iface
                    for iface in all_ifaces
                    if not iface.startswith(excluded)
                    and (prefix and iface.startswith(prefix))
                ]
            )

            logger.debug("Detected network interfaces: %r", interfaces)
            return interfaces

        except Exception as e:
            raise ex.NetworkDiscoveryException(
                f"Unexpected error during network discovery: {e}"
            ) from e

    def resolve_raw_nodes(self) -> list[str]:
        if self._resolved_raw_nodes is not None:
            return self._resolved_raw_nodes

        if (
            bool(self.nodes_arg)
            + bool(self.nodelists_arg)
            + bool(self.partitions_arg)
        ) > 1:
            raise click.UsageError(
                "Only one of --nodes, --nodelists, or --partitions can be "
                "provided."
            )

        raw_nodes = []
        if self.nodes_arg:
            raw_nodes = [
                n.strip() for n in self.nodes_arg.split(",") if n.strip()
            ]

        elif self.nodelists_arg:
            try:
                output = subprocess.run(
                    ["scontrol", "show", "hostname", self.nodelists_arg],
                    text=True,
                    check=True,
                    capture_output=True,
                    timeout=5,
                ).stdout.strip()
                raw_nodes = [n for n in output.split("\n") if n]
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise ex.ChsSchedulerException(
                    f"Failed to expand nodelists: {e}"
                ) from e

        else:
            sinfo_cmd = ["sinfo", "-e", "-N", "-h", "-o", "%N"]
            if self.partitions_arg:
                sinfo_cmd.extend(["-p", self.partitions_arg])
            try:
                output = subprocess.run(
                    sinfo_cmd, text=True, check=True, capture_output=True
                ).stdout.strip()
                raw_nodes = [
                    line.strip() for line in output.split("\n") if line.strip()
                ]
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise ex.ChsSchedulerException(
                    f"Failed to get raw nodes: {e}"
                ) from e

        self._resolved_raw_nodes = raw_nodes
        return self._resolved_raw_nodes

    def get_nodes_under_test(self) -> list[str]:
        if self._resolved_nodes is not None:
            return self._resolved_nodes

        raw_nodes = self.resolve_raw_nodes()
        if not raw_nodes:
            self._resolved_nodes = []
            return []

        try:
            output = subprocess.run(
                ["sinfo", "-e", "-t", "idle", "-N", "-h", "-o", "%N"],
                text=True,
                check=True,
                capture_output=True,
            ).stdout.strip()
            idle_nodes = set(
                line.strip() for line in output.split("\n") if line.strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise ex.ChsSchedulerException(
                f"Failed to get free nodes: {e}"
            ) from e

        busy_nodes = set(raw_nodes) - idle_nodes
        if busy_nodes:
            logger.info(
                "The following nodes are currently executing workloads and "
                "will be skipped to avoid disruption: %s",
                ", ".join(sorted(busy_nodes)),
            )

        valid_nodes = [node for node in raw_nodes if node in idle_nodes]
        self._resolved_nodes = valid_nodes
        return self._resolved_nodes

    def get_topology(self) -> dict[str, dict[str, list[str]]]:
        """Discovers network topology using the GCE API via gcloud.

        Returns a dictionary mapping block names to dicts of subblocks to lists
        of nodes.
        """
        nodes = self.get_nodes_under_test()
        if not nodes:
            return {}

        try:
            credentials, project = google.auth.default()
            if not project:
                logger.warning(
                    "Could not determine GCP project via Application Default"
                    " Credentials. GCE topology discovery skipped. Set"
                    " 'GOOGLE_CLOUD_PROJECT' or run 'gcloud auth"
                    " application-default login'."
                )
                return {}

            client = compute_v1.InstancesClient()
            topology: dict[str, dict[str, list[str]]] = {}
            batch_size = 100

            def fetch_batch(batch_nodes):
                filter_str = " OR ".join([f'name = "{n}"' for n in batch_nodes])
                request = compute_v1.AggregatedListInstancesRequest(
                    project=project,
                    filter=filter_str,
                )
                return client.aggregated_list(request=request)

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=_MAX_CONCURRENT_API_REQUESTS
            ) as executor:
                futures = [
                    executor.submit(fetch_batch, nodes[i : i + batch_size])
                    for i in range(0, len(nodes), batch_size)
                ]

                for future in concurrent.futures.as_completed(futures):
                    page_result = future.result()
                    for zone, instances_in_zone in page_result:
                        if instances_in_zone.instances:
                            for item in instances_in_zone.instances:
                                name = item.name
                                if not name:
                                    continue

                                resource_status = getattr(
                                    item, "resource_status", None
                                )
                                if not resource_status:
                                    continue
                                topo_info = getattr(
                                    resource_status,
                                    "physical_host_topology",
                                    None,
                                )
                                if not topo_info:
                                    continue

                                block = topo_info.block or "default-block"
                                subblock = (
                                    topo_info.subblock or "default-subblock"
                                )

                                topology.setdefault(block, {}).setdefault(
                                    subblock, []
                                ).append(name)

            return topology

        except GoogleAPICallError as e:
            logger.warning(
                "Failed to retrieve node topology information "
                "via the Slurm API. Error: %s",
                e,
            )
            return {}
        except Exception as e:
            logger.error(
                "Encountered an unexpected error while discovering "
                "Slurm node topology. Error: %s",
                e,
            )
            return {}

    def _ensure_enroot_directories(self) -> None:
        """Ensures that Enroot runtime and cache directories exist."""
        for k, d in self._enroot_paths.items():
            if k != "ENROOT_CONFIG_CREDENTIALS" and not os.path.exists(d):
                try:
                    os.makedirs(d, exist_ok=True)
                    logger.debug("Created Enroot directory: %s", d)
                except OSError as e:
                    raise ex.ChsSchedulerException(
                        f"Failed to create Enroot directory {d}: {e}"
                    ) from e

    def get_extra_env(self) -> dict[str, str]:
        """Returns extra environment variables for the scheduler."""
        return self._enroot_paths

    def write_ramble_config(
        self,
        workspace_config_dir: str,
        tests: list[dict[str, Any]] | None,
        machine_type: str,
        deadline: int | None = None,
        run_id: str | None = None,
        images_dir: str | None = None,
    ) -> None:
        self._ensure_enroot_directories()
        nodes = self.get_nodes_under_test()
        logger.debug("Configuring workspace with slurm_template.yaml")
        template_path = pkg_resources.files("chs.schedulers").joinpath(
            "slurm_template.yaml"
        )
        with pkg_resources.as_file(template_path) as p:
            with open(p, "r") as f:
                config = yaml.safe_load(f)

        tcp_interfaces = self._discover_network_interfaces()
        if not tcp_interfaces:
            raise ex.NetworkDiscoveryException(
                "Network discovery failed on Slurm. "
                "Could not identify required TCP interfaces."
            )

        tcp_interfaces_str = ",".join(tcp_interfaces)

        images_dir_value = images_dir or "{workload_input_dir}"

        variables = config["ramble"]["variables"]
        variables["nodes"] = nodes or []
        variables["machine_type"] = machine_type
        variables["tests"] = tests or []
        variables["container_dir"] = images_dir_value
        variables["tcp_interfaces"] = tcp_interfaces_str

        if deadline is not None:
            variables["deadline"] = deadline

        variables["chs_run_id"] = run_id

        logger.debug("Configuration: %r", config)

        self._write_topology_json(workspace_config_dir)

        with open(os.path.join(workspace_config_dir, "ramble.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # TODO: Test and make slurm workflow consistent with GKE
    def has_no_failed_jobs(self) -> bool:
        """Checks if there are any failed CHS jobs.

        Returns:
            True if all are healthy (active/completed).
        """
        cmd = [
            "sacct",
            "-P",
            "-X",
            "-n",
            "--format=JobName,State",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False
            )
            if not result.stdout.strip():
                raise NoActiveJobsException("No jobs found on the cluster.")

            has_chs_jobs = False
            for line in result.stdout.splitlines():
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    job_name = parts[0]
                    state = parts[1].upper()
                    if job_name.startswith("chs-"):
                        has_chs_jobs = True
                        if state in (
                            "FAILED",
                            "NODE_FAIL",
                            "OUT_OF_MEMORY",
                            "BOOT_FAIL",
                        ):
                            logger.warning(
                                "Diagnostic job '%s' has failed "
                                "with state '%s'. "
                                "Initiating resource cleanup.",
                                job_name,
                                state,
                            )
                            return False

            if not has_chs_jobs:
                raise NoActiveJobsException("No CHS jobs found on the cluster.")

            return True
        except FileNotFoundError:
            logger.warning(
                "Slurm CLI tool 'sacct' was not found in PATH. Please ensure"
                " Slurm client utilities (sacct/scancel) are installed and"
                " available in your PATH."
            )
        return False

    def get_job_statuses(self, job_ids: list[str]) -> dict[str, str]:
        """Gets the status of the given jobs using sacct."""
        if not job_ids:
            return {}

        statuses = {}
        cmd = [
            "sacct",
            "-j",
            ",".join(job_ids),
            "--noheader",
            "--format=JobID,State",
            "-P",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False
            )
            logger.debug(" sacct output: %r", result.stdout)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "|" in line:
                        jid, state = line.split("|", 1)
                        jid = jid.strip()
                        if jid.isdigit():
                            statuses[jid] = JobStatus.from_string(
                                state.strip().split()[0]
                            )

            # Fallback for any jobs not found in sacct output
            for jid in job_ids:
                if jid not in statuses:
                    statuses[jid] = JobStatus.UNKNOWN.value

        except FileNotFoundError:
            logger.debug(
                "The 'sacct' command was not found. Job states cannot be "
                "verified and will default to UNKNOWN."
            )
            logger.warning(
                "The 'sacct' command was not found. Unable to "
                "track job completion statuses."
            )
            for jid in job_ids:
                statuses[jid] = JobStatus.UNKNOWN.value

        return statuses

    def fetch_job_log(self, job_id: str, log_path: str) -> None:
        """Fetches the log for a completed job.

        For Slurm, logs are typically written directly to the shared
        filesystem by the job itself (via --output), so this might be a no-op
        """
        if os.path.exists(log_path):
            logger.debug("Log file for job %s exists at %s", job_id, log_path)
        else:
            logger.debug(
                "The execution log file for job ID '%s' "
                "could not be found at '%s'.",
                job_id,
                log_path,
            )

    def extract_job_id(self, experiment_dir: str) -> str:
        """Extracts the job ID from the experiment directory."""
        job_file = os.path.join(experiment_dir, ".slurm_job")
        try:
            with open(job_file, "r") as f:
                job_id = f.read().strip()
                if job_id.isdigit():
                    return job_id
            raise ex.ChsSchedulerException(
                f"Slurm job file {job_file} does not contain a valid job ID"
            )
        except IOError as e:
            raise ex.ChsSchedulerException(
                f"Failed to read Slurm job file at {job_file}: {e}"
            ) from e

    def cancel_jobs(self, job_ids: list[str]) -> None:
        """Cancels the given jobs using scancel."""
        if not job_ids:
            logger.warning("No job IDs to cancel.")
            return

        cmd = ["scancel", ",".join(job_ids)]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                logger.debug("Successfully canceled jobs: %s", job_ids)
            else:
                logger.warning(
                    "The 'scancel' command failed while attempting "
                    "to terminate jobs. Error: %s",
                    result.stderr,
                )
        except FileNotFoundError:
            logger.warning(
                "Slurm CLI tool 'scancel' was not found in PATH. Please ensure"
                " Slurm client utilities (sacct/scancel) are installed and"
                " available in your PATH."
            )
        except (OSError, subprocess.SubprocessError) as e:
            err_msg = ex.format_subprocess_error(e)
            logger.error(
                "Failed to cancel active Slurm jobs. Error: %s", err_msg
            )
            raise ex.ChsCancellationException(
                f"Failed to cancel jobs: {err_msg}"
            ) from e

    def get_active_job_names(self) -> list[str]:
        """Returns a list of active job names on the cluster."""
        try:
            cmd = ["squeue", "-h", "-o", "%.200j"]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                return []

            job_names = []
            for line in result.stdout.splitlines():
                name = line.strip()
                if name:
                    job_names.append(name)
            return job_names
        except FileNotFoundError:
            logger.warning(
                "The 'squeue' command was not found in your PATH. "
                "Cannot retrieve the list of active jobs."
            )
            return []

    def cancel_jobs_by_prefix(self, prefix: str) -> None:
        """Cancels all jobs matching the specified prefix natively."""
        logger.debug("Attempting native cancel for prefix: %s", prefix)
        try:
            # List jobs for current user and extract job IDs matching the prefix
            cmd = [
                "squeue",
                "-h",
                "-o",
                "%i %.200j",
                "-u",
                os.environ.get("USER", ""),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                logger.warning(
                    "Failed to retrieve the list of active jobs using "
                    "'squeue' during the cancellation process."
                )
                return

            job_ids = []
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[1].startswith(prefix):
                    job_ids.append(parts[0])

            if job_ids:
                logger.debug(
                    "Found %d jobs to cancel for prefix %s.",
                    len(job_ids),
                    prefix,
                )
                self.cancel_jobs(job_ids)
            else:
                logger.debug("No matching jobs found for prefix %s.", prefix)
        except FileNotFoundError:
            logger.warning(
                "The 'squeue' command was not found in your PATH. "
                "Cannot perform native job cancellation."
            )
        except Exception as e:
            logger.warning(
                "Failed to execute native job "
                "cancellation via Slurm. Error: %s",
                e,
            )
