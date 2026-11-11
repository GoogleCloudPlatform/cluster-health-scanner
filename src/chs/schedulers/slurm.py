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

import collections
import concurrent.futures
import importlib.resources as pkg_resources
import logging
import os
import re
import subprocess
import yaml
import google.auth

from chs.resources import normalize_machine_type

from typing import Any


from chs.schedulers.base import BaseScheduler
from chs.utils.lifecycle import JobStatus
import click

from chs import exceptions as ex
from chs.constants import SLURM_CMD_TIMEOUT_SEC, SLURM_RUN_ID_COMMENT_PREFIX

_MAX_CONCURRENT_API_REQUESTS = 10
_MAX_CONCURRENT_HOSTLIST_EXPANSIONS = 10
# squeue prints these, in any case, when a job has no requested or allocated
# nodes.
_EMPTY_HOSTLISTS = frozenset({"", "(null)", "n/a", "none"})

_BENIGN_SCANCEL_ERROR = re.compile(
    r"already completing or completed|Invalid job id specified"
)


def boolean_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:bool", str(data))


yaml.Dumper.add_representer(bool, boolean_representer)

logger = logging.getLogger(__name__)


class SlurmScheduler(BaseScheduler):
    """Slurm scheduler implementation."""

    _INTERDOMAIN_RESTRICTED_MACHINE_TYPES = frozenset(
        {
            "a4x-highgpu-4g",
            "a4x-maxgpu-4g-metal",
        }
    )

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

    def _run_slurm_cmd(
        self,
        cmd: list[str],
        check: bool = False,
        timeout: int = SLURM_CMD_TIMEOUT_SEC,
    ) -> subprocess.CompletedProcess:
        """Runs a Slurm client command with a timeout."""
        return subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=check,
            timeout=timeout,
        )

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
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                raise ex.ChsSchedulerException(
                    f"Failed to expand nodelists: {e}"
                ) from e

        else:
            sinfo_cmd = ["sinfo", "-e", "-N", "-h", "-o", "%N"]
            if self.partitions_arg:
                sinfo_cmd.extend(["-p", self.partitions_arg])
            try:
                output = self._run_slurm_cmd(
                    sinfo_cmd, check=True
                ).stdout.strip()
                raw_nodes = [
                    line.strip() for line in output.split("\n") if line.strip()
                ]
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                raise ex.ChsSchedulerException(
                    f"Failed to get raw nodes: {e}"
                ) from e

        self._resolved_raw_nodes = raw_nodes
        return self._resolved_raw_nodes

    def get_nodes_under_test(self) -> list[str]:
        """Returns idle nodes not targeted by other CHS runs."""
        if self._resolved_nodes is not None:
            return self._resolved_nodes

        raw_nodes = self.resolve_raw_nodes()
        if not raw_nodes:
            self._resolved_nodes = []
            return []

        try:
            output = self._run_slurm_cmd(
                ["sinfo", "-e", "-t", "idle", "-N", "-h", "-o", "%N"],
                check=True,
            ).stdout.strip()
            idle_nodes = set(
                line.strip() for line in output.split("\n") if line.strip()
            )
        except (subprocess.SubprocessError, FileNotFoundError) as e:
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

        targeted_nodes = set()
        for run_id, run_nodes in sorted(
            self.find_runs_targeting_nodes(raw_nodes).items()
        ):
            targeted_nodes.update(run_nodes)
            logger.warning(
                "Skipping %d node(s) targeted by CHS run '%s'.",
                len(run_nodes),
                run_id,
            )

        valid_nodes = [
            node
            for node in raw_nodes
            if node in idle_nodes and node not in targeted_nodes
        ]
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

        # Lazy import to avoid heavy compute_v1 import overhead on CLI startup.
        from google.api_core.exceptions import GoogleAPICallError
        from google.cloud import compute_v1

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
        variables["allow_interdomain_pairs"] = self.allow_interdomain_pairs(
            machine_type
        )

        logger.debug("Configuration: %r", config)

        self._write_topology_json(workspace_config_dir)

        with open(os.path.join(workspace_config_dir, "ramble.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def allow_interdomain_pairs(self, machine_type: str = "") -> bool:
        """Returns whether the scheduler allows cross-domain node pairs."""
        # TODO: Revisit after we fix inter partition scheduling.
        return (
            normalize_machine_type(machine_type).lower()
            not in self._INTERDOMAIN_RESTRICTED_MACHINE_TYPES
        )

    def get_job_statuses(self, job_ids: list[str]) -> dict[str, str]:
        """Gets the status of the given jobs using sacct."""
        if not job_ids:
            return {}

        statuses = {jid: JobStatus.UNKNOWN.value for jid in job_ids}
        cmd = [
            "sacct",
            "-j",
            ",".join(job_ids),
            "--noheader",
            "--format=JobID,State",
            "-P",
        ]
        try:
            result = self._run_slurm_cmd(cmd)
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

        except FileNotFoundError:
            logger.debug(
                "The 'sacct' command was not found. Job states cannot be "
                "verified and will default to UNKNOWN."
            )
            logger.warning(
                "The 'sacct' command was not found. Unable to "
                "track job completion statuses."
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "Timed out querying job states with sacct. Job states will "
                "default to UNKNOWN."
            )

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
            result = self._run_slurm_cmd(cmd)
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

    def find_runs_targeting_nodes(
        self, nodes: list[str]
    ) -> dict[str, list[str]]:
        """Finds CHS runs whose unfinished jobs target the given nodes.

        A job targets the nodes it is allocated and, while pending, the nodes
        it requested.

        Raises:
          ChsSchedulerException: If Slurm cannot be queried.
        """
        if not nodes:
            return {}
        if self._jobs_are_private():
            logger.warning(
                "This Slurm cluster sets PrivateData=jobs, so jobs of other "
                "users are hidden unless you are a Slurm operator. CHS only "
                "guarantees exclusive nodes among your own runs."
            )
        try:
            run_hostlists = self._list_run_hostlists()
            nodes_by_hostlist = self._expand_hostlists(
                {hostlist for _, hostlist in run_hostlists}
            )
        except (OSError, subprocess.SubprocessError) as e:
            raise ex.ChsSchedulerException(
                "Failed to list active CHS jobs; cannot verify that no other "
                "CHS run targets the requested nodes: "
                f"{ex.format_subprocess_error(e)}"
            ) from e

        requested_nodes = set(nodes)
        targeted: dict[str, set[str]] = collections.defaultdict(set)
        for run_id, hostlist in run_hostlists:
            targeted[run_id] |= nodes_by_hostlist[hostlist] & requested_nodes
        return {
            run_id: sorted(run_nodes)
            for run_id, run_nodes in targeted.items()
            if run_nodes
        }

    def _list_run_hostlists(self) -> list[tuple[str, str]]:
        """Lists (run ID, hostlist) pairs of the visible CHS jobs.

        Each job contributes its requested and its allocated hostlist. Jobs
        without a run ID comment are skipped.
        """
        # The comment is free text, so it comes last.
        cmd = ["squeue", "--all", "--noheader", "--format=%n|%N|%k"]
        result = self._run_slurm_cmd(cmd, check=True)
        run_hostlists = []
        for line in result.stdout.splitlines():
            requested, _, rest = line.partition("|")
            allocated, _, comment = rest.partition("|")
            comment = comment.strip()
            if not comment.startswith(SLURM_RUN_ID_COMMENT_PREFIX):
                continue
            run_id = comment.removeprefix(SLURM_RUN_ID_COMMENT_PREFIX)
            run_hostlists.extend(
                (run_id, hostlist)
                for hostlist in (requested.strip(), allocated.strip())
                if hostlist.lower() not in _EMPTY_HOSTLISTS
            )
        return run_hostlists

    def _expand_hostlists(
        self, hostlists: set[str]
    ) -> dict[str, frozenset[str]]:
        """Expands Slurm hostlist expressions into node names, concurrently.

        `scontrol show hostnames` runs locally without contacting slurmctld.
        """

        def expand(hostlist: str) -> frozenset[str]:
            cmd = ["scontrol", "show", "hostnames", hostlist]
            return frozenset(
                self._run_slurm_cmd(cmd, check=True).stdout.split()
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=_MAX_CONCURRENT_HOSTLIST_EXPANSIONS
        ) as executor:
            return dict(zip(hostlists, executor.map(expand, hostlists)))

    def _jobs_are_private(self) -> bool:
        """Returns whether Slurm hides other users' jobs (PrivateData=jobs)."""
        try:
            result = self._run_slurm_cmd(
                ["scontrol", "show", "config"], check=True
            )
        except (OSError, subprocess.SubprocessError) as e:
            logger.debug("Could not read the Slurm PrivateData setting: %s", e)
            return False
        for line in result.stdout.splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "PrivateData":
                return "jobs" in re.split(r"[,\s]+", value.strip().lower())
        return False

    def cancel_run(self, run_id: str) -> bool:
        """Cancels the visible jobs whose comment stamps run_id.

        Returns:
          Whether the run was active, i.e. had jobs on the cluster.

        Raises:
          ChsCancellationException: If the jobs cannot be listed or any of
            them cannot be cancelled, e.g. another user's job without
            operator privileges.
        """
        comment = f"{SLURM_RUN_ID_COMMENT_PREFIX}{run_id}"
        job_ids = [
            job_id
            for job_id, job_comment in self._list_visible_jobs("%k")
            if job_comment == comment
        ]
        self._scancel(job_ids, f"run '{run_id}'")
        return bool(job_ids)

    def cancel_all_runs(self) -> None:
        """Cancels all CHS jobs visible to the caller.

        With PrivateData=jobs, only Slurm operators see other users' jobs.

        Raises:
          ChsCancellationException: If the jobs cannot be listed or any of
            them cannot be cancelled, e.g. another user's job without
            operator privileges.
        """
        if self._jobs_are_private():
            logger.warning(
                "This Slurm cluster sets PrivateData=jobs, so only your own "
                "CHS jobs are cancelled unless you are a Slurm operator."
            )
        self._scancel(
            [
                job_id
                for job_id, comment in self._list_visible_jobs("%k")
                if comment.startswith(SLURM_RUN_ID_COMMENT_PREFIX)
            ],
            "all CHS runs",
        )

    def _list_visible_jobs(self, field: str) -> list[tuple[str, str]]:
        """Lists (job ID, field value) pairs of the jobs visible to the caller.

        Raises:
          ChsCancellationException: If the jobs cannot be listed.
        """
        cmd = ["squeue", "--all", "--noheader", f"--format=%A|{field}"]
        try:
            result = self._run_slurm_cmd(cmd, check=True)
        except (OSError, subprocess.SubprocessError) as e:
            raise ex.ChsCancellationException(
                "Failed to list the Slurm jobs to cancel: "
                f"{ex.format_subprocess_error(e)}"
            ) from e
        return [
            (job_id.strip(), value.strip())
            for job_id, _, value in (
                line.partition("|") for line in result.stdout.splitlines()
            )
        ]

    def _scancel(self, job_ids: list[str], scope: str) -> None:
        """Cancels the jobs; jobs that already finished are not errors.

        Raises:
          ChsCancellationException: If any job cannot be cancelled.
        """
        if not job_ids:
            logger.debug("No Slurm jobs to cancel for %s.", scope)
            return
        logger.debug("Cancelling %d Slurm jobs of %s.", len(job_ids), scope)
        try:
            result = self._run_slurm_cmd(["scancel", ",".join(job_ids)])
        except (OSError, subprocess.SubprocessError) as e:
            raise ex.ChsCancellationException(
                f"Failed to cancel the Slurm jobs of {scope}: "
                f"{ex.format_subprocess_error(e)}"
            ) from e
        if result.returncode == 0:
            return
        stderr = result.stderr.strip()
        errors = [
            line
            for line in stderr.splitlines()
            if line.strip() and not _BENIGN_SCANCEL_ERROR.search(line)
        ]
        if not stderr:
            errors = [f"scancel exited with code {result.returncode}"]
        if errors:
            raise ex.ChsCancellationException(
                f"Failed to cancel some Slurm jobs of {scope}. Jobs of other "
                "users can only be cancelled by their owner or a Slurm "
                "operator. Errors: " + "; ".join(errors)
            )
