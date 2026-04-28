"""Slurm scheduler implementation for Cluster Health Scanner."""

import importlib.resources as pkg_resources
import logging
import os
import re
import subprocess
import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from chs.utils.app_data import ChsAppData


from chs.schedulers.base import BaseScheduler
from chs.utils.lifecycle import JobStatus
import click

from chs import exceptions as ex

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
        nodelist: str | None = None,
        partition: str | None = None,
        app_data: "ChsAppData | None" = None,
    ):
        super().__init__(app_data)
        self.nodes_arg = nodes
        self.nodelist_arg = nodelist
        self.partition_arg = partition
        self._resolved_nodes: list[str] | None = None

    @property
    def name(self) -> str:
        return "slurm"

    def _run_srun(self, args: list[str], timeout: int = 60) -> str:
        """Runs a command via srun and returns filtered output."""
        cmd = ["srun", "-N1", "-n1"]
        if self.partition_arg:
            cmd.extend(["-p", self.partition_arg])
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

            logger.info("Detected network interfaces: %r", interfaces)
            return interfaces

        except Exception as e:
            raise ex.NetworkDiscoveryException(
                f"Unexpected error during network discovery: {e}"
            ) from e

    def get_nodes_under_test(self) -> list[str]:
        if self._resolved_nodes is not None:
            return self._resolved_nodes

        if (
            bool(self.nodes_arg)
            + bool(self.nodelist_arg)
            + bool(self.partition_arg)
        ) > 1:
            raise click.UsageError(
                "Only one of --nodes, --nodelist, or "
                "--partition can be provided."
            )

        if self.nodes_arg:
            self._resolved_nodes = [
                n.strip() for n in self.nodes_arg.split(",") if n.strip()
            ]
            return self._resolved_nodes

        if self.nodelist_arg:
            try:
                output = subprocess.run(
                    ["scontrol", "show", "hostname", self.nodelist_arg],
                    text=True,
                    check=True,
                    capture_output=True,
                ).stdout.strip()
                self._resolved_nodes = [n for n in output.split("\n") if n]
                return self._resolved_nodes
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise ex.ChsSchedulerException(
                    f"Failed to expand nodelist: {e}"
                ) from e

        sinfo_cmd = ["sinfo", "-t", "idle", "-N", "-h", "-o", "%N"]
        if self.partition_arg:
            sinfo_cmd.extend(["-p", self.partition_arg])

        try:
            output = subprocess.run(
                sinfo_cmd,
                text=True,
                check=True,
                capture_output=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise ex.ChsSchedulerException(
                f"Failed to get free nodes: {e}"
            ) from e

        if not output:
            return []

        all_nodes = [
            line.strip() for line in output.split("\n") if line.strip()
        ]

        self._resolved_nodes = all_nodes
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
            # Batch filter: name=(node1 OR node2 ...)
            nodes_filter_str = f"name=({' '.join(nodes)})"
            args = [
                "gcloud",
                "compute",
                "instances",
                "list",
                f"--filter={nodes_filter_str}",
                "--format=json(name,resourceStatus.physicalHostTopology)",
            ]

            output = subprocess.run(
                args,
                text=True,
                capture_output=True,
                check=True,
            ).stdout

            if not output:
                return {}

            data = json.loads(output)

            topology: dict[str, dict[str, list[str]]] = {}
            for item in data:
                name = item.get("name")
                if not name:
                    continue

                topo_info = item.get("resourceStatus", {}).get(
                    "physicalHostTopology", {}
                )
                block = topo_info.get("block", "default-block")
                subblock = topo_info.get("subblock", "default-subblock")

                topology.setdefault(block, {}).setdefault(subblock, []).append(
                    name
                )

            return topology

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning("Failed to get node topology via gcloud: %s", e)
            return {}
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse gcloud output: %s", e)
            return {}

    def _ensure_enroot_directories(self) -> None:
        """Ensures that Enroot runtime and cache directories exist."""
        for d in self._enroot_paths.values():
            if not os.path.exists(d):
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
        timeout: int | None = None,
        deadline: int | None = None,
    ) -> None:
        self._ensure_enroot_directories()
        nodes = self.get_nodes_under_test()
        logger.debug("Configuring workspace with slurm_template.yaml")
        template_path = pkg_resources.files("chs.schedulers").joinpath(
            "slurm_template.yaml"
        )
        with pkg_resources.as_file(template_path) as p:
            with open(p, "r") as f:
                template_content = f.read()

        if nodes:
            nodes_str = json.dumps(nodes)
        else:
            nodes_str = "[]"

        tests_str = json.dumps(tests) if tests else "[]"

        tcp_interfaces = self._discover_network_interfaces()
        if not tcp_interfaces:
            raise ex.NetworkDiscoveryException(
                "Network discovery failed on Slurm. "
                "Could not identify required TCP interfaces."
            )

        tcp_interfaces_str = ",".join(tcp_interfaces)

        images_dir_value = (
            self._app_data and self._app_data.get_images_dir()
        ) or "{workload_input_dir}"

        rendered_content = template_content.format(
            nodes=nodes_str,
            machine_type=machine_type,
            tests=tests_str,
            images_dir=images_dir_value,
            tcp_interfaces=tcp_interfaces_str,
            timeout=timeout,
            deadline=deadline,
        )
        logger.debug("Configuration: %r", rendered_content)

        self._write_topology_json(workspace_config_dir)

        with open(os.path.join(workspace_config_dir, "ramble.yaml"), "w") as f:
            f.write(rendered_content)

    def check_jobs_running(self, job_ids: list[str]) -> bool:
        """Checks if the given jobs are still running."""
        if not job_ids:
            logger.warning(
                "No job IDs provided to check. Proceeding with analyzing logs. "
                "Did chs run submit jobs?"
            )
            return False

        cmd = ["squeue", "--jobs", ",".join(job_ids), "--noheader"]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False
            )
            if result.stdout.strip():
                logger.debug(
                    "Jobs are still running. Output: %s", result.stdout
                )
                return True
            logger.debug("No running jobs found in squeue.")
            return False
        except FileNotFoundError:
            logger.warning("squeue command not found. Cannot check job status.")
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
            logger.warning("sacct command not found. Assuming UNKNOWN.")
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
            logger.warning(
                "Log file for job %s not found at %s", job_id, log_path
            )

    def parse_job_ids(self, output: str) -> list[str]:
        """Parses job IDs from the experiment execution output."""
        return re.findall(r"Submitted batch job (\d+)", output)

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
                    "scancel returned non-zero exit code: %s", result.stderr
                )
        except FileNotFoundError:
            logger.warning("scancel command not found. Cannot cancel jobs.")
        except (OSError, subprocess.SubprocessError) as e:
            logger.error("Failed to cancel jobs. Error: %s", e)
            raise ex.ChsCancellationException(
                f"Failed to cancel jobs: {e}"
            ) from e
