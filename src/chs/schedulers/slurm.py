"""Slurm scheduler implementation for Cluster Health Scanner."""

import importlib.resources as pkg_resources
import logging
import os
import re
import subprocess
import json

from chs.schedulers.base import BaseScheduler
import click

logger = logging.getLogger(__name__)


class SlurmScheduler(BaseScheduler):
    """Slurm scheduler implementation."""

    def __init__(
        self,
        nodes: str | None = None,
        nodelist: str | None = None,
        partition: str | None = None,
    ):
        self.nodes_arg = nodes
        self.nodelist_arg = nodelist
        self.partition_arg = partition
        self._resolved_nodes: list[str] | None = None

    @property
    def name(self) -> str:
        return "slurm"

    def get_nodes_under_test(self) -> list[str]:
        if self._resolved_nodes is not None:
            return self._resolved_nodes

        provided = [
            arg
            for arg in [self.nodes_arg, self.nodelist_arg, self.partition_arg]
            if arg is not None
        ]
        if len(provided) > 1:
            raise click.UsageError(
                "Only one of --nodes, --nodelist, or --partition can be "
                "provided."
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
                raise click.ClickException(f"Failed to expand nodelist: {e}")

        # If partition is given or nothing is given (all free nodes)
        sinfo_cmd = ["sinfo", "-t", "idle", "-h", "-o", "%N"]
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
            raise click.ClickException(f"Failed to get free nodes: {e}")

        if not output:
            self._resolved_nodes = []
            return self._resolved_nodes

        all_nodes = []
        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                out2 = subprocess.run(
                    ["scontrol", "show", "hostname", line],
                    text=True,
                    check=True,
                    capture_output=True,
                ).stdout.strip()
                if out2:
                    all_nodes.extend([n for n in out2.split("\n") if n])
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.warning("Failed to expand nodes %s: %s", line, e)

        self._resolved_nodes = all_nodes
        return self._resolved_nodes

    def write_ramble_config(
        self,
        workspace_config_dir: str,
        tests: list[str] | None,
        machine_type: str,
    ) -> None:
        nodes = self.get_nodes_under_test()
        logger.debug("Configuring workspace with slurm_template.yaml")

        import chs.schedulers

        with pkg_resources.path(chs.schedulers, "slurm_template.yaml") as p:
            with open(p, "r") as f:
                template_content = f.read()

        if nodes:
            nodes_str = json.dumps(nodes)
        else:
            nodes_str = "[]"

        if tests:
            # Compact JSON representation for tests to avoid issues with
            # newlines
            tests_str = json.dumps(tests)
        else:
            tests_str = "[]"

        rendered_content = template_content.format(
            nodes=nodes_str,
            machine_type=machine_type,
            tests=tests_str,
        )

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
        """Gets the status of the given jobs using squeue."""
        if not job_ids:
            return {}

        statuses = {}
        # squeue format: JobID,StateCompact
        cmd = [
            "squeue",
            "--jobs",
            ",".join(job_ids),
            "--noheader",
            "--format=%i,%t",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "," in line:
                        jid, state = line.split(",", 1)
                        # Map Slurm states to common statuses if needed,
                        # but raw states are usually fine (R, PD, CG, etc.)
                        statuses[jid.strip()] = state.strip()

            # Jobs not in squeue are likely completed (or invalid).
            # We can't strictly distinguish without accounting, but for CHS
            # polling logic, absence implies completion.
            for jid in job_ids:
                if jid not in statuses:
                    statuses[jid] = "COMPLETED"

        except FileNotFoundError:
            logger.warning("squeue command not found. Assuming UNKNOWN.")
            for jid in job_ids:
                statuses[jid] = "UNKNOWN"

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
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug("Successfully canceled jobs: %s", job_ids)
        except FileNotFoundError:
            logger.warning("scancel command not found. Cannot cancel jobs.")
        except subprocess.CalledProcessError as e:
            logger.error("Failed to cancel jobs. Error: %s", e.stderr)
            raise click.ClickException(
                f"Failed to cancel jobs: {e.stderr}"
            ) from e
