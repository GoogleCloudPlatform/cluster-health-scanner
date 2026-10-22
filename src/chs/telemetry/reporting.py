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

"""Turns CLI state into telemetry metrics.

chs.telemetry only moves metrics; this module decides what they contain. Only
aggregate values are reported: flag names without their values, node counts
and zones, never node, cluster, bucket or custom run names.
"""

import collections
import datetime
import functools
import logging
from collections.abc import Callable
from typing import TypeVar

import click
from click.core import ParameterSource

from chs import telemetry
from chs.core.summary_aggregator import RunSummaryData, aggregate_boh_summary
from chs.protos import bill_of_health_pb2 as pb
from chs.schedulers.base import BaseScheduler
from chs.telemetry.supported_metrics import MetricName

logger = logging.getLogger(__name__)

_FLAGS_META_KEY = "chs.telemetry.command_flags"

_TEST_PROJECTS = {"hpc-toolkit-dev"}

_CLUSTER_TYPES = {"slurm": "SLURM", "kubernetes": "GKE"}

_INTERRUPTED_EXIT_CODE = 130

_SUPPORTED_COMMANDS = {"run", "results", "list", "cancel", "history"}

_HELP_AND_VERSION_FLAGS = {"-h", "--help", "--version"}

_R = TypeVar("_R")


def _never_raises(f: Callable[..., _R]) -> Callable[..., _R | None]:
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.debug("Telemetry reporting failed in %s: %r", f.__name__, e)
            return None

    return wrapper


def is_command_supported(args: list[str]) -> bool:
    if _HELP_AND_VERSION_FLAGS.intersection(args):
        return False
    commands = [arg for arg in args if not arg.startswith("-")]
    return bool(commands) and commands[0] in _SUPPORTED_COMMANDS


class TelemetryCommand(click.Command):
    def invoke(self, ctx: click.Context):
        record_command_flags(ctx)
        return super().invoke(ctx)


class TelemetryGroup(click.Group):
    command_class = TelemetryCommand
    group_class = type

    def main(self, *args, **kwargs):
        try:
            return super().main(*args, **kwargs)
        except SystemExit as e:
            code = e.code
            if not isinstance(code, int):
                code = 0 if code is None else 1
            telemetry.send_metric(MetricName.EXIT_CODE, code)
            raise
        except BaseException:
            telemetry.send_metric(MetricName.EXIT_CODE, 1)
            raise

    def resolve_command(self, ctx: click.Context, args: list[str]):
        cmd_name, cmd, args = super().resolve_command(ctx, args)
        if ctx.parent is None and cmd is not None:
            telemetry.send_metric(MetricName.PARENT_COMMAND, cmd.name)
        return cmd_name, cmd, args

    def invoke(self, ctx: click.Context):
        record_command_flags(ctx)
        if ctx.parent is not None:
            return super().invoke(ctx)
        try:
            return super().invoke(ctx)
        except click.exceptions.Exit as e:
            if e.exit_code == _INTERRUPTED_EXIT_CODE:
                telemetry.send_metric(
                    MetricName.ERROR_CATEGORY, KeyboardInterrupt.__name__
                )
            raise
        except (Exception, KeyboardInterrupt) as e:
            telemetry.send_metric(MetricName.ERROR_CATEGORY, type(e).__name__)
            raise


@_never_raises
def record_command_flags(ctx: click.Context) -> None:
    flags = ctx.meta.setdefault(_FLAGS_META_KEY, [])
    for param in ctx.command.params:
        if not isinstance(param, click.Option):
            continue
        if (
            ctx.get_parameter_source(param.name)
            is not ParameterSource.COMMANDLINE
        ):
            continue
        flag = max(param.opts, key=len).lstrip("-")
        if flag not in flags:
            flags.append(flag)
    if flags:
        flags_value = ",".join(sorted(flags))
        telemetry.send_metric(MetricName.COMMAND_FLAGS, flags_value)


@_never_raises
def record_run(
    scheduler_obj: BaseScheduler,
    storage_config: dict,
    machine_type: str,
    nodes: list[str],
    suite: str | None,
    test_names: list[str],
) -> None:
    telemetry.set_telemetry_project_id(scheduler_obj.project_id)
    telemetry.send_metric(
        MetricName.CLUSTER_TYPE,
        _CLUSTER_TYPES.get(scheduler_obj.name, "UNKNOWN"),
    )
    storage_type = storage_config.get("type") or "none"
    telemetry.send_metric(MetricName.STORAGE_TYPE, storage_type.upper())
    telemetry.send_metric(MetricName.MACHINE_TYPE, machine_type)
    telemetry.send_metric(MetricName.NODE_COUNT, len(nodes))
    zones = {
        scheduler_obj.node_to_zone[node]
        for node in nodes
        if node in scheduler_obj.node_to_zone
    }
    if zones:
        telemetry.send_metric(MetricName.ZONE, ",".join(sorted(zones)))
    is_test_data = scheduler_obj.project_id in _TEST_PROJECTS
    telemetry.send_metric(MetricName.IS_TEST_DATA, str(is_test_data).lower())
    if suite:
        telemetry.send_metric(MetricName.TEST_SUITE, suite)
    telemetry.send_metric(MetricName.TEST_NAME, ",".join(sorted(test_names)))


@_never_raises
def record_test_duration(
    submission_time: str | None,
    now: datetime.datetime | None = None,
) -> None:
    if not submission_time:
        return
    try:
        submitted_at = datetime.datetime.fromisoformat(submission_time)
    except ValueError:
        logger.debug("Unparseable submission time: %r", submission_time)
        return
    if submitted_at.tzinfo is None:
        submitted_at = submitted_at.replace(tzinfo=datetime.timezone.utc)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    duration_ms = int((now - submitted_at).total_seconds() * 1000)
    if duration_ms >= 0:
        telemetry.send_metric(MetricName.TEST_DURATION_MS, duration_ms)


def _test_results(summary: RunSummaryData) -> list[dict[str, str | int]]:
    return [
        {
            "test": test.canonical_name or "unknown",
            "level": test.test_level or "default",
            "unit": test.entity_noun,
            "total": test.denominator,
            "met_threshold": test.met_threshold,
            "unmet_threshold": test.unmet_threshold,
            "missing_metrics": test.missing_metrics,
            "failed_to_execute": test.failed_to_execute,
            "timeout": test.timeout_count,
            "deadline": test.deadline_count,
            "cancelled": test.cancelled_count,
        }
        for test in summary.test_summaries
    ]


def _detected_errors(summary: RunSummaryData) -> list[dict[str, str | int]]:
    tests = {
        test.display_name: (
            test.canonical_name or "unknown",
            test.test_level or "default",
        )
        for test in summary.test_summaries
    }
    nodes = collections.defaultdict(set)
    for event in summary.errors_and_warnings:
        test, level = tests.get(event.test, ("unknown", "default"))
        nodes[(test, level, event.type, event.category)].add(event.node)
    return [
        {
            "test": test,
            "level": level,
            "severity": severity,
            "category": category,
            "nodes": len(issue_nodes),
        }
        for (test, level, severity, category), issue_nodes in sorted(
            nodes.items()
        )
    ]


@_never_raises
def record_test_results(boh: pb.BillOfHealth) -> None:
    summary = aggregate_boh_summary(boh)
    if not summary:
        return
    telemetry.send_metric(MetricName.TEST_RESULTS, _test_results(summary))
    telemetry.send_metric(MetricName.DETECTED_ERRORS, _detected_errors(summary))
