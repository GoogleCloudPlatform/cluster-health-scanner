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

"""A CLI for diagnosing cluster issues.

For more information, run:
    $ cluster_diag --help
"""

import click

import common
import configcheck
import healthscan

SUPPORTED_ORCHESTRATORS: frozenset[str] = common.SUPPORTED_ORCHESTRATORS


@click.group()
@click.option(
    '-o',
    '--orchestrator',
    required=True,
    type=click.Choice(
        list(SUPPORTED_ORCHESTRATORS),  # Must be a sequence for click.Choice
        case_sensitive=False,
    ),
    help='Cluster orchestrator type.',
)
@click.version_option(version='1.0.0')
@click.pass_context
def cluster_diag(ctx: click.Context, orchestrator: str):
  """A CLI for diagnosing cluster issues."""
  if orchestrator not in SUPPORTED_ORCHESTRATORS:
    raise ValueError(
        f'Unsupported orchestrator: {orchestrator}.'
        + f'Supported orchestrators are: {SUPPORTED_ORCHESTRATORS}'
    )
  ctx.obj = {'orchestrator': orchestrator}


cluster_diag.add_command(healthscan.cli)
cluster_diag.add_command(configcheck.cli)

if __name__ == '__main__':
  cluster_diag()
