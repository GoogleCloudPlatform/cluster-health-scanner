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

"""Runs the CHS CLI in its own process behind the integration mocks.

Usage:
  python -m tests.integration.test_utils.mocked_chs_cli CONFIG -- CHS_ARGS...

Separate processes behave like separate users invoking CHS against the same
cluster, which in-process runs cannot reproduce because the mocks, stdout
redirection and signal handlers are process-wide.
"""

import sys

from chs.cli import cli
from tests.integration.create_kind_cluster import resolve_config_path
from tests.integration.mocks.output_generator import RambleOutputGenerator
from tests.integration.test_utils import integration_test_util as test_utils
import yaml


def main(argv: list[str]) -> None:
    separator = argv.index("--")
    config_path_str = argv[separator - 1]
    chs_args = argv[separator + 1 :]

    config = yaml.safe_load(resolve_config_path(config_path_str).read_text())
    cluster_name = config.get("cluster_name", "chs-cluster")
    output_gen = RambleOutputGenerator(config_path_str)

    with test_utils.mock_chs_environment(cluster_name, output_gen):
        cli(chs_args, prog_name="chs")


if __name__ == "__main__":
    main(sys.argv)
