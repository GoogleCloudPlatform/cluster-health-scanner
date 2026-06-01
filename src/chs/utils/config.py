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

import functools
import logging
import os
import pathlib
from typing import Any
import yaml

logger = logging.getLogger(__name__)


@functools.lru_cache
def load_yaml_config(filename: str) -> dict[str, Any]:
    """Loads a YAML configuration file from the configs directory.

    Uses lru_cache to avoid repeated disk reads.
    """
    configs_dir = os.environ.get("CHS_CONFIGS_DIR")
    if configs_dir:
        path = pathlib.Path(configs_dir) / filename
    else:
        path = pathlib.Path(__file__).parent.parent / filename

    if not path.exists():
        raise FileNotFoundError(f"{filename} not found at {path}")
    with open(path, "r") as f:
        logger.debug("Loading config from %s", path)
        return yaml.safe_load(f) or {}
