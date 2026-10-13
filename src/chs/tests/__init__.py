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

"""Loads each test directory as a package under chs.tests.

Python cannot import names with hyphens, so nccl-cluster is loaded as
chs.tests.nccl_cluster.
"""

import importlib.util
import logging
import pathlib
import sys
from types import ModuleType

logger = logging.getLogger(__name__)


def _load_test_packages() -> list[ModuleType]:
    registered = []
    for test_dir in sorted(pathlib.Path(__file__).parent.iterdir()):
        init_file = test_dir / "__init__.py"
        if not init_file.is_file():
            continue
        package_name = test_dir.name.replace("-", "_")
        module_name = f"{__name__}.{package_name}"
        spec = importlib.util.spec_from_file_location(
            module_name, init_file, submodule_search_locations=[str(test_dir)]
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = package
        registered.append((package_name, module_name, spec, package))

    packages = []
    for package_name, module_name, spec, package in registered:
        try:
            spec.loader.exec_module(package)
        except Exception as e:
            del sys.modules[module_name]
            logger.warning(
                "Failed to load the Cluster Health Scanner (CHS) "
                "visualization plugin '%s'. Some UI features may be "
                "unavailable. Error: %s",
                module_name,
                e,
            )
            continue
        globals()[package_name] = package
        packages.append(package)
    return packages


TEST_PACKAGES = _load_test_packages()
