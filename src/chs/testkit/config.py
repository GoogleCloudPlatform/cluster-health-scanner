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

import copy
import functools
import os
from collections.abc import Iterator, Mapping

import yaml

SCHEDULERS = ("kubernetes", "slurm", "standalone")

PLATFORMS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "platforms",
)

TESTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
)

DOMAIN_TYPES = ("intradomain", "interdomain")


def deep_merge(base: dict, layer: Mapping) -> dict:
    for key, value in layer.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_platform(machine_type: str) -> dict:
    return _load_yaml(os.path.join(PLATFORMS_ROOT, f"{machine_type}.yaml"))


def load_test_config(test_name: str) -> dict:
    return _load_yaml(os.path.join(TESTS_ROOT, test_name, "config.yaml"))


@functools.cache
def _load_yaml(path: str) -> dict:
    """Parses each file once; the files do not change during a run."""
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def supported_platforms(config: Mapping) -> set[tuple[str, str]]:
    """Returns the (scheduler, machine_type) pairs a test runs on.

    A machine listed by name runs on every scheduler in its platform file.
    A machine listed as ``{machine_type: [scheduler, ...]}`` runs only on
    those.
    """
    supported_pairs = set()
    for entry in config["platforms"]:
        if isinstance(entry, str):
            machine_type = entry
            schedulers = load_platform(machine_type)["schedulers"]
        else:
            ((machine_type, schedulers),) = entry.items()
        supported_pairs.update(
            (scheduler, machine_type) for scheduler in schedulers
        )
    return supported_pairs


def _layers_in_override_order(
    platform: Mapping, config: Mapping, machine_type: str, scheduler: str
) -> Iterator[dict]:
    """Yields configuration layers from broadest to narrowest."""
    yield {"variables": platform.get("variables", {})}
    yield {"variables": platform.get(scheduler, {}).get("variables", {})}

    defaults = config.get("defaults", {})
    yield _without_scheduler_blocks(defaults)
    yield _without_scheduler_blocks(defaults.get(scheduler, {}))

    machine_block = config.get("machines", {}).get(machine_type, {})
    yield _without_scheduler_blocks(machine_block)
    yield _without_scheduler_blocks(machine_block.get(scheduler, {}))


def _without_scheduler_blocks(block: Mapping) -> dict:
    return {key: value for key, value in block.items() if key not in SCHEDULERS}


def resolve(
    platform: Mapping,
    config: Mapping,
    machine_type: str,
    scheduler: str,
    domain_type: str | None = None,
) -> dict:
    """Merges the layers for one machine type and scheduler.

    Narrower layers override broader ones. Modifiers add up across layers
    instead: a new name is appended and a repeated name replaces the
    earlier entry in place.
    """
    resolved: dict = {}
    modifiers: dict = {}
    for layer in _layers_in_override_order(
        platform, config, machine_type, scheduler
    ):
        for modifier in layer.pop("modifiers", []):
            modifiers[modifier["name"]] = modifier
        deep_merge(resolved, layer)
    resolved["modifiers"] = copy.deepcopy(list(modifiers.values()))

    for domain in DOMAIN_TYPES:
        section = resolved.pop(domain, {})
        if domain == domain_type:
            deep_merge(
                resolved.setdefault("variables", {}),
                section.get("variables", {}),
            )

    return {
        "variables": resolved.pop("variables", {}),
        **{name: value for name, value in resolved.items() if value},
    }


def resolve_test(
    test_name: str,
    machine_type: str,
    scheduler: str,
    domain_type: str | None = None,
) -> dict:
    return resolve(
        load_platform(machine_type),
        load_test_config(test_name),
        machine_type,
        scheduler,
        domain_type,
    )
