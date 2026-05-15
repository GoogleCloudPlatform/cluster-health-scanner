from dataclasses import dataclass
import logging
from typing import Any

from chs.utils.config import load_yaml_config

logger = logging.getLogger(__name__)


def _load_tests() -> dict[str, Any]:
    """Loads the full tests configuration from tests.yaml."""
    return load_yaml_config("tests.yaml")


def get_all_supported_tests() -> list[str]:
    """Derives list of all supported tests from tests.yaml."""
    data = _load_tests()
    return [test.get("name") for test in data.get("tests", [])]


def get_all_supported_machine_types() -> list[str]:
    """Derives list of all supported machine types from tests.yaml."""
    tests_map = get_tests_map()
    machine_types = set()
    for scheduler in tests_map.data:
        machine_types.update(tests_map.data[scheduler].keys())
    return sorted(list(machine_types))


def get_all_supported_schedulers() -> list[str]:
    """Derives list of all supported schedulers from tests.yaml."""
    data = _load_tests()
    schedulers = set()
    for test in data.get("tests", []):
        for platform in test.get("platforms", []):
            schedulers.add(platform.get("scheduler"))
    return sorted(list(schedulers))


@dataclass
class TestsMap:
    """Represents a mapping of tests by scheduler and machine type."""

    data: dict[str, dict[str, list[str]]]

    def get(self, scheduler: str) -> dict[str, list[str]] | None:
        """Returns mapping of tests by machine type for scheduler."""
        return self.data.get(scheduler)


def get_tests_map() -> TestsMap:
    """Derives mapping of tests by scheduler and machine type."""
    data = _load_tests()
    tests_map = {}
    for test in data.get("tests", []):
        name = test.get("name")
        for platform in test.get("platforms", []):
            scheduler = platform.get("scheduler")
            mt = platform.get("machine_type")

            if scheduler not in tests_map:
                tests_map[scheduler] = {}
            if mt not in tests_map[scheduler]:
                tests_map[scheduler][mt] = []
            if name not in tests_map[scheduler][mt]:
                tests_map[scheduler][mt].append(name)
    return TestsMap(data=tests_map)


def get_tests_by_machine_type(machine_type: str) -> list[str]:
    """Returns list of tests supported on a specific machine type."""
    tests_map = get_tests_map()
    tests = set()
    for scheduler in tests_map.data:
        if machine_type in tests_map.data[scheduler]:
            tests.update(tests_map.data[scheduler][machine_type])
    return sorted(list(tests))
