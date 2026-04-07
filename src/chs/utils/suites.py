import logging
import pathlib
import yaml
from typing import Any

from chs.exceptions import SuiteNotFoundException

logger = logging.getLogger(__name__)


def load_suite_tests(suite_name: str, configs_dir: str) -> list[dict[str, Any]]:
    """Loads a list of tests for a given suite from suites.yaml.

    Args:
        suite_name: The name of the suite to load.
        configs_dir: The directory containing suites.yaml.

    Returns:
        A list of test definitions corresponding to the suite.

    Raises:
        SuiteNotFoundException: If the suite is not found in suites.yaml.
        FileNotFoundError: If suites.yaml is not found.
    """
    suites_path = pathlib.Path(configs_dir) / "suites.yaml"
    if not suites_path.exists():
        raise FileNotFoundError(f"suites.yaml not found at {suites_path}")

    try:
        with open(suites_path, "r") as f:
            suites_data = yaml.safe_load(f) or {}

        suites = suites_data.get("suites", [])

        for suite in suites:
            if suite.get("name") == suite_name:
                return suite.get("tests", [])

        available_suites = [s.get("name") for s in suites]
        raise SuiteNotFoundException(
            f"Suite '{suite_name}' not found in suites.yaml. "
            f"Available suites: {', '.join(available_suites)}"
        )

    except Exception as e:
        logger.error("Failed to load suite '%s': %s", suite_name, e)
        raise
