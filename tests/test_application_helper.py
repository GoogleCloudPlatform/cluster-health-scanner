import unittest
import sys
from unittest.mock import MagicMock

# Mock ramble module
sys.modules["ramble"] = MagicMock()
sys.modules["ramble.context"] = MagicMock()

from chs.repos.applications.chs.test_helper import TestHelper  # noqa: E402


class TestApplicationHelper(unittest.TestCase):
    def test_merge_configs_precedence(self):
        # Suite variables (context) should take precedence over defaults
        # (configs)
        context_dict = {
            "variables": {"diag_level": [3], "other_var": "suite_val"}
        }

        configs = {
            "machine-type": {
                "app-name": {
                    "variables": {
                        "diag_level": [1, 2],
                        "default_var": "default_val",
                    }
                }
            }
        }

        TestHelper._merge_configs(
            context_dict, configs, "machine-type", "app-name"
        )

        self.assertEqual(
            context_dict["variables"]["diag_level"],
            [3],
            "Suite variable should be preserved",
        )
        self.assertEqual(
            context_dict["variables"]["other_var"],
            "suite_val",
            "Unique suite variable should be preserved",
        )
        self.assertEqual(
            context_dict["variables"]["default_var"],
            "default_val",
            "Default variable should be added",
        )

    def test_merge_configs_modifiers(self):
        # Modifiers should be extended
        context_dict = {"modifiers": [{"name": "suite-mod"}]}

        configs = {
            "machine-type": {
                "app-name": {"modifiers": [{"name": "default-mod"}]}
            }
        }

        TestHelper._merge_configs(
            context_dict, configs, "machine-type", "app-name"
        )

        expected_modifiers = [{"name": "suite-mod"}, {"name": "default-mod"}]
        self.assertEqual(
            context_dict["modifiers"],
            expected_modifiers,
            "Modifiers should be merged",
        )


if __name__ == "__main__":
    unittest.main()
