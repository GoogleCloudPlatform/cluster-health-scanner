import re
import unittest
import os


class TestDcgmErrorDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        cls.modifier_path = os.path.join(
            project_root,
            "src",
            "chs",
            "repos",
            "modifiers",
            "dcgm-error-detector",
            "modifier.py",
        )

        # Read the file content
        with open(cls.modifier_path, "r") as f:
            cls.content = f.read()

        # Extract regexes
        cls.dcgm_contention_regex = cls.extract_regex(
            "DCGM Resource Contention Error Detected"
        )
        cls.nvml_error_regex = cls.extract_regex("NVML Error Detected")
        cls.page_retirement_regex = cls.extract_regex(
            "DCGM Page Retirement Error Detected"
        )
        cls.ecc_error_regex = cls.extract_regex("DCGM ECC Error Detected")
        cls.pcie_error_regex = cls.extract_regex("DCGM PCIe Error Detected")
        cls.thermal_error_regex = cls.extract_regex(
            "DCGM Thermal Error Detected"
        )
        cls.power_error_regex = cls.extract_regex("DCGM Power Error Detected")
        cls.nvlink_error_regex = cls.extract_regex("DCGM NVLink Error Detected")

    @classmethod
    def extract_regex(cls, name):
        pattern = (
            rf'figure_of_merit\(\s*"{name}",\s*'
            r'fom_regex=r"((?:\(\?i\))?\(\?P<[^>]+>[^"]+\))"'
        )
        match = re.search(pattern, cls.content, re.DOTALL)
        if match:
            return match.group(1)
        raise ValueError(f"Could not find regex for {name}")

    def test_dcgm_contention_regex(self):
        regex = self.dcgm_contention_regex
        self.assertTrue(re.match(regex, "DCGM_ST_IN_USE"))
        self.assertTrue(re.match(regex, "DCGM_ST_BAD_PARAMETER"))
        self.assertTrue(
            re.match(regex, "some prefix DCGM_ST_IN_USE some suffix")
        )
        self.assertFalse(re.match(regex, "Normal operation"))

    def test_nvml_error_regex(self):
        regex = self.nvml_error_regex
        self.assertTrue(
            re.match(
                regex,
                "NVML: Unable to retrieve NVLink information"
                " as all links are inActive",
            )
        )
        self.assertTrue(
            re.match(
                regex,
                "Failed to initialize NVML: Driver/library version mismatch",
            )
        )
        self.assertTrue(re.match(regex, "some prefix NVML error some suffix"))
        self.assertFalse(re.match(regex, "Normal operation"))

    def test_page_retirement_regex(self):
        regex = self.page_retirement_regex
        self.assertTrue(
            re.match(
                regex,
                "DCGM reports that the maximum number of retired pages "
                "has been reached.",
            )
        )
        self.assertTrue(re.match(regex, "maximum number of retired pages"))
        self.assertFalse(re.match(regex, "Normal operation"))

    def test_ecc_error_regex(self):
        regex = self.ecc_error_regex
        self.assertTrue(
            re.match(
                regex,
                "Detection of double-bit errors that the hardware "
                "cannot fix automatically.",
            )
        )
        self.assertTrue(re.match(regex, "ECC error"))
        self.assertTrue(re.match(regex, "Double-bit error"))
        self.assertTrue(re.match(regex, "uncorrectable error"))
        self.assertFalse(re.match(regex, "Normal operation"))

    def test_pcie_error_regex(self):
        regex = self.pcie_error_regex
        self.assertTrue(
            re.match(regex, "The PCIe throughput is lower than expected")
        )
        self.assertTrue(re.match(regex, "PCIe link width has downgraded"))
        self.assertFalse(re.match(regex, "Normal operation"))
        self.assertFalse(re.match(regex, "Current PCIe link width is x16"))
        self.assertFalse(re.match(regex, "Measured PCIe throughput: 12.5 GB/s"))

    def test_thermal_error_regex(self):
        regex = self.thermal_error_regex
        self.assertTrue(
            re.match(
                regex, "The test reports that the GPU reached its thermal limit"
            )
        )
        self.assertTrue(re.match(regex, "thermal throttling detected"))
        self.assertFalse(re.match(regex, "Normal operation"))

    def test_power_error_regex(self):
        regex = self.power_error_regex
        self.assertTrue(
            re.match(regex, "The GPU cannot maintain the requested power draw")
        )
        self.assertTrue(re.match(regex, "power limit reached"))
        self.assertFalse(re.match(regex, "Normal operation"))
        self.assertFalse(re.match(regex, "Power draw: 200W"))
        self.assertFalse(re.match(regex, "Power limit is set to 300W"))

    def test_nvlink_error_regex(self):
        regex = self.nvlink_error_regex
        self.assertTrue(re.match(regex, "High CRC Error counts"))
        self.assertTrue(re.match(regex, "link down status"))
        self.assertFalse(re.match(regex, "Normal operation"))


if __name__ == "__main__":
    unittest.main()
