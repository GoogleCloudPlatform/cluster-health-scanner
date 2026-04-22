import re
import unittest


class TestGenericErrorDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        cls.modifier_path = os.path.join(
            project_root,
            "src",
            "chs",
            "repos",
            "modifiers",
            "generic-error-detector",
            "modifier.py",
        )

        # Read the file content
        with open(cls.modifier_path, "r") as f:
            cls.content = f.read()

        # Extract regexes
        cls.cuda_error_regex = cls.extract_regex("CUDA Error Detected")
        cls.hardware_error_regex = cls.extract_regex("Hardware Error Detected")
        cls.system_error_regex = cls.extract_regex("System Error Detected")
        cls.host_oom_regex = cls.extract_regex("Host OOM Error Detected")
        cls.mpi_error_regex = cls.extract_regex("MPI Error Detected")
        cls.timeout_regex = cls.extract_regex("Timeout Error Detected")

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

    def test_cuda_error_regex(self):
        regex = self.cuda_error_regex
        self.assertTrue(re.match(regex, "cudaErrorMemoryAllocation"))
        self.assertTrue(re.match(regex, "cudaErrorIllegalAddress"))
        self.assertTrue(re.match(regex, "cudaErrorLaunchOutOfResources"))
        self.assertTrue(re.match(regex, "some prefix CUDA error some suffix"))
        self.assertTrue(re.match(regex, "Cuda failure"))
        self.assertFalse(re.match(regex, "Normal operation"))

    def test_hardware_error_regex(self):
        regex = self.hardware_error_regex
        self.assertTrue(re.match(regex, "Xid 31"))
        self.assertTrue(re.match(regex, "GPU has fallen off the bus"))
        self.assertTrue(re.match(regex, "Uncorrectable ECC"))
        self.assertTrue(re.match(regex, "ECC fail"))
        self.assertTrue(re.match(regex, "ECC_ERROR"))
        self.assertFalse(re.match(regex, "Normal operation"))
        self.assertFalse(re.match(regex, "ECC is ON"))
        self.assertFalse(re.match(regex, "ECC Errors"))

    def test_system_error_regex(self):
        regex = self.system_error_regex
        self.assertTrue(re.match(regex, "Segmentation fault"))
        self.assertTrue(re.match(regex, "Segfault"))
        self.assertTrue(re.match(regex, "core dumped"))
        self.assertTrue(re.match(regex, "Bus error"))
        self.assertFalse(re.match(regex, "Normal operation"))

    def test_host_oom_regex(self):
        regex = self.host_oom_regex
        self.assertTrue(re.match(regex, "Out of memory"))
        self.assertTrue(re.match(regex, "Killed"))
        self.assertFalse(re.match(regex, "Normal operation"))

    def test_mpi_error_regex(self):
        regex = self.mpi_error_regex
        self.assertTrue(re.match(regex, "MPI_Abort"))
        self.assertTrue(re.match(regex, "Fatal error in PMPI_Init"))
        self.assertTrue(re.match(regex, "primary job terminated"))
        self.assertFalse(re.match(regex, "Normal operation"))

    def test_timeout_regex(self):
        regex = self.timeout_regex
        self.assertTrue(re.match(regex, "Time limit exceeded"))
        self.assertTrue(re.match(regex, "watchdog timeout"))
        self.assertTrue(re.match(regex, "timed out"))
        self.assertFalse(re.match(regex, "Normal operation"))


if __name__ == "__main__":
    unittest.main()
