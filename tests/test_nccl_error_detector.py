import re
import unittest


class TestNcclErrorDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Path to the modifier file
        cls.modifier_path = (
            "src/chs/repos/modifiers/nccl-error-detector/modifier.py"
        )

        # Read the file content
        with open(cls.modifier_path, "r") as f:
            cls.content = f.read()

        # Extract regexes
        cls.network_error_regex = cls.extract_regex("Network Error Detected")
        cls.cuda_error_regex = cls.extract_regex("CUDA Error Detected")
        cls.topo_warn_regex = cls.extract_regex("Topology Warning Detected")
        cls.internal_error_regex = cls.extract_regex("Internal Error Detected")
        cls.other_error_regex = cls.extract_regex("Other NCCL Error Detected")

    @classmethod
    def extract_regex(cls, name):
        # Find the block for the given name
        # Example:
        # figure_of_merit(
        #     "Network Error Detected",
        #     fom_regex=r".*(NCCL failure: network error"
        #     r"|Connection refused|Connection timed out).*",
        #     ...
        # )
        pattern = rf'figure_of_merit\(\s*"{name}",\s*fom_regex=r"([^"]+)"'
        match = re.search(pattern, cls.content, re.DOTALL)
        if match:
            return match.group(1)
        raise ValueError(f"Could not find regex for {name}")

    def test_network_error_regex(self):
        regex = self.network_error_regex
        self.assertTrue(
            re.match(
                regex, "some prefix NCCL failure: network error some suffix"
            )
        )
        self.assertTrue(re.match(regex, "NCCL failure: remote error"))
        self.assertTrue(re.match(regex, "Connection refused"))
        self.assertTrue(re.match(regex, "Connection timed out"))
        self.assertFalse(re.match(regex, "Normal operation"))

    def test_cuda_error_regex(self):
        regex = self.cuda_error_regex
        self.assertTrue(re.match(regex, "NCCL failure: unhandled cuda error"))
        self.assertTrue(re.match(regex, "cudaErrorMemoryAllocation"))
        self.assertTrue(re.match(regex, "cudaErrorIllegalAddress"))
        self.assertTrue(
            re.match(
                regex,
                "a4xlustre-a4xnodeset1-5: Test NCCL failure common.cu:1279 "
                "'unhandled cuda error (run with '"
                "NCCL_DEBUG=INFO for details) / '",
            )
        )
        self.assertFalse(re.match(regex, "Normal operation"))

    def test_internal_error_regex(self):
        regex = self.internal_error_regex
        self.assertTrue(re.match(regex, "NCCL failure: system error"))
        self.assertTrue(re.match(regex, "NCCL failure: internal error"))
        self.assertTrue(re.match(regex, "NCCL failure: invalid argument"))
        self.assertTrue(re.match(regex, "NCCL failure: invalid usage"))
        self.assertTrue(
            re.match(
                regex,
                "allgather-worker-0-23: Test NCCL failure common.cu:432 "
                "'internal error - please report this '"
                "issue to the NCCL developers / '",
            )
        )
        self.assertFalse(re.match(regex, "Normal operation"))

    def test_topo_warn_regex(self):
        regex = self.topo_warn_regex
        self.assertTrue(re.match(regex, "Topology detection failed"))
        self.assertTrue(re.match(regex, "via PXB"))
        self.assertTrue(re.match(regex, "via PHB"))
        self.assertTrue(re.match(regex, "via SYS"))
        self.assertFalse(re.match(regex, "Normal operation"))

    def test_other_error_regex(self):
        regex = self.other_error_regex
        self.assertTrue(re.match(regex, "NCCL ERROR: some generic error"))
        self.assertTrue(re.match(regex, "NCCL FATAL error"))
        self.assertFalse(re.match(regex, "NCCL failure: some generic error"))
        self.assertFalse(re.match(regex, "NCCL failure: operation failed"))
        self.assertFalse(re.match(regex, "NCCL failure: job aborted"))
        self.assertFalse(re.match(regex, "NCCL INFO: Failed to find symbol"))
        self.assertFalse(re.match(regex, "Normal operation"))


if __name__ == "__main__":
    unittest.main()
