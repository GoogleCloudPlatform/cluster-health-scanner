from ramble.modkit import *


class GpuPersistenceMode(BasicModifier):
    """Modifier to ensure NVIDIA persistence mode is enabled before execution."""

    name = "gpu-persistence-mode"

    tags("gpu")

    maintainers("bytetwin")

    mode("standard", description="Standard injection mode")
    default_mode("standard")

    register_builtin("source_pm", injection_method="prepend")

    def source_pm(self):
        return ["nvidia-smi -pm 1 || true"]
