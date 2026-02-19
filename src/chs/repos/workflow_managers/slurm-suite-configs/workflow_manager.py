import sys
from ramble.repository import ObjectTypes, paths

config_dir = (
    paths[ObjectTypes.workflow_managers]
    .repo_for_obj("slurm-suite-configs")
    .dirname_for_object_name("slurm-suite-configs")
)
sys.path.append(config_dir)
from nccl_config import NCCLConfig
from ramble.wmkit import *
from ramble.experiment_result import ExperimentStatus


class SlurmSuiteConfigs(WorkflowManagerBase):
    """Configurations for the various machine types supported by CHS in Slurm"""

    name = "slurm-suite-configs"

    tags("gcp-configs")

    maintainers("bytewin")

    variant(
        "machine_type",
        default="{machine_type}",
        description="Machine type for the suite",
    )

    variable(
        "machine_type",
        default="a3-ultra",
        description="Machine type for the suite",
        values=["a3-high", "a3-mega", "a3-ultra", "a4", "a4x", "tpuv7x"],
    )

    # Default values
    variable(
        "processes_per_node",
        default="1",
        description="Number of processes per node",
    )
    variable("n_ranks", default="1", description="Number of ranks")
    variable(
        "mpi_command",
        default="srun {srun_args}",
        description="slurm mpi command",
    )
    variable(
        "batch_submit",
        default="sbatch {slurm_experiment_sbatch}",
        description="slurm batch submit",
    )

    register_template(
        name="slurm_experiment_sbatch",
        src_path="slurm_experiment_sbatch.tpl",
    )

    with when("machine_type=a3-high"):
        variable("gpus_per_node", default="8", description="A3U gpus per node")

        with when("application_name=nccl-tests"):
            NCCLConfig.a3_high()

    with when("machine_type=a3-ultra"):
        variable("gpus_per_node", default="8", description="A3U gpus per node")

        with when("application_name=nccl-tests"):
            NCCLConfig.a3_ultra()

    @property
    def template_render_vars(self):
        vars = {}
        app_inst = self.app_inst
        expander = app_inst.expander
        # Adding pre-defined and custom headers
        pragmas = [
            "#SBATCH -N {n_nodes}",
            "#SBATCH -J {job_name}",
            "#SBATCH -o {experiment_run_dir}/slurm-%j.out",
            "#SBATCH -e {experiment_run_dir}/slurm-%j.err",
            "#SBATCH --ntasks-per-node {processes_per_node}",
            "#SBATCH --gpus-per-node {gpus_per_node}",
            "#SBATCH --exclusive",
        ]
        partition = expander.expand_var_name("slurm_partition")
        if partition:
            pragmas.append("#SBATCH -p {slurm_partition}")
        extra_headers = (
            expander.expand_var_name("extra_sbatch_headers").strip().split("\n")
        )
        pragmas = pragmas + extra_headers
        header_str = "\n".join(self.conditional_expand(pragmas))
        return {
            **vars,
            "workflow_pragmas": header_str,
        }

    def get_status(self, workspace):
        return ExperimentStatus.UNRESOLVED
