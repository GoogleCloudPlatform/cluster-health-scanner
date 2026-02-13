from ramble.wmkit import *


class NCCLConfig:
  """Helper class for NCCL configurations on GKE."""

  @staticmethod
  def a3_ultra():
    """Define environment variables and variables for A3 Ultra NCCL configuration."""

    # MPI and CUDA Env Vars
    environment_variable(
        "OMPI_ALLOW_RUN_AS_ROOT",
        "1",
        description="Allow OMPI as root",
        method="set",
        workload=None,
    )
    environment_variable(
        "CUDA_VISIBLE_DEVICES",
        "0,1,2,3,4,5,6,7",
        description="CUDA Visible Devices",
        method="set",
        workload=None,
    )
    # Container and Jobset config
    variable(
        "jobset_template",
        default="jobset_mpi.yaml.tpl",
        description="Jobset Template for NCCL",
    )
    variable(
        "container_image",
        default="us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-diagnostic:v1.1.0",
        description="Container Image",
    )
    variable("ssh_port", default="2222", description="SSH port for MPI")

    # Execution scaling
    variable(
        "processes_per_node",
        default="{gpus_per_node}",
        description="Processes per node",
    )
    variable(
        "n_ranks",
        default="{n_nodes} * {processes_per_node}",
        description="Total number of MPI ranks",
    )

    # MPI command config
    variable(
        "nccl-tests_path",
        default="/third_party/nccl-tests",
        description="Path to NCCL tests in container",
    )
    variable(
        "mpi_env_vars",
        default=(
            "$(echo ${!NCCL*} ${!OMPI*} LD_LIBRARY_PATH | sed 's/ / -x /g')"
        ),
        description="Propagated MPI Env Vars",
    )

    mpi_cmd = (
        "mpirun -n {n_ranks} -N {processes_per_node} \\\n  --hostfile"
        " /tmp/hostfile --bind-to none \\\n  --mca btl self,tcp --mca"
        " btl_tcp_if_include eth0 \\\n  --mca orte_keep_fqdn_hostnames 1 --mca"
        ' plm_rsh_no_tree_spawn 1 \\\n  --mca plm_rsh_agent "ssh -q -p'
        ' {ssh_port} -o StrictHostKeyChecking=no -o BatchMode=yes" \\\n  -x'
        ' CUDA_VISIBLE_DEVICES \\\n  -x PATH="{nccl-tests_path}/build:$PATH"'
        " \\\n  -x $(echo ${!NCCL*} ${!OMPI*} LD_LIBRARY_PATH | sed 's/ / -x"
        " /g')"
    )
    variable(
        "mpi_command", default=mpi_cmd, description="MPI Execution Command"
    )
