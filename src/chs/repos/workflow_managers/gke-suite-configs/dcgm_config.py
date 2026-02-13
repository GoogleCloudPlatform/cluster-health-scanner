from ramble.wmkit import *
class DCGMConfig:
    @staticmethod
    def standard():
        variable("jobset_template", default="jobset_standard.yaml.tpl", description="Jobset Template for DCGM")
        variable("container_image", default="nvcr.io/nvidia/cloud-native/dcgm:4.5.1-1-ubi9", description="Container Image")
        variable("processes_per_node", default="1", description="Processes per node")
        variable("n_ranks", default="1", description="Total number of ranks")
        variable("mpi_command", default="", description="MPI Execution Command")
