import sys
import os
from ramble.repository import ObjectTypes, paths

config_dir = paths[ObjectTypes.applications].repo_for_obj("chs-gke").dirname_for_object_name("chs-gke")
sys.path.append(config_dir)

from ramble.appkit import *
from test_helper import TestHelper
from ramble.context import create_context_from_dict
from copy import deepcopy

class ChsGke(ExecutableApplication):
    """GKE variant of cluster health scanner for running diagnostics """
    
    name = "chs-gke"
    maintainers("bytetwin")
    
    tags("gke", "chs-diagnostics")
    
    executable('cluster_health', template=["echo 'Initiating GKE cluster health diagnostics for {machine_type}'"])    

    workload('cluster_health', executables=["cluster_health"])

    def create_experiment_chain(self, workspace):
        
        tests = self.expander.expand_var_name("tests", typed=True)
        if not tests:
            logger.warn(f"No tests defined for {self.name}. Skipping experiments generation.")
            return

        experiment_name = self.expander.experiment_name
        for test in tests:
            remove_keys = {"n_ranks", "processes_per_node", "n_nodes", "jobset_template", "mpi_command", "container_image"}
            test_vars = self.non_reserved_variables(workspace, remove_keys=remove_keys)
            
            if test == "dcgm":
                context = TestHelper.dcgm_test(test_vars, experiment_name)
                self.experiment_set.render_chained_experiments(
                    "nvidia-dcgm", "diag", context
                )
            elif test == "nccl-cluster":
                workloads = ["all-gather", "all-reduce"]
                for workload in workloads:
                    context = TestHelper.nccl_test(test_vars, experiment_name, workload)
                    self.experiment_set.render_chained_experiments(
                        "nccl-tests", workload, context
                    )
            elif test == "nccl-pairwise":
                workloads = ["all-gather", "all-reduce"]
                for workload in workloads:
                    contexts = TestHelper.nccl_pairwise(test_vars, experiment_name, workload)
                    for ctx in contexts:
                        self.experiment_set.render_chained_experiments(
                            "nccl-tests", workload, ctx
                        )
            else:
                logger.warn(f"Unknown test {test} requested. Skipping.")
