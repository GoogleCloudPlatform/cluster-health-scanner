This folder provides dcgm and nccl test for slurm A3, A3Mega, A3Ultra and A4.

Steps to do before running the scripts:

*   Login to the Slurm login node.

*   Run `git clone
    https://github.com/GoogleCloudPlatform/cluster-health-scanner.git` to
    download code to your login node.

*   Run `cd cluster-health-scanner/deploy/slurm` to access to the available
    scripts.

*   Run `chmod +x cluster-validation.sh` to give execute permission. Note that
    `cluster-validation.sh` is the scheduler and uses other scripts internally.
    This is the only script you need to run. Example commands are shown below.

The examples in this directory are used to show how to test Slurm cluster node
health.

Contents:

*   `dcgmi_diag.sh`: A helping Slurm batch script for running the gpu test
    `dcgmi diag -r 3`.
*   `nccl-gcp.sh`: A helping Slurm batch script for running nccl test
    `all_gather_perf` benchmark.
*   `build-nccl-tests-a3mega.sh`: A Slurm batch script for building the
    nccl-tests for a3mega.
*   `build-nccl-tests-gib.sh`: A Slurm batch script for building the
    nccl-tests for a3ultra.
*   `cluster-validation.sh`: A Slurm batch script for running the gpu tests and
    nccl-tests `all_gather_perf` benchmark on the entire cluster or a subset of
    the cluster.

# Running GPU-Tests and/or NCCL-Tests on the Slurm cluster

* To run nccl test only on a3mega cluster:
`
 ./cluster-validation.sh --nccl --partition a3mega --machine-type a3mega \
  --nodes 10 --nodelist a3m123-a3meganodeset-[1-10]
`
* To run nccl test only on a3ultra cluster:
`
./cluster-validation.sh --nccl --partition a3ultra --machine-type a3ultra \
  --nodes 10 --nodelist a3m123-a3ultranodeset-[1-10]
`
* To run gpu test only on cluster:
`
 ./cluster-validation.sh --dcgm --partition a3mega --machine-type a3mega \
  --nodes 10 --nodelist a3m123-a3meganodeset-[1-10]
`
* To run both nccl test and gpu test on a3mega cluster:
`
 ./cluster-validation.sh --partition a3mega --machine-type a3mega \
  --nodes 10 --nodelist a3m123-a3meganodeset-[1-10]
`

If you would like to drain the bad nodes, add `--drain-bad-node` to the command.

Results are shown once the tests are done. An example of looks like:
`
Starting DCGM Diagnostics... DCGM diagnostics passing on all nodes!
Building the NCCL tests... NCCL tests built successfully! Starting NCCL
all_reduce_perf... CURR_NODES: a3m123-a3meganodeset-11 a3m123-a3meganodeset-12
CURR_BUSBW: 127 CURR_NODES: a3m123-a3meganodeset-13 a3m123-a3meganodeset-14
CURR_BUSBW: 128 CURR_NODES: a3m123-a3meganodeset-15 a3m123-a3meganodeset-16
CURR_BUSBW: 127 CURR_NODES: a3m123-a3meganodeset-17 a3m123-a3meganodeset-18
CURR_BUSBW: 127 CURR_NODES: a3m123-a3meganodeset-19 a3m123-a3meganodeset-20
CURR_BUSBW: 126 NCCL test passing on all nodes!
`

*Second round of NCCL test*

For any nodes that fail the first round of nccl test, run the following command
to drain them:
`
<!-- mdlint off(LINE_OVER_80) -->
 ./cluster-validation.sh --nccl --partition a3mega ---machine-type a3mega \
  --nodes 4 --nodelist a3m123-a3meganodeset-[1,5,7,8]
`

Detailed logs are stored under
`./cluster-health-scanner/results/cluster_validation`.
You will see
`dcgmi-<jobid>.out` and `nccl_gcp.sh_<jobid>.log` in this folder.
