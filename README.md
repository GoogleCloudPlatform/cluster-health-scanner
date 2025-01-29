# cluster_diag Tool

A tool for diagnosing cluster issues.

## Installation

1. Clone this repository
2. From the _root dir_ of this repository, run `python3 cli/cluster_diag.py`

    NOTE: `cluster_diag` currently __only works__ from the root dir of this
          repo. See the Usage section for more details.

3. (Optional) Use an alias to simplify usage and store common flags.
    For example, if you only use clusters orchestrated by GKE, you can use:

    `alias cluster_diag="python3 cli/cluster_diag.py -o gke"`

## Usage

```bash
$ cluster_diag
Usage: cluster_diag.par [OPTIONS] COMMAND [ARGS]...

  A tool for diagnosing cluster issues.

Options:
  -o, --orchestrator [gke]  Cluster orchestrator type.  [required]
  --version                 Show the version and exit.
  --help                    Show this message and exit.

Commands:
  healthscan  Run a healthscan on a cluster.
```

### healthscan
Run a healthscan on a cluster.

```bash
$ cluster_diag -o gke healthscan a3-megagpu-8g --help
Usage: cluster_diag.par healthscan [OPTIONS] {a3-megagpu-8g}

Run a healthscan on a cluster.

Options:
  -c, --check [status|nccl|gpu|straggler|neper]
                                  Check to run. Available checks:

                                  - status: (Default) Checks the current healthscan status of the cluster.
                                  - nccl: Runs a pairwise NCCL bandwidth test on the cluster.
                                  - gpu: Runs a GPU check on the cluster.
                                  - straggler: Instruments a straggler check on the cluster.
                                  - neper: Runs a Network Performand eval on the cluster.
  -n, --nodes TEXT                Nodes to run checks on. Defaults to running
                                  on all nodes.
  --run_only_on_available_nodes   Force running the healthcheck only on
                                  available nodes. Unavailable nodes will be
                                  skipped.
  --help                          Show this message and exit.
```

#### Sample Usage

|Action                                            |Command to Run                                                                           |
|--------------------------------------------------|-----------------------------------------------------------------------------------------|
|Get cluster status                                |`$ cluster_diag -o gke healthscan a3-megagpu-8g -c status`                             |
|Running a DCGM/GPU check                          |`$ cluster_diag -o gke healthscan a3-megagpu-8g -c gpu`                                |
|Running a DCGM/GPU check _only on available nodes_|`$ cluster_diag -o gke healthscan a3-megagpu-8g -c gpu --run_only_on_available_nodes`|

## For Developers

See
[README-developers.md](https://github.com/GoogleCloudPlatform/cluster-health-scanner/blob/main/README-developers.md)
for developer details.