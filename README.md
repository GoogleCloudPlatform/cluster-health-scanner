# Cluster Health Scanner

## 1. Introduction

The
[Cluster Health Scanner (CHS)](https://github.com/GoogleCloudPlatform/cluster-health-scanner)
is a tool that checks the health of a GPU cluster. It runs various tests to
ensure the cluster is ready for training workloads, specifically:

*   **NCCL check**: Validates the network communication between GPUs using the
    NCCL library.
*   **GPU check**: Utilizes NVIDIA's DCGM tool to diagnose the health of
    individual GPUs.
*   **Neper check**: Leverages the Neper tool to assess network performance
    within the cluster.
*   **Straggler detection**: Runs a network traffic pattern between nodes that
    closely resemble patterns seen during LLM training workload pipeline
    parallelism.
*   **Tinymax check**: Uses Maxtext open source LLM framework to assess ml
    training within the cluster.

CHS serves two main purposes:

*   **Proactive health checks**: Ensures the cluster is in optimal condition for
    running training workloads.
*   **Debugging tool**: Helps diagnose issues when you encounter problems with a
    training workload.

__GPU Cluster availability__: A3, A3+, A3U and A4<br>
__Orchestrator support__: GKE and Slurm

The **Cluster Health Scanner** tool or simply **CHS** runs a series of tests
called _health checks_ to analyze the health of a cluster of GPU nodes.

For instructions on how to run CHS, go directly to the
['Running CHS' section](#3-advanced-running-via-helm).

While currently structured for Google Kubernetes Engine (GKE), CHS can
theoretically run on clusters using other Kubernetes orchestration
implementations.
We have enabled CHS to run on additional cluster orchestrators, such
as [Slurm](https://github.com/GoogleCloudPlatform/cluster-health-scanner/tree/main/deploy/slurm/README.md)
for HPC.

## 2. Running via `cluster_diag`

A tool for diagnosing cluster issues.

The `cluster_diag` tool is a helpful wrapper around the CHS diagnostic tool for
the
[Accelerator-optimized machine family](https://cloud.google.com/compute/docs/accelerator-optimized-machines)
. It is exposed via the
`healthscan` command and aims to provide a _single line_ command that can run
CHS, with no prior knowledge needed of CHS implementation details. See the
documentation for `healthscan` below for more details on supported machines and
tests.

NOTE: The `cluster_diag` tool aims to provide a joyful experience for running
Cluster Health Scanner; however it may not support all use cases. To run CHS
directly, see the instructions
[via the instructions in the developer guide](https://github.com/GoogleCloudPlatform/cluster-health-scanner/blob/main/README-developer.md#32-running-chs)
.

### Installation

Note the following:

* `cluster_diag` expects that you have already authenticated the `gcloud` cli to access the Google Cloud Platform with Google user credentials.
* `gcloud` should already have credentials for the cluster-under-test.
* Python >= 3.9 version is required on the host machine to run the `cluster_diag` CLI.

1. Clone this repository
2. If you don't already have them, install dependencies for the CLI:
   
   ```
   pip3 install -r cli/requirements.txt
   ```

3. (Optional) Add your GCloud SSH key to your local SSH Agent:

```bash
ssh-add ~/.ssh/google_compute_engine
```

This will allow the configcheck command to fetch configuration values from
your cluster without needing to reauthenticate for each machine.

4. From the _root dir_ of this repository, run `python3 cli/cluster_diag.py`

    NOTE: `cluster_diag` currently __only works__ from the root dir of this
          repo. See the [Usage](#usage) section for more details.

5. (Optional) Use an alias to simplify usage and store common flags.
    For example, if you only use clusters orchestrated by GKE, you can use:

    `alias cluster_diag="python3 cli/cluster_diag.py -o gke"`

### Usage

```bash
$ cluster_diag --help
Usage: cluster_diag [OPTIONS] COMMAND [ARGS]...

  A tool for diagnosing cluster issues.

Options:
  -o, --orchestrator [gke|slurm]  Cluster orchestrator type.  [required]
  --version                 Show the version and exit.
  --help                    Show this message and exit.

Commands:
  configcheck  Run a configcheck on a cluster.
  healthscan  Run a healthscan on a cluster.
```

### healthscan
Runs a CHS healthscan on a cluster.

```bash
$ cluster_diag -o gke healthscan a3-megagpu-8g --help
Usage: cluster_diag healthscan [OPTIONS] {a3-highgpu-8g | a3-megagpu-8g | a3-ultragpu-8g | a4-highgpu-8g}

Run a healthscan on a cluster.

Options:
  -c, --check [status|nccl|gpu|straggler|neper|tinymax]
                                  Check to run. Available checks:

                                  - status: (Default) Checks the current healthscan status of the cluster.
                                  - nccl: Runs a pairwise NCCL bandwidth test on the cluster.
                                  - gpu: Runs a GPU check on the cluster.
                                  - straggler: Instruments a straggler check on the cluster.
                                  - neper: Runs a Network Performance eval on the cluster.
                                  - tinymax: Runs a LLM small training workload on the cluster.
  -n, --nodes TEXT                Nodes to run checks on. Defaults to running
                                  on all nodes. When using slurm, a shortened
                                  node format can be used. For example,
                                  "node-[0-1]"
  --run_only_on_available_nodes   Force running the healthcheck only on
                                  available nodes. Unavailable nodes will be
                                  skipped.
  --dry_run                       Run the healthcheck in dry run mode. This
                                  will print the commands that would be run,
                                  but not run them.
  --help                          Show this message and exit.
```

#### Sample Usage

|Action                                            |Command to Run                                                                           |
|--------------------------------------------------|-----------------------------------------------------------------------------------------|
|Get GKE cluster status                            |`$ cluster_diag -o gke healthscan a3-megagpu-8g -c status`                             |
|Running a DCGM/GPU check                          |`$ cluster_diag -o gke healthscan a3-megagpu-8g -c gpu`                                |
|Running a DCGM/GPU check _only on available nodes_|`$ cluster_diag -o gke healthscan a3-megagpu-8g -c gpu --run_only_on_available_nodes`|
|Running a DCGM/GPU check on two Slurm Nodes       |`$ cluster_diag -o slurm healthscan a3-megagpu-8g -c gpu -n node-[0-1]`|
|Dry run of a DCGM/GPU check                       |`$ cluster_diag -o slurm healthscan a3-megagpu-8g -c gpu -n node-[0-1] --dry_run`|

### configcheck
Check the configuration of your cluster and workload container.

```bash
$ cluster_diag -o gke configcheck --help
Usage: cluster_diag.py configcheck [OPTIONS] {a3-megagpu-8g}

  Run a configcheck on a cluster.

Options:
  -n, --nodes TEXT                A comma-separated list of nodes to run
                                  checks on. Defaults to running on all nodes.
  --skip_diff, --nodiff           If true, only print the node configs without
                                  diffing against the golden config.
  --run_async, --async            [Experimental] If true, run the configcheck
                                  in async mode. This will reset your terminal
                                  as part of the process.
  --project TEXT                  The project of the workload to be checked.
                                  If not specified, the project will be
                                  inferred from `gcloud config get project`
  --zone TEXT                     The zone of the workload to be checked. If
                                  not specified, the zone will be inferred per
                                  node from the `topology.kubernetes.io/zone`
                                  label.
  --workload_container TEXT       The name of the workload container to fetch
                                  workload configs from. If not specified, the
                                  workload container will be inferred from the
                                  node.
  --output_format [markdown|json]
                                  The format to print the output in. Defaults
                                  to markdown. Other supported formats are
                                  `csv` and `json`.
  --help                          Show this message and exit.

```

NOTE: `configcheck` for Slurm is supported in the following scenarios:

*   Running from a Slurm login node.
*   Providing a list of nodes using the `--nodes` flag.
If `sinfo` is not
found (e.g., you are not on a Slurm login node), the `configcheck` CLI will:

*   Print an error message to the user, indicating that `sinfo` was not found.
*   Exit with a success code (0).

#### Sample Usage

|Action                                                                                                                                |Command to Run                                                                         |
|--------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
|Check the configuration of the current cluster and diff against a qualified source of truth.                                          |`$ cluster_diag -o gke configcheck a3-megagpu-8g`                                      |
|Pull the configuration of the current cluster, and simply print it without diffing.                                                   |`$ cluster_diag -o gke configcheck a3-megagpu-8g --skip_diff`                          |
|Pull the configuration of the current cluster, including workload specific information for the 'sample-container' cluster.            |`$ cluster_diag -o gke configcheck a3-megagpu-8g --workload_container sample-container`|
|[Experimental, will reset cluster] Asynchronously pull the configuration information for a cluster and diff against a source of truth.|`$ cluster_diag -o gke configcheck a3-megagpu-8g --async`                              |

## 3. (Advanced) Running via 'helm'

WARNING: Running CHS directly as described in this section is not preferred, and
         could result in unintended behavior. Please read this entire section
         before continuing.

Running CHS only requires installing the Health Runner on the cluster.

The Health Runner will be able to launch health checks on the cluster based on
the user's installation configuration.

> Note:
> Currently this is done on GKE/Kubernetes using Helm charts.
> The description below focuses on running CHS using Helm charts on GKE.

### 3.1 Setup

#### Labeling Nodes to be Tested

Nodes to be included in a health check are marked using a corresponding node
label.

The node label keys depend on the health check,
with expected values of `"true"`:

- NCCL Health Check: `aiinfra/nccl-healthcheck-test`
- GPU Health Check: `aiinfra/gpu-healthcheck-test`
- Neper Health Check: `aiinfra/neper-healthcheck-test`
- Tinymax Health Check: `aiinfra/tinymax-healthcheck-test`
These label keys & values can be set using the `kubectl` tool
using the following command:

```bash
kubectl label nodes \
    --all \
    aiinfra/nccl-healthcheck-test="true"
```

> Note:
> This sets all nodes to be labeled for the NCCL health check.

#### Troubleshooting with Google Cloud Logging (Optional)

To assist Google Cloud engineers in diagnosing and resolving potential issues
with your cluster, you can configure CHS to send its logs to Google Cloud
Support. This allows our engineers to access only CHS logs, isolating them from
the rest of the cluster's logs.

Currently, this feature is supported for Google Kubernetes Engine (GKE)
clusters. Support for Slurm clusters will be added in the near future.

To configure this, follow these steps:

* Create a Service Account: In the Google Cloud project where your cluster resides, create a dedicated service account for sending CHS logs to Google.
* Contact Cloud Google Support: Contact Google Cloud Support and provide the name of the service account you created. They will grant the necessary permissions for this service account to write logs to Google Cloud Logging. It can take upto 8-hours for permissions to take effect.
* Generate a Service Account Key: Generate a JSON key file for the service account. CHS will use this to authenticate with Google Cloud Logging.
* Create a Kubernetes Secret: Use the following command to create a Kubernetes secret containing the service account key generated in the previous step:

  `kubectl create secret generic fluentbit-key --from-file=key.json=key.json`

By following these steps, you can enable log forwarding of CHS logs to Google
and help our engineers provide you with the best possible support for your
cluster.

#### Configuration of the Health Runner & Health Checks

The user can configure the Health Runner via the command line or as part of a
YAML configuration file. This configuration also gives the settings for the
health checks to be run.

Refer to the [_'Default Configuration'_ section](#default-configuration) for an
example of a full configuration file.

The following are the Health Runner configuration options:

##### `health_runner.name`

This will be used as the name of the Kubernetes Job for the Health Runner.

```yaml
health_runner:
  name: "health-runner"
```

##### `health_checks.HC_NAME`

Each health check is listed under the `health_checks` section. It is specific
to each health check though there are specific settings that apply to all
health checks.

Note in the section below we use the placeholder `HC_NAME` that would
be replaced with an identifying name of a health check, such as
`nccl_healthcheck`.

##### `health_checks.HC_NAME.run_check`

This is either `true` or `false` and gives the name of the Job to be used to
run the health check.

##### `health_checks.HC_NAME.runner_name`

The value for `runner_name` will be used as the base of the name of the
Kubernetes Job for each health check instance launched.

##### `health_checks.HC_NAME.image`

This section specifies information regarding the Docker image for health check.

- `health_checks.HC_NAME.image.repo`:
  the base repo URL for the Docker image for the health check.
- `health_checks.HC_NAME.image.tag`:
  the image tag for the Docker image for the health check.
- `health_checks.HC_NAME.image.pull_policy`:
  the pull policy for the Docker image for the health check.

Example:

```yaml
health_checks:
  HC_NAME:
    ...
    image:
      repo: "us-docker.pkg.dev/gce-ai-infra/health-check/health-runner"
      tag: "subset"
      pull_policy: "Always"
    ...
```

##### `health_checks.HC_NAME.blast_mode`

The `blast_mode` section of the configuration gives settings for running health
checks in parallel.

- `health_checks.HC_NAME.blast_mode.blast_mode_enabled`:
  set to `"true"` or "false". If set to `"false"`, a failed health check will
  taint the corresponding node(s).
- `health_checks.HC_NAME.blast_mode.BLAST_MODE_NUM_TESTS_LIMIT`:
  set to an integer specifying how many health checks can be launched
  simultaneously across the cluster.
- `health_checks.HC_NAME.blast_mode.NODES_CHECKED_PER_TEST`:
  set to an integer to specify how many nodes are run for each test. NCCL &
  neper health checks use 2 nodes while the GPU & tinymax health checks only
  uses 1.

##### `health_checks.HC_NAME.env`

The `env` section of the configuration is specific to each health check and is
used to modify the settings for the health check(s) to be kicked off by the
Health Runner. Some settings are specific to the health check type but there
are others that are universal to all health checks.

###### Universal Health Check Settings

- `health_checks.HC_NAME.env.DRY_RUN`:
  this is either set to `"true"` or `"false"`. If set to `"false"`, if a health
  check fails on a node or nodes it will taint the respective node/nodes.
- `health_checks.HC_NAME.env.SLEEP_TIME_MINUTES`:
  this is set to integer value and acts as a timeout for the health check,
  specifying the maximum time allowed for completion. If a health check exceeds
  this time, it is canceled, and the test result is not updated.
- `health_checks.HC_NAME.env.YAML_FILE`:
  this specifies the YAML file used by the Health Runner to launch the health
  check. This YAML file must be present in the Health Runner container (via the
  Docker image).

###### NCCL Health Check Settings

- `health_checks.HC_NAME.env.YAML_FILE`:
  must be set to either `"a3ultra/nccl_healthcheck.yaml"` or
  `"a3plus/nccl_healthcheck.yaml"` or `"a3/nccl_healthcheck.yaml"` or
  `"a4/nccl_healthcheck.yaml"`,
  depending on the nodes' accelerator type.

###### GPU Health Check Settings

- `health_checks.HC_NAME.env.YAML_FILE`:
  must be set to `"gpu_healthcheck.yaml"`.
- `health_checks.HC_NAME.env.R_LEVEL`:
  set to `1`, `2`, `3`, or `4` defining what level of diagnostics to run.
  Lower numbers indicate faster but more basic diagnostics.
  It is recommended to set to `2` or `3` with the `3` being a longer more
  extensive diagnostic check.

###### Neper Health Check Settings

- `health_checks.HC_NAME.env.YAML_FILE`:
  must be set to `"neper_healthcheck.yaml"`.

###### Tinymax Health Check Settings

- `health_checks.HC_NAME.env.YAML_FILE`:
  must be set to `"tinymax_healthcheck.yaml"`.

#### Default Configuration

The default configuration is set so that the Health Runner will run only the
NCCL health check every 5 minutes (10 health checks at a time) for A3+ GPU
nodes.

The default configuration for the Health Runner (found in the Helm chart
[values.yaml](deploy/helm/health_runner/values.yaml) file) is shown below:

```yaml
health_runner:
  base_name: "chs-hr"
health_checks:
  nccl_healthcheck:
    run_check: true
    image:
      repo: "us-docker.pkg.dev/gce-ai-infra/health-check/health-runner"
      tag: "4.2-latest"
      pull_policy: "Always"
    env:
      HC_IMAGE_TAG: "4.2-latest"
      MACHINE_TYPE: "a3-megagpu-8g"
      DRY_RUN: "true"
      SLEEP_TIME_MINUTES: "30"
      FILTER_LABEL_NAME: "aiinfra/nccl-healthcheck-test"
      FILTER_LABEL_VALUE: "true"
      HELM_CHART: "/app/health_checks/nccl_healthcheck"  # Path to Helm chart in container
      HELM_INSTALL_FLAGS: "-f /app/health_checks/nccl_healthcheck/a3plus.yaml --set health_check.image.tag=${MACHINE_TYPE}_${HC_IMAGE_TAG}"  # Specific to A3+
      ACCELERATOR_TYPE: "nvidia-h100-mega-80gb"
      HEALTH_APP: "nccl"
      PAIRING_MODE: "random"
      SECOND_PASS_ENABLED: "true"
    # Blast Mode
    blast_mode:
      blast_mode_enabled: true
      env:
        # BLAST_MODE_NUM_TESTS_LIMIT: "200"  # Number of health checks to run in parallel
        NODES_CHECKED_PER_TEST:  "2"
  gpu_healthcheck:
    run_check: false
    image:
      repo: "us-docker.pkg.dev/gce-ai-infra/health-check/health-runner"
      tag: "4.2-latest"
      pull_policy: "Always"
    env:
      HC_IMAGE_TAG: "4.2-latest"
      MACHINE_TYPE: "a3-megagpu-8g"
      DRY_RUN: "true"
      SLEEP_TIME_MINUTES: "30"
      HELM_CHART: "/app/health_checks/gpu_healthcheck"  # Path to Helm chart in container
      HELM_INSTALL_FLAGS: "--set health_check.image.tag=${MACHINE_TYPE}_${HC_IMAGE_TAG}"
      ACCELERATOR_TYPE: "nvidia-h100-mega-80gb"
      HC_ENV_R_LEVEL: 3
    # Blast Mode
    blast_mode:
      blast_mode_enabled: true  # Defaults to run multiple health checks in parallel
      env:
        # BLAST_MODE_NUM_TESTS_LIMIT: "200"  # Number of health checks to run in parallel
        NODES_CHECKED_PER_TEST:  "1"
  neper_healthcheck:
    run_check: false
    image:
      repo: "us-docker.pkg.dev/gce-ai-infra/health-check/health-runner"
      tag: "4.2-latest"
      pull_policy: "Always"
    env:
      HC_IMAGE_TAG: "4.2-latest"
      MACHINE_TYPE: "a3-megagpu-8g"
      DRY_RUN: "true"
      SLEEP_TIME_MINUTES: "30"
      HELM_CHART: "/app/health_checks/neper_healthcheck"  # Path to Helm chart in container
      HELM_INSTALL_FLAGS: "--set health_check.image.tag=${MACHINE_TYPE}_${HC_IMAGE_TAG}"
      ACCELERATOR_TYPE: "nvidia-h100-mega-80gb"
    blast_mode:
      blast_mode_enabled: true  # Defaults to run multiple health checks in parallel
      env:
        # BLAST_MODE_NUM_TESTS_LIMIT: "200"  # Number of health checks to run in parallel
        NODES_CHECKED_PER_TEST:  "2"
  straggler_healthcheck:
    run_check: false
    image:
      repo: "us-docker.pkg.dev/gce-ai-infra/health-check/health-runner"
      tag: "4.2-latest"
      pull_policy: "Always"
    env:
      HC_IMAGE_TAG: "4.2-latest"
      MACHINE_TYPE: "a3-megagpu-8g"
      DRY_RUN: "true"
      SLEEP_TIME_MINUTES: "30"
      HELM_CHART: "/app/health_checks/straggler_healthcheck"  # Path to Helm chart in container
      HELM_INSTALL_FLAGS: "--set health_check.image.tag=${MACHINE_TYPE}_${HC_IMAGE_TAG}"
      ACCELERATOR_TYPE: "nvidia-h100-mega-80gb"
      HOSTS_CSV: nil  # Allow health runner to identify the nodes
      N_NODES: nil  # Default to run on all nodes in the cluster
      GCS_BUCKET_NAME: "straggler-healthcheck-logs"
    blast_mode:
      blast_mode_enabled: false  # Defaults to run multiple health checks in parallel
  tinymax_healthcheck:
    run_check: false
    image:
      repo: "us-docker.pkg.dev/gce-ai-infra/health-check/health-runner"
      tag: "4.2-latest"
      pull_policy: "Always"
    env:
      HC_IMAGE_TAG: "4.2-latest"
      MACHINE_TYPE: "a3-megagpu-8g"
      DRY_RUN: "true"
      SLEEP_TIME_MINUTES: "10"
      FILTER_LABEL_NAME: "aiinfra/tinymax-healthcheck-test"
      FILTER_LABEL_VALUE: "true"
      HELM_CHART: "/app/health_checks/tinymax_healthcheck"  # Path to Helm chart in container
      HELM_INSTALL_FLAGS: "-f /app/health_checks/tinymax_healthcheck/a3plus.yaml --set health_check.image.tag=${MACHINE_TYPE}_${HC_IMAGE_TAG}"  # Specific to A3+
      ACCELERATOR_TYPE: "nvidia-h100-mega-80gb"
    # Blast Mode
    blast_mode:
      blast_mode_enabled: true
      env:
        # BLAST_MODE_NUM_TESTS_LIMIT: "200"  # Number of health checks to run in parallel
        NODES_CHECKED_PER_TEST:  "1"
  nccl_cluster_healthcheck:
    run_check: false
    image:
      repo: "us-docker.pkg.dev/gce-ai-infra/health-check/health-runner"
      tag: "4.2-latest"
      pull_policy: "Always"
    env:
      HC_IMAGE_TAG: "4.2-latest"
      MACHINE_TYPE: "a3-megagpu-8g"
      DRY_RUN: "true"
      SLEEP_TIME_MINUTES: "30"
      FILTER_LABEL_NAME: "aiinfra/nccl-healthcheck-test"
      FILTER_LABEL_VALUE: "true"
      HELM_CHART: "/app/health_checks/nccl_healthcheck"  # Path to Helm chart in container
      HELM_INSTALL_FLAGS: "-f /app/health_checks/nccl_healthcheck/a3plus.yaml --set health_check.image.tag=${MACHINE_TYPE}_${HC_IMAGE_TAG}"  # Specific to A3+
      ACCELERATOR_TYPE: "nvidia-h100-mega-80gb"
      SECOND_PASS_ENABLED: "true"
      HC_ENV_NHOSTS: "4"
    # Blast Mode
    blast_mode:
      blast_mode_enabled: true
      env:
        NODES_CHECKED_PER_TEST:  "4"
```

### 3.2 Running CHS

To start with, download the repository: with
```
git clone https://github.com/GoogleCloudPlatform/cluster-health-scanner.git
cd cluster-health-scanner/
```

Running CHS involves installing Health Runner.
This is done on a Kubernetes orchestration by deploying the Helm chart for
Health Runner.

The Health Runner Helm chart can be used to install the release using the
`helm` command shown below:

```bash
MY_HEALTH_RUNNER_RELEASE_NAME="my-hr-release"

helm install "${MY_HEALTH_RUNNER_RELEASE_NAME}" \
  deploy/helm/health_runner
```

This will install the Health Runner with the default configuration which will
kick off the health checks automatically to be run on the nodes in the cluster.

You can also specify your own configuration using your own value files:

```bash
MY_HEALTH_RUNNER_RELEASE_NAME="my-hr-release-custom-config"
MY_CONFIG="./my-config.yaml"

helm install "${MY_HEALTH_RUNNER_RELEASE_NAME}" \
  deploy/helm/health_runner \
  -f "${MY_CONFIG}"
```

You can also set specific configurations in the command line using
`helm install` `--set` parameter.
For example, the following command launches only the GPU health check on the
nodes using `R_LEVEL: "1"` instead of the default values.

```bash
MY_HEALTH_RUNNER_RELEASE_NAME="my-hr-release-gpu-only"

helm install "${MY_HEALTH_RUNNER_RELEASE_NAME}" \
  deploy/helm/health_runner \
  --set health_checks.nccl_healthcheck.run_check=false \
  --set health_checks.gpu_healthcheck.run_check=true \
  --set health_checks.gpu_healthcheck.R_LEVEL="1" \
```

### 3.3 Viewing Results

As the Health Runner launches health checks, runs them on nodes, and they
complete, users can view the health check results.

Health check results are stored as node labels and can be viewed using the
Kubernetes `kubectl` tool.

The following command displays results for the NCCL health check for each node:

```bash
CUSTOM_COLS="NODE:.metadata.name,MARK:.metadata.labels.aiinfra/nccl-healthcheck-test,BANDWIDTH:.metadata.labels.aiinfra/nccl-healthcheck-bandwidth,RESULT:.metadata.labels.aiinfra/nccl-healthcheck-result,RUNTIME:.metadata.labels.aiinfra/nccl-healthcheck-runtime-sec"

kubectl get nodes -o custom-columns="${CUSTOM_COLS}"
```

This outputs a table with columns showing the node names and the status of each
of their tags.

If the command `watch` is installed, you can create a dynamic display for live
updates.

```bash
watch -n 10 -d "kubectl get nodes -o custom-columns=${CUSTOM_COLS}"
```

`watch` reruns the table display command every 10 seconds, highlighting any
changes.
#### Viewing results using Cloud Monitoring:

CHS integrates with Cloud Monitoring to provide a clear dashboard for tracking
scan runs and identifying passing/failing nodes.

To create the dashboard in your Cloud project:

1. Navigate to `View Dashboard Templates` within the Cloud Monitoring console.
2. Search and select the `Cluster Health Scanner` template.
3. Click the `Copy Dashboard` button to deploy it to your project.

### 3.4 Cleanup

After deploying and running CHS, users should ensure that the installation is
fully cleaned up. This will prevent any potential issues of lingering
configurations, jobs, or other resources.

#### Uninstalling Health Runner Helm Release

To uninstall the Health Runner (a Helm release), use the release name
(`MY_HEALTH_RUNNER_RELEASE_NAME`) in the following command:

```bash
helm uninstall "${MY_HEALTH_RUNNER_RELEASE_NAME}"
```

#### Removing Leftover and Jobs

While the Health Runner Helm chart simplifies cleanup, it's important to remove
any lingering Jobs in the cluster that are not removed automatically.

You can list these with a command like the following:

```bash
kubectl get jobs | grep "chs-hc-"
```

To remove lingering Jobs:

```bash
kubectl delete jobs $JOB_NAME_0 $JOB_NAME_1
```

Because Jobs from CHS tend to have similar names, you can filter those jobs
by name (such as `healthcheck` in this example) with something like below:

```bash
# Gets list of jobs, filters for `healthcheck`, selects only the Job name
kubectl get jobs \
  | grep "chs-hc-" \
  | cut -d ' ' -f1
```

After confirming the jobs listed are the ones to delete, you can use the
command below to delete those jobs:

```bash
kubectl get jobs --no-headers \
  | grep "chs-hc-" \
  | cut -d ' ' -f1 \
  | xargs kubectl delete jobs
```

