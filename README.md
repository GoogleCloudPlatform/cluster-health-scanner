## 1. Introduction

The **Cluster Health Scanner** tool or simply **CHS** runs a series of tests
called _health checks_ to analyze the health of a cluster of GPU nodes.

For instructions on how to run CHS, go directly to the
['Running CHS' section](#3-running-chs).

While currently structured for Google Kubernetes Engine (GKE), CHS can
theoretically run on clusters using other Kubernetes orchestration
implementations.
There are plans to enable CHS to run on additional cluster orchestrators, such
as [Slurm](https://slurm.schedmd.com/overview.html) for HPC.


## 2. Architecture

The CHS structure runs with the concept of a 'Health Runner' that controls the
scheduling, launching, and post-processing of individual health checks that run
on the nodes in the cluster.


### 2.1 Repo Folder Structure

This repository is structured to separate running/deploying CHS and
building CHS.

#### Deploying CHS

The **[`deploy/`](deploy/)** directory contains the code to deploy CHS on a
cluster.
This is currently done for GKE clusters using [Helm](https://helm.sh/). If
you're looking to quickly start using CHS on your cluster, you only need to
install the Health Runner Helm chart, which contains all the health checks
included in CHS.
See the [section below](#3-running-chs) for details on running CHS.


#### Building CHS

The **[`src/`](src/)** directory contains the code that is run as part of the
Health Runner & health checks.

***This is not necessary to run CHS.*** Instead the code in `src/` is provided
for those wanting to build CHS directly.
See the [section below](#4-building-chs-from-src) for more details.


The **[`docker/`](docker/)** directory contains the Dockerfiles that can be
used with `src/` to build your own Docker images for CHS.
***This is not necessary to run CHS.*** However, certain users might find it 
useful to customize their version of CHS.


### 2.2 CHS Design

CHS is broken up into the following parts:
- Health Runner
  * Manages how the health checks are launched on the nodes and then later 
  cleaned up after completion
- Health Checks
  * Kicked off by the Health Runner and run on the nodes in the cluster to
  report their results to reflect the health of the node(s)/cluster

A user is able to configure the Health Runner to run particular health checks.
This configuration includes settings for how the Health Runner will run as well
as settings for the given health checks that will be run on nodes.

#### Health Runner

The Health Runner coordinates how and what health checks get launched on what
nodes.

The user configures the Health Runner to specify:

- Which health checks to run
- Settings/configuration of those health checks
- Other features of the Health Runner launching health checks such as
  how many health checks should run in parallel,
  what Docker image to use for the health check,
  and how often to launch health checks on the appropriate nodes.


#### Health Checks

Health checks can run on single or multiple nodes.
As health checks complete, results are reported
by node labels.

These health checks are all launched by the Health Runner & are set by the
Health Runner's configuration.

##### NCCL Health Check

Runs on two nodes to check networking across these nodes.
A 'pass' is given when both nodes have an average bandwidth that meets or
exceeds a given threshold.

##### GPU Health Check

Runs the [NVIDIA's DCGM diagnostic tool](https://developer.nvidia.com/dcgm) to
report a single node's health.
A 'pass' is given when no errors appear while running the tool.

##### Neper Health Check

Runs the [neper Linux networking performance tool](https://github.com/google/neper)
to report the health of a node.
A 'pass' is given when the bandwidth across connections in the node meet or
exceed a given threshold.


## 3. Running CHS

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
- GPU Health Check: `aiinfra/nccl-healthcheck-test`
- Neper Health Check: `aiinfra/neper-healthcheck-test`

These label keys & values can be set using the `kubectl` tool 
using the following command:

```bash
kubectl label nodes \
    --all \
    aiinfra/nccl-healthcheck-test="true"
```

> Note:
> This sets all nodes to be labeled for the NCCL health check.


#### Configuration of the Health Runner & Health Checks

The user can configure the Health Runner via the command line or as part of a
YAML configuration file. This configuration also gives the settings for the
health checks to be run.

Go to the [_'Default Configuration'_ section](#default-configuration) for an
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

##### `health_checks.HC_NAME.schedule`

`schedule` is given as a [unix-cron string format](https://cloud.google.com/scheduler/docs/configuring/cron-job-schedules) and will specify how often a batch of these health checks
should run.

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
  set to an integer specifying how many health checks can be launched simultaneously across the cluster.
- `health_checks.HC_NAME.blast_mode.NODES_CHECKED_PER_TEST`: 
  set to an integer to specify how many nodes are run for each test. NCCL & 
  neper health checks use 2 nodes while the GPU health check only uses 1.


##### `health_checks.HC_NAME.env`

The `env` section of the configuration is specific to each health check and is
used to modify the settings for the health check(s) to be kicked off by the
health runner. Some settings are specific to the health check type but there
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
  must be set to either `"a3plus/nccl_healthcheck.yaml"` or
  `"a3/nccl_healthcheck.yaml"`, depending on the nodes' accelerator type.

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



#### Default Configuration

The default configuration is set so that the Health Runner will run only the
NCCL health check every 5 minutes (10 health checks at a time) for A3+ GPU
nodes.

The default configuration for the Health Runner (found in the Helm chart 
[values.yaml](deploy/helm/health_runner/values.yaml) file) is shown below:

```yaml
health_runner:
  name: "health-runner"
health_checks:
  nccl_healthcheck:
    run_check: true
    runner_name: nccl-health-runner-a3plus
    schedule: "*/5 * * * *"  # run every five minutes
    image:
      repo: "us-docker.pkg.dev/gce-ai-infra/health-check/health-runner"
      tag: "subset"
      pull_policy: "Always"
    env:
      DRY_RUN: "true"
      SLEEP_TIME_MINUTES: "5"
      YAML_FILE: "a3plus/nccl_healthcheck.yaml"
      ACCELERATOR_TYPE: "nvidia-h100-mega-80gb"
      IMAGE_TAG: "subset"
    blast_mode:
      blast_mode_enabled: false
      env:
        BLAST_MODE_NUM_TESTS_LIMIT: "10"
        NODES_CHECKED_PER_TEST:  "2"
  gpu_healthcheck:
    run_check: false
    runner_name: gpu-health-runner
    schedule: "0 * * * *"  # run every hour
    image:
      repo: "us-docker.pkg.dev/gce-ai-infra/health-check/health-runner"
      tag: "subset"
      pull_policy: "Always"
    env:
      DRY_RUN: "true"
      SLEEP_TIME_MINUTES: "15"
      YAML_FILE: "gpu_healthcheck.yaml"
      R_LEVEL: "2"
      ACCELERATOR_TYPE: "nvidia-h100-mega-80gb"
      IMAGE_TAG: "subset"
  neper_healthcheck:
    run_check: false
    runner_name: neper-health-runner
    schedule: "*/5 * * * *"  # run every 5 min
    image:
      repo: "us-docker.pkg.dev/gce-ai-infra/health-check/health-runner"
      tag: "v2.1.9-public"
      pull_policy: "Always"
    env:
      DRY_RUN: "true"
      SLEEP_TIME_MINUTES: "10"
      YAML_FILE: "neper_healthcheck.yaml"
      ACCELERATOR_TYPE: "nvidia-h100-mega-80gb"
      IMAGE_TAG: "v2.1.9-public"
```


### 3.2 Running CHS

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
CUSTOM_COLS="NODE:.metadata.name,MARK:.metadata.labels.aiinfra/nccl-healthcheck-test,BANDWIDTH:.metadata.labels.aiinfra/nccl-healthcheck-bandwidth,RESULT:.metadata.labels.aiinfra/nccl-healthcheck-result,VALID_TILL:.metadata.labels.aiinfra/nccl-healthcheck-valid-till-sec"

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


### 3.4 Cleanup

After deploying and running CHS, users may wish to clean up the installation.

#### Uninstalling Health Runner Helm Release 

To uninstall the Health Runner (a Helm release), use the release name
(`RELEASE_NAME`) in the following command:

```bash
helm uninstall RELEASE_NAME
```

#### Removing Leftover CronJobs and Jobs
While the Health Runner Helm chart simplifies cleanup, it's important to remove
any lingering CronJobs and Jobs in the cluster that are not removed
automatically.

You can list these with commands such as the following:

```bash
kubectl get cronjobs | grep healthcheck
kubectl get jobs | grep healthcheck
```

To remove lingering CronJobs & Jobs:

```bash
kubectl delete cronjobs $CRONJOB_NAME

kubectl delete jobs $JOB_NAME_0 $JOB_NAME_1
```

Because Jobs from CHS tend to have similar names, you can filter those jobs
by name (such as `healthcheck` in this example) with something like below:

```bash
# Gets list of jobs, filters for `healthcheck`, selects only the Job name
kubectl get jobs \
  | grep healthcheck \
  | cut -d ' ' -f1
```

After confirming the jobs listed are the ones to delete, you can use the
command below to delete those jobs:

```bash
kubectl get jobs --no-headers \
  | grep healthcheck \
  | cut -d ' ' -f1 \
  | xargs kubectl delete jobs
```