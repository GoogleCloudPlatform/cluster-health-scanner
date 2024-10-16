## 1. Introduction

The **Cluster Health Scanner** tool or simply **CHS** runs a series of tests
called _health checks_ to analyze the health of a cluster of GPU nodes.

Currently CHS is structured to run on [Google Kubernetes Engine](https://cloud.google.com/kubernetes-engine) (GKE) and can in theory run on clusters also run on other
Kubernetes orchestration implementations. There are plans to have CHS be able 
to run on additional cluster orchestrations such as [Slurm](https://slurm.schedmd.com/overview.html)
for HPC.


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
you're looking to quickly start using CHS on your cluster, you will only need
to install the Health Runner Helm chart which will already contain all the
health checks that are part of CHS.
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

The current practice is to mark nodes in the cluster to be tested in a given
health check using a node label related to that health check.

The node label keys are dependent on the health check
(values expected are `"true"`):

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

> Note this sets all nodes to be labeled for the NCCL health check


#### Configuration of the Health Runner & Health Checks

The user can configure the Health Runner via the command line or as part of a
YAML configuration file. This configuration also gives the settings for the
health checks to be run.


##### Default Configuration



### 3.2 Running CHS

Running CHS involves installing Health Runner.
This is done on a Kubernetes orchestration by deploying the Helm chart for
Health Runner.

We can use the Health Runner Helm chart to install the release with the `helm`
command shown below:

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

You can also set set specific configurations in the command line using 
`helm install` `--set` parameter.
For example, the following command will only launch the GPU health check on the
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

As the Health Runner launches health checks, are run on nodes, and complete,
users can view the health check results.

These health check results are available as node labels and can be viewed using
Kubernetes `kubectl` tool.

The following command displays results for the NCCL health check for each node:

```bash
CUSTOM_COLS="NODE:.metadata.name,MARK:.metadata.labels.aiinfra/nccl-healthcheck-test,BANDWIDTH:.metadata.labels.aiinfra/nccl-healthcheck-bandwidth,RESULT:.metadata.labels.aiinfra/nccl-healthcheck-result,VALID_TILL:.metadata.labels.aiinfra/nccl-healthcheck-valid-till-sec"

kubectl get nodes -o custom-columns="${CUSTOM_COLS}"
```

This will output a table with columns showing the node names and the status of
each of their tags.

If the command `watch` is installed, you can create a quick screen for live
updates.

```bash
watch -n 10 -d "kubectl get nodes -o custom-columns=${CUSTOM_COLS}"
```

Watch will rerun the table display command every 10 seconds, highlighting any
changes that occur each time.


### 3.4 Cleanup


## 4. Building CHS from src 


## 5. Miscellaneous configuration options 


## 6. Useful commands / cheat sheet
