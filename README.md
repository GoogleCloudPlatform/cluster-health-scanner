## 1. Introduction
> Description what this is, who should run this 

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


### Repo Folder Structure

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
useful to customize their version of CHS


### CHS Design

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

##### NCCL

Runs on two nodes to check networking across these nodes.
A 'pass' is given when both nodes have an average bandwidth that meets or
exceeds a given threshold.

##### GPU

Runs the [NVIDIA's DCGM diagnostic tool](https://developer.nvidia.com/dcgm) to
report a single node's health.
A 'pass' is given when no errors appear while running the tool.

##### Neper

Runs the [neper Linux networking performance tool](https://github.com/google/neper)
to report the health of a node.
A 'pass' is given when the bandwidth across connections in the node meet or
exceed a given threshold.