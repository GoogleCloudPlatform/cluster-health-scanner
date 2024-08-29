# Overview

The 'Bad Node Detector' is a series of health checks or tests run in separate
CronJobs that analyze the health of a GKE cluster attached to A3/A3+ nodepools.

## Architecture

There is a parent container (`health_runner`), whose only purpose is to
schedule the individual health checks of each sub-type. This parent container
runs as a separate CronJob for each test, taking a path to the YAML file to
deploy for each test, and optionally an image tag for that test.

## Results

Each node is labeled with the output of the test if applicable (today just
NCCL). A separate label is added to each tested node stating the expiration of
the test that ran (the default expiration is 24hrs).

Each test emits a log line that can be consumed in Cloud Logging.

## Running Health Checks

For detailed information about running health checks see the
[README-usage.md](README-usage.md).

You also might want to check the [cheatsheet](cheatsheet.md) that lists a few
useful commands on various parts of using BND.

### Note on Implementation 

Bad Node Detector is implemented through a provided Helm Chart
([health_runner](deploy/helm/health_runner/)) which will kick off the specified health
checks with given configurations. See the chart's [README.md](deploy/helm/health_runner/README.md)

Although the preferred method of implementing is using the `health_runner` Helm
Chart, some users may find it useful working with an equivalent YAML file.
Details regarding generating a YAML file from the Helm Chart can be foud in 
[this section](deploy/helm/health_runner/README.md#generating-yaml-files-from-helm) along
with the equivalent YAML file of the default configuration.