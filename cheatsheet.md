# Cheat Sheet

Below are a set of potentially useful commands for different steps in using
Bad Node Detector

## Preparation

### Installing Helm

Bad Node Detector uses [Helm](https://helm.sh) as the main method of
deployment. Installing the Helm CLI will be the most straight-forward way of
taking advantage of what BND has to offer.

You can see Helm's [installation documentation](https://helm.sh/docs/intro/install/)
for details, but most users will find using the commands below to get adn use
the installer script to be sufficient:

```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
```

### Check for past deployments

If BND was deployed previously, you might find you need to do some clean up.
Below are some commands that can help in identifying resources/installations to
clean up.

To list current Helm Release installations:
```bash
helm ls
```

To view current CronJobs in Kubernetes cluster:
```bash
kubectl get cronjobs
```

To view current Jobs in Kubernetes cluster:
```bash
kubectl get jobs
```

> NOTE: Since the list of Jobs & CronJobs can be quite large, you may find it useful to filter
> the output from the above commands by piping the results with `grep`.
>
> For example, the following will get the list of Jobs that have the phrase 'healthcheck' in
> them: `kubectl get jobs | grep healthcheck`


### Remove old labels

It might be desirable or necessary to remove some node labels that were part of
previous BND runs/deployments.

#### Identifying labels

First, it's helpful to identify the labels on nodes.

You can inspect a particular node `$NODE_NAME` in a few ways.


This command will give details of the node including the labels near the
beginning of the output:
```bash
kubectl describe nodes $NODE_NAME
```

Below will printout information about the node in a table and include a list of
the nodes
```bash
kubectl get nodes $NODE_NAME --show-labels=true
```

You can also run either of these commands on multiple nodes.

This can be done by a subset by listing them:
```bash
kubectl get nodes $NODE_NAME_0 $NODE_NAME_1 $NODE_NAME_2 --show-labels=true
kubectl describe nodes $NODE_NAME_0 $NODE_NAME_1 $NODE_NAME_2
```

You can also run the same command for all the nodes by not listing any specific
nodes. This however can be overwhelming & take time to collect information 
depending on the nubmer of nodes in your cluster:
```bash
kubectl get nodes --show-labels=true
kubectl describe nodes
```

#### Removing labels

To remove labels from nodes, you can use the following on specified nodes where
`LABEL_NAME` is the name of the label (note the `-` suffix):

```bash
kubectl label nodes $NODE_NAME_0 $NODE_NAME_1 LABEL_NAME-
```

This can also be done with multiple labels

```bash
kubectl label nodes $NODE_NAME_0 $NODE_NAME_1 LABEL_NAME- LABEL_NAME_2-
```

You may also want to remove labels from all nodes in the cluster:
```bash
kubectl label nodes --all LABEL_NAME-
```

You can filter to only apply to certain nodes based on another label
`FILTER_LABEL` of the nodes.

```bash
# Any nodes with FILTER_LABEL (any value)
kubectl label nodes -l FILTER_LABEL

# Any nodes with the value 'VALUE' for the label FILTER_LABEL
kubectl label nodes -l FILTER_LABEL=VALUE
```

### Labeling nodes (for the NCCL health check)

Some health checks may require you to label the nodes before running the health
check.

Below labels a specific set of nodes with the label `LABEL_NAME` and value
`VALUE`:
```bash
kubectl label nodes $NODE_NAME_0 $NODE_NAME_1 LABEL_NAME=VALUE
```

You can also apply to all nodes or apply to nodes with a specific different
label `FILTER_LABEL` & value `FILTER_VALUE`:
```bash
# Applies to all nodes in cluster
kubectl label nodes --all LABEL_NAME=VALUE

# Any nodes with FILTER_LABEL (any value)
kubectl label nodes -l FILTER_LABEL LABEL_NAME=VALUE

# Any nodes with the value 'FILTER_VALUE' for the label FILTER_LABEL
kubectl label nodes -l FILTER_LABEL=FILTER_VALUE LABEL_NAME=VALUE
```


## Deployment

BND uses Helm as the preferred way of deploying. Below will briefly show some
options for deploying by utilizing Helm. However, users may find the specific
documentation within the Chart to be helpful.

The below command will install the Helm Chart `BND_CHART` with the Release name
`MY_RELEASE` using the default configurations for the Chart:
```bash
helm install MY_RELEASE BND_CHART
```

We can overwrite the defaults in the command with the `--set` flag like below:
```bash
helm install MY_RELEASE BND_CHART \
  --set health_check.param_0=true \
  --set health_check.env.setting_0="value"
```

You can also specify a file with a set configuration `my_values.yaml` as part
of your installation:
```bash
helm install MY_RELEASE BND_CHART \
  -f my_values.yaml
```

> **Note on deploying without Helm**
>
> Some users may find it useful for their workflow to not use Helm to deploy
> but still work with an equivalent YAML file.
>
> For details on how to generate a YAML file for Helm, see the section 
> ["Generating YAML from Helm"](#generating-yaml-from-helm)


## Observations

After deploying the health runner for BND, you might find it useful to make
observations of the cluster's nodes that are relevant.

### Use the `watch` command to get periodic updated

The `watch` command can be paired with other commands to show updates to
relevant observations.

For example, the following command will display jobs that match 'healthcheck'
every 10 seconds while highlighting the differences from each update:

```bash
watch -n5 -d "kubectl get jobs | grep healthcheck"
```

### Displaying information about nodes

Since BND uses nodel labeling to update information about health checks for
that node, it can be useful to print information about those nodes.

We already saw briefly how to get some information on nodes in the
['Identifying labels' section](#identifying-labels):

```bash
kubectl describe nodes
kubectl get nodes
```

#### Using labels to filter nodes

We can go further by filtering for nodes with a particular label `LABEL` and
specific values on those labels `VALUE`:

```bash
# Get nodes that have the 'LABEL' label (any value)
kubectl get nodes -l LABEL

# Get nodes that have the value 'VALUE' for the label 'LABEL'
kubectl get nodes -l LABEL=VALUE
```

#### Custom columns for node information

We can also customize what is printed out in `kubectl get nodes` by specifying
the data to display using the flag `-o custom-columns=$CUSTOM_COLS` where
`CUSTOM_COLS` is a string that specifies how and what data to display.

For example, we can define `CUSTOM_COLS` where `TEST_LABEL` is a label for if
the node should be tested and `RESULT_LABEL` is the label with the value for
the health check's result:
```bash
CUSTOM_COLS="NODE:.metadata.name,TEST:.metadata.labels.TEST_LABEL,RESULT:.metadata.labels.RESULT_LABEL

kubectl get nodes -o custom-columns=$CUSTOM_COLS
```

The printout something like this:
```
NODE                                 TEST     RESULT
gke-cluter-name-np-0-dcs1a6c6-24rm   true     pass
gke-cluter-name-np-0-dcs1a6c6-7qd5   true     <none>
gke-cluter-name-np-0-dcs1a6c6-81aw   <none>   <none>
gke-cluter-name-np-0-dcs1a6c6-8lc9   false    fail
gke-cluter-name-np-0-dcs1a6c6-j3q0   true     fail
gke-cluter-name-np-0-dcs1a6c6-tzl6   false    pass
gke-cluter-name-np-0-dcs1a6c6-wkd3   <none>   fail
gke-cluter-name-np-0-dcs1a6c6-z0mn   <none>   true
```

Note that the `<none>` values mean that the node does not have that label.

This also does not filter nodes and will print the specified custom columns for
all nodes. To further specify nodes, you can pass other flags such as `-l` to
`kubectl get nodes` like below:

```bash
kubectl get nodes -l RESULT_LABEL='fail' -o custom-columns=$CUSTOM_COLS
```

Which will produce something like this (using the previous example result):
```
NODE                                 TEST     RESULT
gke-cluter-name-np-0-dcs1a6c6-8lc9   false    fail
gke-cluter-name-np-0-dcs1a6c6-j3q0   true     fail
gke-cluter-name-np-0-dcs1a6c6-wkd3   <none>   fail
```

## Clean Up

After deploying and running BND, users might desire to clean up their
installation.


### Uninstalling with Helm

To uninstall a Helm Release installed, simply use the name of the Release
`RELEASE_NAME` in the following command:

```bash
helm uninstall RELEASE_NAME
```

### Uninstalling with `kubectl`

Because some users maybe have opted to use `kubectl` over directly using the
Helm Chart to deploy, it's important to uninstall using the same YAML file used
in `kubectl apply`:

```bash
kubectl delete -f my_deployment.yaml
```

### Cleaning up lingering CronJobs and Jobs

Using a Helm Chart makes clean up simpler, but it's important to remove any
lingering CronJobs and Jobs in your cluster that doesn't get cleaned up
automatically.

You can list these with the commands like the following using `grep`:

```bash
kubectl get cronjobs | grep healthcheck
kubectl get jobs | grep healthcheck
```

To remove lingering CronJobs & Jobs:
```bash
kubectl delete cronjobs CRONJOB_NAME

kubectl delete jobs JOB_NAME_0 JOB_NAME_1
```

Because Jobs from BND tend to have similar names, you can filter those jobs
by name (such `healthcheck` in this example) with something like below:

```bash
# Gets list of jobs, filters for `healthcheck`, selects only the Job name
kubectl get jobs \
  | grep healthcheck \
  | cut -d ' ' -f1
```

After confirming the jobs listed are the ones to delete, you can use the above
command to delete those jobs:

```bash
kubectl delete jobs $(kubectl get jobs | grep healthcheck | cut -d ' ' -f1 )
```



## Generating YAML from Helm

Although deployment with Helm is the recommended method for BND, users may find
it useful for their workflow to produce an equivalent YAML file.

To generate the file, you can run the same `helm` command with configurations
as you would with installation but instead replacing `helm install` with
`helm template` and then redirecting the standard output to a file
`my_deployment.yaml`:
```bash
helm template MY_RELEASE BND_CHART \
  -f my_values.yaml \
  --set health_check.param_0=true \
  --set health_check.env.setting_0="value" \
  > my_deployment.yaml
```

A user can then install with `kubectl` using the generated YAML file with
something like below:

```bash
kubectl apply -f my_deployment.yaml
```