# Health Check Scheduler Application

This application runs scheduled health checks on a cluster and operates within
a Kubernetes (k8s) environment.
These health checks will be deployed via the `health_runner` container.

In the [Quickstart Instructions](#quickstart-instructions-nccl-health-check),
there is a detailed guide in running on the NCCL health check.
The [Appendix](#appendix) gives more details about other health checks
that can be run.

> Note: For more details in using the Helm Chart for `health_runner`, please
> see the chart's [documentation](deploy/helm/health_runner/README.md)

## Quickstart Instructions: NCCL Health Check

The following instructions assume that you have cloned this repo and your
current working directory is the directory of this README.

#### 1. (Optional) Check Your Cluster Context

Before you begin, check that you are running your commands in the correct
cluster. Check that your context is correct with:

```
kubectl config current-context
```

You can create a context in a new cluster with:

```
gcloud container clusters get-credentials <cluster-name> --region=<region> --project=<project-id>
```

#### 2. Mark Nodes for Testing

If you want to check a subset of the nodes in a cluster, you need to add a label
indicating that they should be checked.

You can add a label to a set of nodes with the following command:



```
kubectl label nodes $NODE1 $NODE2 aiinfra/nccl-healthcheck-test=true
```

To run on all nodes, simply label all nodes as follows:


```
kubectl label nodes --all aiinfra/nccl-healthcheck-test=true
```

On success, this command should output a line for each node saying that it was
labeled.

To view which nodes have labels, run the following

```
kubectl get nodes -o custom-columns="NODE:.metadata.name,MARK:.metadata.labels.aiinfra/nccl-healthcheck-test" 
```

This will show a table of all nodes in the cluster with their name and the
status of their health marker in separate columns.

#### 3. Run the tests

To run the tests, simply use the Helm Chart to install the release with 

```
# Go to the health_runner deployment folder
cd deploy/helm/
# Use your own release name instead of `my-bnd-release`
helm install my-bnd-release health_runner \
  --set health_checks.nccl_healthcheck.run_check=true
```

This will create a CronJob that will check 2 nodes every 5 minutes. The results
of the check will be added as labels on the selected nodes. You can watch their
progress as follows.

> NOTE: This means that it will take a while for all of the nodes to be checked.


```
kubectl get nodes -o custom-columns="NODE:.metadata.name,MARK:.metadata.labels.aiinfra/nccl-healthcheck-test,BANDWIDTH:.metadata.labels.aiinfra/nccl-healthcheck-bandwidth,RESULT:.metadata.labels.aiinfra/nccl-healthcheck-result,VALID_TILL:.metadata.labels.aiinfra/nccl-healthcheck-valid-till-sec" 
```

If the command `watch` is installed, you can create a quick screen for live
updates.

```
watch -n10 -d "kubectl get nodes -o custom-columns=\"NODE:.metadata.name,MARK:.metadata.labels.aiinfra/nccl-healthcheck-test,BANDWIDTH:.metadata.labels.aiinfra/nccl-healthcheck-bandwidth,RESULT:.metadata.labels.aiinfra/nccl-healthcheck-result,VALID_TILL:.metadata.labels.aiinfra/nccl-healthcheck-valid-till-sec\""
```

This will output a table with columns showing the node names and the status of
each of their tags. Watch will rerun the table display command every 10 seconds,
highlighting any changes that occur each time.

Note that it may take a while for the nodes to get annotated, as only 2 nodes
are annotated every 5 minutes and it may take up to 5 minutes for the job to run
for the first time.

#### 4. (Optional) Remove Test Results

You can remove the resulting labels from a set of nodes using the following
command.

```
kubectl label nodes $NODE1 $NODE2 aiinfra/nccl-healthcheck-bandwidth- aiinfra/nccl-healthcheck-result- aiinfra/nccl-healthcheck-valid-till-sec-
```

This may be particularly useful if you want to re-run tests on a specific node
within 24 hours, because the aiinfra/nccl-healthcheck-valid-till-sec label marks
the node as not needing to be rerun for 24 hours. Removing the tag opens the
node up for being scheduled for testing.

> NOTE: This command may complain that it isn't finding labels (e.g. `node/$NAME`
> not labeled, label "aiinfra/nccl-healthcheck-bandwidth" not found).This often
> happens even if it is successfully removing labels. You can verify that the
> state looks like you want by checking the output of the `kubectl get nodes`
> command above.

#### 5. (Optional) Unmark Nodes for Testing

You can unmark nodes for testing with the following command:

```
kubectl label nodes $NODE1 aiinfra/nccl-healthcheck-test- 
```

#### 6. (Optional) Deschedule Testing

If a new version of the health check code becomes available (or if you simply
want to limit the number of jobs in queue), it may be desirable to delete the
existing CronJob to put up a new test. This can be done as follows

```
# Using your release name
helm uninstall my-bnd-release

kubectl get cronjobs -o name | grep "cronjob.batch/nccl-health-runner-" | xargs -r kubectl delete
kubectl get jobs -o name | grep "job.batch/nccl-healthcheck-" | xargs -r kubectl delete
```

#### 7. (Debugging) Full Cluster Queue

It’s possible that the health runner may fail to run because the cluster is
busy. If you suspect this is the case (for instance, if a lot/all of the nodes
remain unannotated), below are some steps to get you jump-started on fixing the
issue.

Before you begin, you may want to deschedule all remaining checks by following
the instructions in step 7.

To do this, we'll list out all of the jobs that don't have any NCCL
test-generated labels and have uncompleted jobs.

```
for node in $(kubectl get nodes -l '!aiinfra/nccl-healthcheck-result' | cut -d ' ' -f 1 | tail -n +2)
do
  echo "====$node===="
  kubectl get pods --field-selector spec.nodeName=$node | grep -v Complete
done
```

From here, you can take a look at what uncompleted pods are running to see if
they have either errored or can otherwise be killed. To free up the node, you
can delete the corresponding pod (or, possibly job), with a command such as

```
kubectl delete pod $POD1 $POD2
```

Once all of the offending pods have been cleaned up, you can re-mark them for
annotation and the re-run them. In the code block below, we simply re-mark all
files without annotations (and dump them in to a file called
untested_nodes.txt), but you can instead load the list however you want. For
instance, ignoring the first command and instead writing their names in the file
read by the for-loop.

```
# Get untested nodes listed into file
kubectl get nodes -l '!aiinfra/nccl-healthcheck-result' | cut -d ' ' -f 1 > untested_nodes.txt


# Iterate over untested nodes & marked to label
for node in $(cat untested_nodes.txt)
do
  echo $node
  kubectl label node $node aiinfra/nccl-healthcheck-test=true
done

# Deployed BND NCCL subset
helm install my-bnd-release health_runner \
  --set health_checks.nccl_healthcheck.run_check=true
```



## Appendix

This application runs scheduled health checks on a cluster and operates within
a Kubernetes (k8s) environment. It utilizes four Docker images:
- `health-runner`
- `nccl-healthcheck`
- `gpu-healthcheck`
- `neper-healthcheck`


### Tool Overview

#### 1. **health-runner image**
This includes a python application that consumes some Kubernetes
Job templates and schedules the actual health check workloads on
the same kubernetes cluster that the runner lives in.

#### 2. **nccl-healthcheck image**
This app runs pair-wise
[NCCL AllReduce](https://github.com/NVIDIA/nccl-tests/tree/master#arguments) tests
 against the nodes it get scheduled on. Note that NCCL tests are bidirectional and if a NCCL test failed it will taint both of the nodes it get scheduled on.

The tests are run two passes. If the first pass fails, then a second pass is run
that pairs the failed nodes with known good nodes from other runs. This helps to
isolate the actual node that is having connection issues. Failure of the first
pass will taint the node with `aiinfra/nccl-healthcheck=failed:PreferNoSchedule`
until the second pass is run.

If the second pass encounters failures or
errors, it will taint the node with
 `aiinfra/nccl-healthcheck=failed:NoSchedule` preventing
 the scheduling of future workloads on that node, with
 failure reason listed as labels `aiinfra/nccl-healthcheck-bandwidth=${BAD_VM_BW_IN_GB}`.


#### 3. **gpu-healthcheck image**
This app runs level 3 DCGM (NVIDIA Data Center GPU Manager)
checks (`dcgm with r=3`) on the current host GPUs. checks for
unmapped ECC errors on GPUs. If it encounters failures or
errors, it will taint the node with
 `aiinfra/gpu-healthcheck=failed:NoSchedule`, preventing
 the scheduling of future workloads on that node.

#### 4. **neper-healthcheck image**

This app runs pair-wise
[Neper TCP Stream](https://github.com/google/neper#tcp_stream) tests
 against the nodes it get scheduled on. Note that we
 observed tuning the numer of flows and enabling the zero-copy
 feature on this image can "flip" a node from
 failing the tests to passing. We are still investigating
 if this uncovers any real issues.

On the server side, it runs

```
taskset -c 17-32 /scripts/tcp_stream -rw --skip-rx-copy --num-threads=16 \
--num-flows=200 --suicide-length=600 --test-length=30
```

On the client side, it runs

```
taskset -c 17-32 /scripts/tcp_stream -rw --client -H '{dst_ip}' \
--skip-rx-copy --num-threads=16 --num-flows=200 \
--suicide-length=600 --test-length=30
```

If it encounters failures or
errors, it will taint the node with
 `aiinfra/neper-healthcheck=failed:NoSchedule` preventing
 the scheduling of future workloads on that node, with
 failure reason listed as labels `aiinfra/neper-healthcheck_eth{1..4}=${BAD_THROUGHPUT}`.

### Scheduling Behaviors
To minimize the occupation of the cluster by health checks,
the application assigns a label, `aiinfra/${IMAGE-NAME}-valid-till-sec`,
to every node where the health checker runs. This label contains
 the current epoch timestamp plus 24 hours, thus enabling the
 health check to run only once a day for each visible node.
 If a node lacks this label, the health checker will be scheduled
 at the next round hour.

### Example: Deploying and Scheduling an Immediate Health Check

To deploy the health checker to your cluster, run the following command using
the Helm Chart:

```sh
cd deploy/helm/
helm install my-bnd-release health_runner  
  --set health_checks.nccl_healthcheck.run_check=true
```

> Note you can turn on & off other health checks with more usage of the `--set`
> flag. For example, this would turn on the GPU & Neper health checks but turn
> off the NCCL health check:
> ```
> helm install my-bnd-release health_runner
>  --set health_checks.gpu_healthcheck.run_check=true
>  --set health_checks.neper_healthcheck.run_check=true
>  --set health_checks.nccl_healthcheck.run_check=false
> ``

This Helm Chart provides test specific configurable knobs e.g.,

```
# Neper test
SLEEP_TIME_MINUTES (default is 5 mins): Allow the test to run for up to 5 min and kill it on timeouts
DRY_RUN (default is false): Allow the test to do labeling only without tainting the nodes if it fails
# NCCL test
SLEEP_TIME_MINUTES (default is 5 mins): Allow the test to run for up to 5 min and kill it on timeouts
DRY_RUN (default is false): Allow the test to do labeling only without tainting the nodes if it fails
# GPU healthcheck
SLEEP_TIME_MINUTES (default is 15 mins): Allow the test to run for up to 15 min and kill it on timeouts
DRY_RUN (default is false): Allow the test to do labeling only without tainting the nodes if it fails
```

To schedule an immediate check (optional) on a set of nodes run following:

```sh
kubectl create job --from=cronjob/gpu-health-runner gpu-health-runner
# OR
kubectl create job --from=cronjob/nccl-health-runner nccl-health-runner
# OR
kubectl create job --from=cronjob/neper-health-runner neper-health-runner
```

After health-runner is scheduled it will up to 15 min to do the
check. After that daemon set will be killed by health checker job.

To query results you run:

```sh
kubectl get nodes -o json | jq '.items[] | select((.spec.taints // []) | any(.key == "aiinfra/gpu-healthcheck")) | .metadata.name'
# OR
kubectl get nodes -o json | jq '.items[] | select((.spec.taints // []) | any(.key == "aiinfra/neper-healthcheck")) | .metadata.name'
# OR
kubectl get nodes -o json | jq '.items[] | select((.spec.taints // []) | any(.key == "aiinfra/nccl-healthcheck")) | .metadata.name'
```

Examples of nodes that failed the health check:
```
$ kubectl describe node/gke-my-node
Name:               gke-my-node
Roles:              <none>
Labels:             aiinfra/gpu-healthcheck-valid-till-sec=1699664935
                    aiinfra/nccl-healthcheck=failed
                    aiinfra/nccl-healthcheck-bandwidth=37
                    aiinfra/nccl-healthcheck-valid-till-sec=1699615262
                    aiinfra/neper-healthcheck=failed
                    aiinfra/neper-healthcheck-valid-till-sec=1699600358
                    aiinfra/neper-healthcheck_eth1=115368761582

```