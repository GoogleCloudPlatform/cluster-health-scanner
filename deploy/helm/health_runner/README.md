# Health Check Scheduler - Helm Chart

This application runs scheduled health checks on a cluster and operates within
a Kubernetes (k8s) environment.
These health checks will be deployed via the `health_runner` container.

This is deployed via a Helm Chart to allow ease of defining different
configurations for one or more health checks.

For example, the NCCL health check can be implemented with the following

```
helm install my-nccl-health-check ./health_runner \
  --set health_checks.nccl_healthcheck.run_check=true
```

We can also add change settings for the health check. For example, the command
below will run the NCCL health check, disable `DRY_RUN`, and enable blast mode:

```
helm install my-nccl-health-check ./health_runner \
  --set health_checks.nccl_healthcheck.run_check=true \
  --set health_checks.nccl_healthcheck.DRY_RUN="false" \
  --set health_checks.nccl_healthcheck.blast_mode.blast_mode_enabled=true
```

## Preparation Before Installation

Bad Node Detector uses some labels on nodes to determine how some health checks
should run. For this preparation step, please see the [usage guide](../README-usage.md)
for details.

## Installing Helm Chart

### Default - NCCL Health Check

The default configuration provided will run only the NCCL health check (blast
mode disabled).

It can be installed with the following where `my-default-release` is the name of
your release:

```sh
helm install my-default-release ./health_runner
```

Or by explicitly using a specified values file created by a user (in this
example `my_values.yaml`):

```sh
helm install my-default-release ./health_runner \
  --values my_values.yaml
```

> For more information on value files, see the Helm's official documentation:
> https://helm.sh/docs/helm/helm_install/#helm-install

### Specifying Health Check Configuration


#### Using the `--set` flag

You can alter the configuration of your Chart install using the `set` flag.

For example:

```sh
helm install my-custom-nccl-check ./health_runner \
  --set health_checks.nccl_healthcheck.run_check=true \
  --set health_checks.nccl_healthcheck.DRY_RUN="true" \
  --set health_checks.nccl_healthcheck.SLEEP_TIME_MINUTES="5" \
  --set health_checks.nccl_healthcheck.YAML_FILE="a3plus/nccl_healthcheck.yaml" \
  --set health_checks.nccl_healthcheck.ACCELERATOR_TYPE="nvidia-h100-mega-80gb" \
  --set health_checks.nccl_healthcheck.IMAGE_TAG="subset"
```


This command overwrites the specified configuration by `values.yaml`,
specifically enabling the NCCL health check with its various settings.

> Note this happens to still recreate the default behavior when installing the
> Chart as discussed in the [instructions above](#default-nccl-health-check).

#### Using a custom values YAML file

You can also provide a custom values YAML file and pass it to the `helm install`
command.

For example, if we have the following `my-custom-values.yaml` file:

```yaml
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
```

Then we can use it in our install with the following command:

```sh
helm install my-custom-values ./health_runner \
  --values my-custom-values.yaml
```

> Note this happens to recreate the same install as using the `set` flag method.
> See ["Using the `--set` flag" section](#using-the-set-flag).


### Specifying Health Checks

You can also specify which health checks to run with the `set` flag within helm.

```
helm install my-nccl-health-check-only ./health_runner \
  --set health_checks.nccl_healthcheck.run_check=true
```

> Note this happens to still recreate the default behavior when installing the
> Chart as discussed in the [instructions above](#default-nccl-health-check).


Alternatively, you can instead edit `values.yaml` or provide an alternative YAML
file. For example, using your own values YAML file `my-custom-values.yaml`

```sh
helm install my-custom-release ./health_runner \
  --values my-custom-values.yaml
```

You can similarly specify multiple health checks and their associated values as
in this example:

```sh
helm install my-custom-nccl-check ./health_runner \
  --set health_checks.nccl_healthcheck.run_check=false \
  --set health_checks.gpu_healthcheck.run_check=true \
  --set health_checks.gpu_healthcheck.R_LEVEL="3" \
  --set health_checks.gpu_healthcheck.run_check=true \
  --set health_checks.gpu_healthcheck.IMAGE_TAG="subset" \
  --set health_checks.neper_healthcheck.run_check=false \
```

The above command diverges from the default configuration:
- Disables the NCCL health check
- Disables the Neper health check (explicitly; the default configuration
  disables the Neper health check)
- Enables the GPU health check
- Sets the GPU health check to use `R_LEVEL: "3"` (default is `R_LEVEL: "2"`)


## Generating YAML Files from Helm

Although we recommend using the provided Helm Chart for Bad Node Detector,
users may find it desirable to work with an equivalent YAML file instead of
using `helm install` directly.

Users can use [`helm template`](https://helm.sh/docs/helm/helm_template/) to
generate an equivalent YAML file. Then BND's health runner can be deployed with
a `kubectl apply` using the generated YAML file.


For example, using the Helm Chart's default values to generate an equivalent
YAML file (`my_generated_file.yaml`) can be done like so:

```bash
helm template my-default-health-check ./health_runner > my_generated_file.yaml
```

> Note you can alter the defaults with `helm template` in the same way as
> `helm install`. Such as this example:
> ```bash
> helm template my-custom-nccl-check ./health_runner \
>  --set health_checks.nccl_healthcheck.run_check=false \
>  --set health_checks.gpu_healthcheck.run_check=true \
>  --set health_checks.gpu_healthcheck.R_LEVEL="3" \
>  > my_gpu_health_check.yaml
> ```


The generated `my_generated_file.yaml` file can be deployed with the command
`kubectl apply -f my_generated_file.yaml`. The file `my_generated_file.yaml`
will contain the following:

```yaml
---
# Source: health_runner/templates/health_runner.yaml
# if .run_check # if .run_check # if .run_check # iteration over .Values.health_checks
## Below should be the same for all health checks
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ds-create
  namespace: default
---
# Source: health_runner/templates/health_runner.yaml
kind: ClusterRole
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: ds-create
rules:
- apiGroups: ["", "apps", "rbac.authorization.k8s.io", "batch"]
  resources: ["daemonsets", "serviceaccounts", "clusterrolebindings", "clusterroles", "nodes", "jobs", "services"]
  verbs: ["list", "get", "create", "delete", "watch", "patch"]
---
# Source: health_runner/templates/health_runner.yaml
kind: ClusterRoleBinding
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: ds-create
  namespace: default
subjects:
- kind: ServiceAccount
  name: ds-create
  namespace: default
roleRef:
  kind: ClusterRole
  name: ds-create
  apiGroup: rbac.authorization.k8s.io
---
# Source: health_runner/templates/health_runner.yaml
# yamllint disable # if .run_check
apiVersion: batch/v1
kind: CronJob
metadata:
  name: nccl-health-runner-a3plus
  labels:
    app: nccl-health-runner-a3plus
spec:
  timeZone: America/Los_Angeles
  schedule: "*/5 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          securityContext:
            runAsUser: 1000
            runAsGroup: 1000
            seccompProfile:
              type: RuntimeDefault
          serviceAccountName: ds-create
          containers:
          - name: "nccl-health-runner-a3plus"
            image: "us-docker.pkg.dev/gce-ai-infra/health-check/health-runner:subset"
            imagePullPolicy: Always
            securityContext:
              allowPrivilegeEscalation: false
              capabilities:
                drop:
                - ALL
            env:
            - name: "ACCELERATOR_TYPE"
              value: "nvidia-h100-mega-80gb"
            - name: "DRY_RUN"
              value: "true"
            - name: "IMAGE_TAG"
              value: "subset"
            - name: "SLEEP_TIME_MINUTES"
              value: "5"
            - name: "YAML_FILE"
              value: "a3plus/nccl_healthcheck.yaml" # iteration over .env # if .blast_mode.blast_mode_enabled # if .blast_mode scope
          restartPolicy: OnFailure
```