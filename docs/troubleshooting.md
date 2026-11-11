# Troubleshooting

## ERROR: The caller doesn't have permission to access the project ...

The active identity is unauthenticated, has expired credentials, or lacks IAM permissions on the project or bucket. This is common when a default Compute Engine service account is in use.

To use your own Application Default Credentials instead of the default service account:

```bash
gcloud auth application-default login
```

To authenticate `gcloud` and `kubectl` with your user identity:

```bash
gcloud auth login --no-browser
```

## Resources left behind after CHS was killed

On Kubernetes, the `chs` process must stay running for the duration of a run. If it was terminated before cleanup, run `chs cancel <RUN_ID>` (or `chs cancel --all`) to remove the leftover jobs and release the nodes.

## CHS runs in a container without access to the host

In standalone mode on a TPU VM, the CHS container must share the VM's PID namespace and see its filesystem. Start it with `--privileged --pid=host -v /:/host`. See [Option 2: Container image](../README.md#option-2-container-image).

## TPUs are busy: ...

In standalone mode, another process on the VM is using the TPUs. Wait for it to finish or stop it, then rerun `chs`.

## The memlock limit is ... bytes; native TPU tests need root or an unlimited memlock limit

In standalone mode on a TPU VM, run CHS as root or raise the limit with `ulimit -l unlimited` before running `chs`.

## Python 3.12 is not available

`pip install` fails with `Package 'chs' requires a different Python` when `python3` is older than 3.12. Create a Python 3.12 environment with `uv` (no root needed):

```bash
python3 -m venv uv-venv
uv-venv/bin/pip install uv
uv-venv/bin/uv venv --python 3.12 chs-venv
source chs-venv/bin/activate
uv-venv/bin/uv pip install git+https://github.com/GoogleCloudPlatform/cluster-health-scanner.git
```

## The TPU test dependencies are not installed

In standalone mode on a TPU VM, CHS runs the TPU tests from its own Python environment, and installs their dependencies only on Linux x86_64 with Python 3.12. Reinstall CHS in a Python 3.12 environment (see [Python 3.12 is not available](#python-312-is-not-available)), or use the container image.

## chs configure storage gcs fails

The bucket must already exist and the active identity must have write access to it. CHS does not create buckets.
