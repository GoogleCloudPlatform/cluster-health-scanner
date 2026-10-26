# Integration Tests Description

## Prerequisites & Kind Installation

Local integration tests require **Docker**, **`kind`**, and **`kubectl`** installed on your host machine or Cloudtop.

### 1. Install Docker
Ensure Docker is installed and your user has permission to run containers:
```bash
docker info
```

### 2. Install `kind` (Kubernetes in Docker)
Download and install the `kind` binary to `~/.local/bin`:
```bash
# Download kind binary (Linux amd64)
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.23.0/kind-linux-amd64
chmod +x ./kind
mkdir -p ~/.local/bin
mv ./kind ~/.local/bin/kind

# Ensure ~/.local/bin is in PATH
export PATH="$HOME/.local/bin:$PATH"
```

Verify installation:
```bash
kind version
```

### 3. Install `kubectl`
Download and install `kubectl`:
```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x ./kubectl
mv ./kubectl ~/.local/bin/kubectl
```

### 4. First Time Setup (Pre-pull Images)
To speed up test runs and avoid timeouts on the first run, pre-pull the required Docker images:
```bash
for img in "kindest/node:v1.30.0" "alpine:latest"; do docker pull "$img"; done
```

---

## Running Integration Tests on Specific Clusters

You can run integration tests for a particular set of clusters using pytest's marker (`-m`) or keyword filtering (`-k`) flags. (Adding `-s` prints logs during test execution).

### By Accelerator Type (Using Markers)
The test suite utilizes custom markers for different accelerator types. You can run tests exclusively for one type:

- **TPU clusters only**:
  ```bash
  pytest -m tpu tests/integration/ -s
  ```

- **GPU clusters only**:
  ```bash
  pytest -m gpu tests/integration/ -s
  ```

### By Cluster Topology/Version (Using Keyword Filtering)
Because test parameters include the configuration file paths (e.g., `tpu/v6e/multi_pool_2x4_4x4.yaml`), you can filter by any substring in the path using the `-k` flag.

- **v6e clusters only**:
  ```bash
  pytest -k "v6e" tests/integration/ -s
  ```

- **tpu7x clusters only**:
  ```bash
  pytest -k "tpu7x" tests/integration/ -s
  ```

- **Specific configuration**:
  ```bash
  pytest -k "multi_pool_2x4_4x4" tests/integration/ -s
  ```

---

## Kubeconfig Isolation and Cluster Reuse

- If `KUBECONFIG` is unset, the suite points it at `~/.kube/chs-integration-kind.yaml`, so Kind never edits `~/.kube/config`. Set `KUBECONFIG` yourself to use a different file.
- The session aborts unless `kubectl config current-context` is the Kind cluster context (`kind-chs-kind-cluster`), so tests never run against a real cluster.
- Set `CHS_INTEGRATION_KEEP_CLUSTER=1` to keep the Kind cluster after the session instead of deleting it, which saves the cluster boot time on the next run.

## Concurrent Run Tests

`concurrency_test.py` seeds Jobs of another CHS run in the `chs` namespace, labeled `app.kubernetes.io/managed-by: chs` and `app.kubernetes.io/instance: <run_id>` and pinned to nodes with `kubernetes.io/hostname` node affinity, like the Jobs CHS submits. It checks that:
- a run targeting a node that a Job of another run is pinned to is rejected, naming that run, and submits nothing; this holds for suspended, running and completed Jobs, since a run's Jobs target their nodes until the run is torn down;
- a run on disjoint nodes succeeds, leaves the other run's Jobs and Service untouched, and deletes its own Jobs, Secrets and `chs-<run_id>` Service while keeping the namespace;
- Jobs without the instance label do not block a run;
- two `chs run` processes on disjoint nodes both succeed, with separate or shared `XDG_DATA_HOME`;
- `chs cancel <run_id>` deletes only that run's Jobs, Secrets and Service, even when another run's ID starts with `<run_id>-`;
- the same rules hold for TPU nodepools, and a dynamic TPU Slice of another run covering a pool's partition blocks that pool.

It provisions `configs/concurrency/gpu_a4x_four_nodes.yaml` (4 GPU nodes) and `tpu/tpu7x/multi_pool_2x2x2_2x2x4.yaml`:
```bash
pytest tests/integration/concurrency_test.py -s
```
