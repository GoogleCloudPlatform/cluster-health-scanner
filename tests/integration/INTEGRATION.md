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
for img in "registry.k8s.io/jobset/jobset:v0.6.0" "quay.io/brancz/kube-rbac-proxy:v0.13.1" "kindest/node:v1.30.0"; do docker pull "$img"; done
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
