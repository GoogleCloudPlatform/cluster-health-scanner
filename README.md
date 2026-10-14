# Cluster Health Scanner (CHS)

Cluster Health Scanner (CHS) is a command-line tool designed to validate the health, performance, and infrastructure of GPU and TPU accelerator clusters on Google Kubernetes Engine (GKE) and Slurm.

> **Note:** Looking for the legacy v1 (`cluster_diag` / Helm `health_runner`) implementation? It is preserved under the [`legacy-archive`](https://github.com/GoogleCloudPlatform/cluster-health-scanner/tree/legacy-archive) tag (`git clone -b legacy-archive https://github.com/GoogleCloudPlatform/cluster-health-scanner.git`).

## Overview

CHS helps site reliability engineers and cluster administrators verify that accelerator infrastructure is operating at peak performance before or between workloads:

- **Run Diagnostics:** Execute curated validation suites (`lite`, `standard`) or individual hardware and network benchmarks across Kubernetes and Slurm clusters.
- **Automatic Hardware & Scheduler Detection:** Automatically detects your active scheduler (`slurm` or `kubernetes`) and node machine types.
- **Evaluate Health & Thresholds:** Evaluates benchmark results against built-in platform thresholds and generates an interactive HTML health report and a structured `bill_of_health.json`.
- **Discover Resources:** Query supported machine types, test suites, individual benchmarks, and schedulers for GPU and TPU platforms.

---

## 1. Prerequisites

Before running CHS, ensure your environment meets the following requirements:

### Google Cloud Storage (GCS) Bucket
You must manually pre-create a GCS bucket before running diagnostics. CHS uses this bucket to persist interactive HTML health reports, `bill_of_health.json` summaries, and diagnostic workspace archives.

### Cluster Credentials

#### Google Kubernetes Engine (GKE)
CHS can run from any environment that can access your GKE cluster. Fetch your cluster credentials using `gcloud`:

```bash
gcloud container clusters get-credentials <cluster_name> \
    --location <region_or_zone> \
    --project <project_id>
```

> **Note (Gang Scheduling):** CHS gang-schedules multi-node tests itself: each test runs as a Kubernetes Job that is created suspended and started only when all of its target nodes are free, so the CHS process must stay running for the duration of the run. If it is killed before it can clean up (for example by `kill -9`), run `chs cancel <RUN_ID>` to remove the leftover resources and release its nodes.

#### Slurm
Commands for Slurm should be run from the Slurm controller/login node. Enroot is required to run containerized workloads. Run the following commands on the Slurm cluster to configure Artifact Registry credentials for Enroot:

```bash
# 1. Verify gcloud authentication
gcloud auth list

# 2. Configure Docker to use the US Artifact Registry
gcloud auth configure-docker us-docker.pkg.dev

# 3. Create the Enroot configuration directory
mkdir -p ~/.enroot

# 4. Set the credentials file path
export ENROOT_CONFIG_CREDENTIALS="${HOME}/.enroot/.credentials"

# 5. Generate the Enroot credentials file using your gcloud access token
cat <<EOF > "${ENROOT_CONFIG_CREDENTIALS}"
machine us-docker.pkg.dev login oauth2accesstoken password $(gcloud auth application-default print-access-token)
EOF
```

---

## 2. Installation & Setup

You can run CHS either via the pre-built **Docker container image** (recommended for GKE / Kubernetes) or by installing the **Python CLI package** in a virtual environment (required for Slurm, optional for Kubernetes).

### Option A: Docker Image (Recommended for GKE / Kubernetes)

1. **Authenticate with Artifact Registry and pull the CHS image:**

   ```bash
   gcloud auth configure-docker us-docker.pkg.dev
   docker pull us-docker.pkg.dev/gce-ai-infra/chs/chs:latest
   ```

2. **Configure a terminal alias:**

   The CHS container requires read access to your host Kubernetes and Google Cloud credentials (`~/.kube` and `~/.config/gcloud`) and uses a named Docker volume (`chs-local-state`) to persist run state and storage configuration. Setting an alias lets you invoke `chs` commands directly:

   ```bash
   alias chs='docker run --rm -it --network host \
     -v ~/.kube:/workspace/.kube:ro \
     -v ~/.config/gcloud:/workspace/.config/gcloud:ro \
     -v chs-local-state:/workspace/.local/share \
     us-docker.pkg.dev/gce-ai-infra/chs/chs:latest'
   ```

   > **Tip:** To run `chs run` in the background without blocking your terminal, replace `--rm -it` with `-d` and pass `--auto-approve` (`-y`).

### Option B: Python Virtual Environment (Slurm or Kubernetes)

```bash
# Create and activate a virtual environment
python3 -m venv chs-venv
source chs-venv/bin/activate

# Install CHS from GitHub
pip install git+https://github.com/GoogleCloudPlatform/cluster-health-scanner.git
```

---

## 3. Quick Start

Once installed (or with the Docker alias configured), configure your GCS bucket once and start a diagnostic run:

```bash
# 1. Configure your pre-existing GCS bucket (only needed once per environment)
chs configure storage gcs <pre-existing-bucket-name>

# 2. Run the default 'lite' suite on your cluster
chs run --nodepools <nodepool_name>
```

---

## 4. Core Operations (CLI Commands)

The general syntax for CHS is:

```bash
chs [OPTIONS] COMMAND [ARGS]...
```

### Global Options
- `-v`, `--verbose`: Enable verbose debug logging.
- `--version`: Show the installed CHS version and exit.
- `-h`, `--help`: Show help message and exit. Append `--help` to any subcommand (e.g., `chs run --help` or `chs list --help`) for detailed usage and examples.

### `chs configure`
Configure persistent storage settings for CHS reports and workspace archives. During configuration, CHS validates write access to the target bucket via a canary check and fails fast if permissions are missing.

- `chs configure storage gcs <BUCKET_NAME>`: Set a pre-existing GCS bucket as the upload destination.
- `chs configure storage show`: Display the currently configured storage target.
- `chs configure storage remove`: Remove the saved storage configuration.

```bash
# Configure GCS bucket
chs configure storage gcs my-chs-results-bucket

# Verify configured storage
chs configure storage show
```

### `chs list`
Query supported machine types, test suites, individual benchmarks, and schedulers.

**Subcommands:**
- `chs list`: List all supported machine types, tests, suites, and schedulers.
- `chs list suites`: List pre-configured test suites (`lite`, `standard`).
- `chs list tests`: List individual benchmarks (optionally filtered by `--suite <SUITE_NAME>` or `--machine-type <MACHINE_TYPE>`).
- `chs list machine-types`: List supported accelerator machine types.
- `chs list schedulers`: List supported cluster schedulers (`kubernetes`, `slurm`).

**Accelerator Filters:**
Append `--tpu` or `--gpu` to any `chs list` command to filter results by accelerator family.

```bash
# List all TPU suites and tests
chs list suites --tpu
chs list tests --tpu

# List GPU tests included in the 'lite' suite
chs list tests --gpu --suite lite

# List tests compatible with a specific machine type
chs list tests --machine-type a3-ultragpu-8g
```

### `chs run`
Execute a diagnostic suite or custom list of tests on your cluster.

**Scheduler & Hardware Auto-detection:** CHS automatically detects the active scheduler (`slurm` via `sinfo`/`squeue`, falling back to `kubernetes` via `kubectl`/in-cluster config) and discovers the machine type of the target nodes.

**Options:**
- `--suite`, `-s <SUITE_NAME>`: Pre-configured test suite to run (`lite` or `standard`). Defaults to `lite` when `--tests` is not specified.
- `--tests`, `-t <TEST_LIST>`: Comma-separated list of individual benchmarks to run instead of a full suite (e.g., `--tests nccl-cluster,gemm`).
- `--nodepools <NODEPOOL_LIST>`: Comma-separated list of Kubernetes node pools to target (required when targeting TPUs on Kubernetes).
- `--nodes <NODE_LIST>`: Comma-separated list of individual GPU nodes to target (not supported for TPUs; use `--nodepools` on Kubernetes or `--partitions` on Slurm).
- `--partitions <PARTITION_LIST>`: Comma-separated list of Slurm partitions to target (required when targeting TPUs on Slurm).
- `--nodelists <NODE_PATTERN_LIST>`: Slurm nodelist pattern to target, e.g., `node-[1-4]` (not supported for TPUs; use `--partitions`).
- `--auto-approve`, `-y`: Automatically confirm the execution preview without interactive prompting (useful for CI and scripts).
- `--run-id <RUN_ID>`: Custom identifier for the run.

#### Supported Suites and Tests

##### GPU Benchmarks
- `nvidia-smi`: Basic GPU health and driver check (GPU count, power draw, clock frequencies, performance state).
- `dcgm`: Deep hardware diagnostics using NVIDIA Data Center GPU Manager (PCIe, memory, and compute stress tests).
- `gemm`: General Matrix Multiply (GEMM) floating-point throughput benchmark (TFLOPS).
- `stream`: Sustainable GPU memory bandwidth benchmark (GB/s).
- `nccl-single-node`: Intra-node GPU-to-GPU NCCL collective bandwidth and latency benchmark.
- `nccl-cluster`: Multi-node NCCL collective bandwidth benchmark across the cluster.
- `nccl-domain`: Intra-domain MNNVL NCCL bandwidth benchmark run per NVLink domain on `a4x-highgpu-4g`, `a4x-maxgpu-4g-metal`, and `a5x-highgpu-4g-metal`.
- `nccl-pairwise`: Point-to-point NCCL bandwidth and latency benchmark between node pairs to isolate degraded links.

##### TPU Benchmarks
- `tpu-info`: Basic TPU health check covering device count, `libtpu` version, and device topology.
- `tpu-gemm`: Generalized matrix multiplication benchmark measuring sustained TPU compute throughput (TFLOP/s).
- `tpu-hbm`: High Bandwidth Memory (HBM) read/write bandwidth benchmark.
- `tpu-host-device`: Host-to-device and device-to-host transfer bandwidth and latency check.
- `tpu-collectives`: Distributed cross-chip collective operations benchmark across TPU slices.
- `tpu-compute-stress`: Sustained matrix multiplication stress test for TPU compute cores.
- `tpu-hbm-stress`: Heavy synthetic load test for TPU High Bandwidth Memory stability.
- `tpu-network-stress`: Sustained point-to-point stress test across TPU interconnect (ICI) links.

##### Suite Composition
| Suite | Est. Duration | Description | Included GPU Tests | Included TPU Tests |
| --- | --- | --- | --- | --- |
| `lite` (default) | ~5 mins | Quick health verification | `nvidia-smi`, `dcgm` (level 2), `gemm`, `nccl-single-node` | `tpu-info`, `tpu-gemm`, `tpu-hbm`, `tpu-collectives` |
| `standard` | ~30 mins | Thorough hardware, memory & collective validation | `nvidia-smi`, `dcgm` (level 3), `stream`, `nccl-cluster`, `nccl-pairwise`, `gemm` | `tpu-info`, `tpu-hbm`, `tpu-gemm`, `tpu-collectives`, `tpu-compute-stress`, `tpu-hbm-stress`, `tpu-network-stress` |

#### Examples

**Kubernetes (GKE):**
```bash
# Run the default 'lite' suite on a specific node pool
chs run --nodepools my-node-pool

# Run the 'standard' validation suite on a TPU node pool
chs run --suite standard --nodepools tpu-pool-1

# Run specific GPU benchmarks on selected nodes
chs run --tests nccl-cluster,gemm --nodes node-01,node-02
```

**Slurm:**
```bash
# Run the default 'lite' suite on discovered idle nodes
chs run

# Run the 'lite' suite on a specific Slurm partition
chs run --partitions gpu-partition

# Run specific tests using a Slurm nodelist range
chs run --tests nccl-pairwise,stream --nodelists node-[1-4]
```

### `chs cancel`
Safely terminates a CHS run: cancels its jobs, cleans up its CHS resources, and releases its nodes so other runs can use them. Without `RUN_ID`, your latest run is cancelled. `RUN_ID` can be any run on the cluster, including one started by another user. `--all` cancels **every** CHS run on the cluster, including other users' runs, and deletes all CHS resources. Prompts for `[y/N]` confirmation unless `--auto-approve` (`-y`) is passed.

```bash
# Cancel your latest run interactively
chs cancel

# Cancel a specific run without prompting
chs cancel my-cluster-lite-a1b2 --auto-approve

# Cancel every CHS run on the cluster
chs cancel --all
```

### `chs results`
Displays the Bill of Health summary in the terminal and outputs the clickable GCS link to the interactive HTML diagnostic report. If `RUN_ID` is omitted, CHS presents an interactive selector for your recent runs.

```bash
# Select from recent runs interactively
chs results

# View results for a specific run ID
chs results my-cluster-lite-a1b2
```

### `chs history`
Displays a summary table of the last 5 runs, including Run ID, submission status, configured storage bucket, age, and the command executed.

```bash
chs history
```

---

## 5. Reporting & Diagnostic Artifacts

Upon completion of a run, CHS evaluates the collected metrics against platform thresholds and uploads three artifacts to your configured GCS bucket:

1. **Interactive HTML Diagnostic Report:** `gs://<bucket>/bills_of_health/<workspace_name>.html` (viewable in your browser via the `https://storage.cloud.google.com/...` link printed in the terminal).
2. **Bill of Health (JSON):** `gs://<bucket>/bills_of_health/<workspace_name>.json` containing structured pass/fail statuses, extracted metrics, and relative log paths.
3. **Workspace Archive:** `gs://<bucket>/workspaces/<workspace_name>.tar.gz` containing raw benchmark outputs, configurations, and debug logs.

Local workspaces are also stored on the host (or inside the `chs-local-state` Docker volume) under `~/.local/share/chs/workspaces/<workspace_name>/`.

---

## 6. FAQ & Troubleshooting

### `ERROR: The caller doesn't have permission to access the project <project-id>`
**Root Cause:** Your current environment is unauthenticated, using expired credentials, or the active identity (such as a default Compute Engine Service Account) lacks IAM permissions on the target GCP project or GCS bucket.

- **Case 1 (Override default Service Account with personal Application Default Credentials):**
  ```bash
  gcloud auth application-default login
  ```
- **Case 2 (Authenticate `gcloud` and `kubectl` CLI tools with your user identity):**
  ```bash
  gcloud auth login --no-browser
  ```

### Targeting TPUs (`--nodepools` / `--partitions`)
TPU benchmarks require an entire slice, so individual node or nodelist targeting (`--nodes`, `--nodelists`) is not supported for TPUs. Always target TPU resources using `--nodepools <nodepool_name>` on Kubernetes or `--partitions <partition_name>` on Slurm. Additionally, all nodes targeted within a single `chs run` invocation must belong to the same machine type.

### GCS Bucket Creation
`chs configure storage gcs <bucket-name>` requires an existing GCS bucket with write permissions; CHS does not create new GCS buckets automatically.

---

## Contributing & Development

Interested in contributing or building CHS from source? See [CONTRIBUTING.md](CONTRIBUTING.md) and [DEVELOPMENT.md](DEVELOPMENT.md).
