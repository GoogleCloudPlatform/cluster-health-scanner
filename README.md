# Cluster Health Scanner (CHS)

Cluster Health Scanner (CHS) is a command-line tool designed to validate the health, performance, and infrastructure of GPU clusters on Google Kubernetes Engine (GKE) and Slurm, and of TPU clusters on GKE.

> **Note:** Looking for the legacy v1 (`cluster_diag` / Helm `health_runner`) implementation? It is preserved under the [`legacy-archive`](https://github.com/GoogleCloudPlatform/cluster-health-scanner/tree/legacy-archive) tag (`git clone -b legacy-archive https://github.com/GoogleCloudPlatform/cluster-health-scanner.git`).

## Overview

CHS helps site reliability engineers and cluster administrators verify that accelerator infrastructure is operating at peak performance before or between workloads:

- **Run Diagnostics:** Execute curated validation suites (`lite`, `standard`) or individual hardware and network benchmarks across Kubernetes and Slurm clusters.
- **Automatic Hardware & Scheduler Detection:** Automatically detects your active scheduler (`slurm`, `kubernetes`, or `standalone` on a single TPU VM) and node machine types.
- **Evaluate Health & Thresholds:** Evaluates benchmark results against built-in platform thresholds and generates an interactive HTML health report and a structured `bill_of_health.json`.
- **Discover Resources:** Query supported machine types, test suites, individual benchmarks, and schedulers for GPU and TPU platforms.

---

## 1. Prerequisites

Before running CHS, ensure your environment meets the following requirements:

### Google Cloud Storage (Optional)
GCS storage is optional. CHS operates entirely on local disk by default, saving health reports and diagnostic logs to `./chs-output/`. Remote storage may be configured via `chs configure storage gcs` if cloud persistence is desired.

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

#### Standalone (single TPU VM)
Run CHS on the TPU VM itself (`ct6e-standard-4t` or `tpu7x-standard-4t`) to test that VM; no Slurm or Kubernetes is needed. CHS uses standalone when it detects neither, or when you pass `--scheduler standalone`.

From the CHS Docker image, give the container access to the host:

```bash
docker run --rm -it --privileged --pid=host -v /:/host \
  -v chs-local-state:/workspace/.local/share \
  us-docker.pkg.dev/gce-ai-infra/chs/chs:latest run --scheduler standalone
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

   The CHS container requires read access to your host Kubernetes and Google Cloud credentials (`~/.kube` and `~/.config/gcloud`), uses a named Docker volume (`chs-local-state`) to persist run state and storage configuration, and bind-mounts `chs-output` to access exported reports and test logs directly in your current directory. Setting an alias lets you invoke `chs` commands directly:

   ```bash
   alias chs='docker run --rm -it --network host \
     -v ~/.kube:/workspace/.kube:ro \
     -v ~/.config/gcloud:/workspace/.config/gcloud:ro \
     -v chs-local-state:/workspace/.local/share \
     -v "$(pwd)/chs-output":/workspace/chs-output \
     us-docker.pkg.dev/gce-ai-infra/chs/chs:latest'
   ```

   > **Tip:** To run `chs run` in the background without blocking your terminal, replace `--rm -it` with `-d` and pass `--auto-approve` (`-y`).

### Option B: Python Virtual Environment (Slurm, Kubernetes, or Standalone)

```bash
# Create and activate a virtual environment
python3 -m venv chs-venv
source chs-venv/bin/activate

# Install CHS from GitHub
pip install git+https://github.com/GoogleCloudPlatform/cluster-health-scanner.git
```

---

## 3. Quick Start

Once installed (or with the Docker alias configured), start a diagnostic run. Results are saved locally under `./chs-output/<workspace_name>/`:

```bash
# Run the default 'lite' suite on your cluster
chs run --nodepools <nodepool_name>

# (Optional) Upload all subsequent run results to a pre-existing GCS bucket (only needed once per environment)
chs configure storage gcs <pre-existing-bucket-name>
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
Configure optional persistent GCS storage for CHS reports and workspace archives (results are always saved locally regardless). During configuration, CHS validates write access to the target bucket via a canary check and fails fast if permissions are missing.

- `chs configure storage gcs <BUCKET_OR_PATH>`: Set a pre-existing GCS bucket, or a path within it (e.g., `my-bucket`, `my-bucket/team-a`, or `gs://my-bucket/team-a`), as the upload destination.
- `chs configure storage show`: Display the currently configured storage target.
- `chs configure storage remove`: Remove the saved storage configuration.
- `chs configure telemetry enable|disable|status`: Turn usage telemetry on or off, or show whether it is on. See [Telemetry](#7-telemetry).

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
- `chs list schedulers`: List supported schedulers (`kubernetes`, `slurm`, `standalone`).

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

**Scheduler & Hardware Auto-detection:** CHS automatically detects the active scheduler (`slurm` via `sinfo`/`squeue`, then `kubernetes` via `kubectl`/in-cluster config, otherwise `standalone`, which tests the VM CHS runs on) and discovers the machine type of the target nodes. Pass `--scheduler` to skip auto-detection.

**Options:**
- `--suite`, `-s <SUITE_NAME>`: Pre-configured test suite to run (`lite` or `standard`). Defaults to `lite` when `--tests` is not specified.
- `--tests`, `-t <TEST_LIST>`: Comma-separated list of individual benchmarks to run instead of a full suite (e.g., `--tests nccl-cluster,gemm`).
- `--nodepools <NODEPOOL_LIST>`: Comma-separated list of Kubernetes node pools to target (can target single-host benchmarks across entire pools, and required for multi-host TPU benchmarks on Kubernetes).
- `--nodes <NODE_LIST>`: Comma-separated list of individual nodes to target (on Kubernetes, single-host TPU benchmarks support targeting static pool nodes, while multi-host benchmarks require whole-slice targeting via `--nodepools`).
- `--partitions <PARTITION_LIST>`: Comma-separated list of Slurm partitions to target.
- `--nodelists <NODE_PATTERN_LIST>`: Slurm nodelist pattern to target, e.g., `node-[1-4]`.
- `--auto-approve`, `-y`: Automatically confirm the execution preview without interactive prompting (useful for CI and scripts).
- `--run-id <RUN_ID>`: Custom identifier for the run.
- `--scheduler {kubernetes,slurm,standalone}`: Scheduler to use instead of auto-detection. `standalone` tests the TPU VM CHS runs on; `--nodes`, `--nodelists`, `--nodepools` and `--partitions` don't apply to it.
- `--output-dir <DIR>`: Directory where test logs, Bill of Health artifacts, and debug logs are saved, in a per-run `<DIR>/<workspace_name>/` subfolder. Overrides the default `./chs-output/` destination for this run. When running via Docker, use a path under `/workspace/chs-output` (or another bind-mounted directory); other paths exist only inside the container and are lost when it exits.
- `--storage-path <bucket-or-path>`: GCS destination bucket or path (e.g., `my-bucket`, `gs://my-bucket`, or `gs://my-bucket/prefix`) to upload results for this run, overriding the global storage configuration.

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
| `lite` (default) | ~5 mins | Quick health verification | `nvidia-smi`, `dcgm` (level 2), `gemm`, `nccl-single-node` | `tpu-info`, `tpu-gemm`, `tpu-hbm`, `tpu-host-device`, `tpu-collectives` |
| `standard` | ~30 mins | Thorough hardware, memory & collective validation | `nvidia-smi`, `dcgm` (level 3), `stream`, `nccl-cluster`, `nccl-pairwise`, `gemm` | `tpu-info`, `tpu-hbm`, `tpu-gemm`, `tpu-host-device`, `tpu-collectives`, `tpu-compute-stress`, `tpu-hbm-stress`, `tpu-network-stress` |

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
Safely terminates a CHS run: cancels its jobs, cleans up its CHS resources, and releases its nodes so other runs can use them. Without `RUN_ID`, your latest run is cancelled. `RUN_ID` can be any run on the cluster, including one started by another user. `--all` cancels **every** CHS run on the cluster, including other users' runs, and deletes all CHS resources. A run you started is cancelled on the scheduler it ran on; otherwise CHS auto-detects the scheduler. Prompts for `[y/N]` confirmation unless `--auto-approve` (`-y`) is passed.

```bash
# Cancel your latest run interactively
chs cancel

# Cancel a specific run without prompting
chs cancel my-cluster-lite-a1b2 --auto-approve

# Cancel every CHS run on the cluster
chs cancel --all
```

### `chs results`
Displays the Bill of Health summary in the terminal, followed by the locations of the run's artifacts: the interactive HTML report and the Bill of Health JSON. For runs uploaded to GCS these are `gs://` links; for local runs they are paths on disk (relative to the current directory when under it). If `RUN_ID` is omitted, CHS presents an interactive selector for your recent runs.

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

Upon completion of a run, CHS evaluates the collected metrics against platform thresholds. By default, CHS exports clean benchmark test logs, the interactive Bill of Health, and workspace diagnostics into a local output directory (`./chs-output/<workspace_name>/`), and optionally uploads artifacts to GCS if configured:

1. **Interactive HTML Diagnostic Report:** Saved locally at `./chs-output/<workspace_name>/bill_of_health.html` (and optionally uploaded to `gs://<bucket>/[<prefix>/]bills_of_health/<workspace_name>.html`).
2. **Bill of Health (JSON):** Saved locally at `./chs-output/<workspace_name>/bill_of_health.json` (and optionally uploaded to `gs://<bucket>/[<prefix>/]bills_of_health/<workspace_name>.json`) containing structured pass/fail statuses, extracted metrics, and log paths.
3. **Workspace Diagnostics and Raw Logs:** Saved locally under `./chs-output/<workspace_name>/` (containing test logs like `nvidia-smi-<node>.log`, `nccl-pairwise-*.log`, `debug.txt`, and `internal.log`).
4. **Workspace Archive (when GCS configured):** Uploaded to `gs://<bucket>/[<prefix>/]workspaces/<workspace_name>.tar.gz` containing full workspace outputs and configs.

Local workspaces are also preserved on the host (or inside the `chs-local-state` Docker volume) under `~/.local/share/chs/workspaces/<workspace_name>/`.

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

### Targeting TPUs (`--nodepools`)
TPUs are supported on Kubernetes and, through the `standalone` scheduler, on a single TPU VM; they are not supported on Slurm. Multi-host TPU benchmarks (such as `tpu-collectives` and `tpu-network-stress`) require an entire slice, so individual node targeting (`--nodes`) is not supported for them. On Kubernetes, single-host TPU benchmarks (such as `tpu-info`, `tpu-gemm`, and `tpu-hbm`) support targeting individual nodes with `--nodes`. On dynamic TPU pools on GKE (version 1.36.0-gke.3712000 or later), CHS provisions a single-host 2x2x1 slice for each targeted node whose 2x2x1 partition is `HEALTHY`, even when the surrounding 4x4x4 partition is degraded, and drops, with a warning, any node whose slice cannot be provisioned. Multi-host benchmarks on dynamic TPU pools require whole-partition provisioning via `--nodepools`. When running suites or mixed benchmarks with `--nodes` on Kubernetes, multi-host benchmarks requiring a full slice are automatically skipped with an informative warning. Additionally, all nodes targeted within a single `chs run` invocation must belong to the same machine type.

### GCS Bucket Creation
`chs configure storage gcs <bucket-name>` requires an existing GCS bucket with write permissions; CHS does not create new GCS buckets automatically.

---

## 7. Telemetry

CHS sends usage metrics to Google (`play.googleapis.com`) to help improve the tool. Telemetry is on by default. Metrics are sent in the background: at exit, CHS waits at most two seconds for the last request, and if a request fails with a network error, CHS stops sending for the rest of the command.

**Opting out:**
- `chs configure telemetry disable` turns telemetry off and saves the setting; `enable` turns it back on, and `status` shows whether it's on.
- Setting `CHS_TELEMETRY=0` or `DO_NOT_TRACK=1` turns telemetry off irrespective of the saved setting. Use this for CI and containers, e.g. `docker run -e CHS_TELEMETRY=0 ...`.

**Seeing what is sent:** Put `-v` before the command, for example `chs -v run`, and CHS prints all the telemetry data it sends.

---

## Contributing & Development

Interested in contributing or building CHS from source? See [CONTRIBUTING.md](CONTRIBUTING.md) and [DEVELOPMENT.md](DEVELOPMENT.md).
