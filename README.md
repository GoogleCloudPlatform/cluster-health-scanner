# Cluster Health Scanner (CHS)
CHS provides a comprehensive suite of tools for validating the health and performance of Accelerators. It acts as a wrapper around Ramble, managing workspaces, configuration, and execution of benchmark suites.

## Overview
CHS (Cluster Health Scanner) is designed to help site reliability engineers and cluster administrators ensure their infrastructure is operating at peak performance. It includes tools for:

- **Running Tests:** Execute validation suites on schedulers like Slurm.
- **Analyzing Results:** Inspect and interpret test outcomes, generating a "Bill of Health".
- **Managing Resources:** List and filter available tests, suites, and schedulers.
- **Visualizing Data:** View detailed results and metrics directly in the terminal.

## Prerequisites
Because CHS interacts directly with your infrastructure to run workloads, you need to grant it access first. Follow the steps below for your specific environment (GKE or Slurm) to fetch your cluster credentials before running your first test.

### Google Cloud & Kubernetes Configuration (GKE)
To run tests on Kubernetes, you first need to get the cluster credentials. For example:

```bash
gcloud container clusters get-credentials {cluster_name} \
    --location {location} \
    --project {project_id}
```

**Note:**
**Pre-flight Checks:** CHS automatically runs pre-flight checks before executing tests. These checks automatically verify that your cluster environment is correctly configured (e.g., checking for required jobset and kueue versions). If you see "pre-flight" logs, this is expected behavior!


### Slurm Configuration

**Note:**
**Future Automation:** These manual Slurm configurations are planned to be automated directly into the CHS tool in a future release, similar to the GKE experience.

Note: Commands for Slurm should be run from the Slurm controller node.

Enroot is required to run the containerized workloads. Run the following commands to configure your credentials on the Slurm cluster.

```bash
# 1. Verify gcloud authentication
gcloud auth list

# 2. Configure Docker to use the US Docker registry
gcloud auth configure-docker us-docker.pkg.dev

# 3. Create the Enroot configuration directory
mkdir -p ~/.enroot

# 4. Set the credentials file path
export ENROOT_CONFIG_CREDENTIALS="${HOME}/.enroot/.credentials"

# 5. Generate the Enroot credentials file using your gcloud access token
cat <<EOF > "${ENROOT_CONFIG_CREDENTIALS}"
machine us-docker.pkg.dev login oauth2accesstoken password $(gcloud auth application-default print-access-token)
EOF

# 6. Verify the credentials were created successfully
cat "${ENROOT_CONFIG_CREDENTIALS}"
```

## Installation
To install CHS, it is recommended to use a virtual environment:

```bash
# Create a virtual environment
python3 -m venv chs-venv

# Activate the virtual environment
source chs-venv/bin/activate

# Install CHS directly from the repository
pip install git+https://cluster-toolkit.googlesource.com/cluster-health-scanner
```

## Execution Modes (GKE & Kubernetes)
CHS supports two different ways to execute tests on Kubernetes, depending on your needs:

### 1. Interactive CLI (For Quick Checks)
Use the `chs run` command when you want to run shorter tests and want to see the progress and results immediately in your terminal.

```bash
# Run the lite suite (default) on discovered nodes
chs run

# Run specific tests
chs run --tests nccl-cluster,gemm
```

**Tip:**
Machine types (e.g., A3 Mega) are automatically detected by the tool, so you do not need to manually specify the hardware type!

### 2. Background Shell Script (For Long-Running Jobs)
**Note:**
**GKE Only:** While the interactive `chs run` CLI works on any Kubernetes cluster, this shell script is a convenience wrapper built exclusively for Google Cloud (GKE) clusters to automate IAM and Workload Identity setups.

Use the `./bin/run_chs_gke.sh` script when you are running jobs that take several hours or days to complete. This method submits the job to the cluster in the background, preventing your local terminal from being blocked while the tests run. Results will be automatically uploaded to your configured storage bucket.

To configure your test run, edit the values in `helm/values.yaml` (e.g., `run.suite`, `run.nodepools`).

Then, simply run the script:

```bash
./bin/run_chs_gke.sh [optional-run-name]
```

This idempotent script fully automates Workload Identity prerequisites, IAM bindings, namespace configuration, and test execution.

## Usage
**TL;DR:** To run a standard health check on your cluster immediately, simply type:

```bash
chs run
```

The general syntax for CHS commands is:

```bash
chs [OPTIONS] COMMAND [ARGS]...
```

### Global Options
- `-v`, `--verbose`: Enable verbose logging (useful for debugging).
- `-h`, `--help`: Show the help message and exit.
  > **Tip:** You can append `--help` to *any* specific subcommand to see its exact arguments and examples (e.g., `chs run --help` or `chs list --help`).

## Commands

### 1. configure
Configure CHS settings.

**Note:** Configuring a storage target is optional. Setting a storage target allows CHS to upload reports and archives for persistence and sharing.

**Commands:**

- `storage gcs <BUCKET_NAME>`: Configures a Google Cloud Storage bucket.
- `storage path <DIRECTORY_PATH>`: Configures a local or mounted directory path.

If no subcommand is provided (i.e., `chs configure storage`), CHS will start an interactive prompt to select and configure the storage type.

During configuration, CHS will validate write access to the target storage (via a canary write/delete test) and fail-fast if validation fails.

**Examples:**

```bash
# Configure GCS bucket
chs configure storage gcs my-results-bucket

# Configure Local Path storage
chs configure storage path /mnt/shared/chs-archive
```

### 2. list
Lists available resources (suites, tests, schedulers)

**Options:**

- `--suites`: List available test suites (pre-configured bundles of tests designed for specific use-cases, like 'lite' or 'burn-in').
- `--tests`: List individual hardware and network benchmarks (e.g., `nccl-cluster`, `gemm`).
- `--schedulers`: List available workload managers (e.g., `kubernetes`, `slurm`).
- `--from-suite <SUITE_NAME>`: Filter tests to only show those belonging to a specific suite (requires `--tests`).
- `--machine-type <MACHINE_TYPE>`: Filter tests supported by a specific hardware accelerator, e.g., `a3-mega`. (Note: If omitted, CHS will list all available tests without filtering by hardware compatibility).

**Examples:**

```bash
# View all available test bundles
chs list suites

# See which specific tests belong to the 'lite' suite
chs list tests --suite lite

# Find all tests that are compatible with A3-Ultra hardware
chs list tests --machine-type a3-ultragpu-8g

# Combine filters: Find A3-Ultra tests within the 'lite' suite
chs list tests --machine-type a3-ultragpu-8g --suite lite

# List only TPU tests
chs list tpu tests

# List only GPU tests
chs list gpu tests
```

### 3. run
Executes a test suite on a specified cluster.

**Scheduler Auto-detection:** CHS automatically detects the scheduler (`slurm` or `kubernetes`) of the active environment. Slurm detection is attempted first (by checking if `sinfo` and `squeue` are available and reachable). If Slurm is not detected, it falls back to Kubernetes (by checking for in-cluster credentials or `kubectl` connectivity).

**Arguments:**

- `--suite` / `-s`: Name of the test suite to run (pre-configured bundles of tests designed for specific use-cases, like 'lite' or 'burn-in').
  - Default: `lite`.
  - Supported values: `lite`, `standard`, `burn-in`.
- `--tests` / `-t`: Comma-separated list of individual hardware and network benchmarks (e.g., `nccl-cluster`, `gemm`).
  - Default (if not provided): Loads tests from the selected suite.
  - Supported values: `nvidia-smi`, `dcgm`, `gemm`, `nccl-single-node`, `stream`, `nccl-cluster`, `nccl-pairwise`, `tpu-info`, `hbm`, `host-device`, `collectives`, `collectives-single-node`.
- `--nodes`: Comma-separated list of individual nodes to run on (e.g., `node-01,node-02`).
- `--nodelists`: Slurm node list pattern/lists to run on (e.g., `node-[1-3]`).
- `--nodepools`: Kubernetes node pools to run on.
- `--partitions`: Slurm partitions to run on.
- `--validation`: Path to a user-defined thresholds YAML file (optional).
- `--skip-validation`: Skip threshold evaluation entirely.
- `--run-id`: Custom run ID for this execution.
- `--attach`: Attach to an in-flight run. If the previous run was interrupted mid-submission, it is canceled and restarted.

#### Supported Suites and Tests
Below are the details of the supported test suites and the tests included in each suite, along with a brief description of each test.

##### Test Descriptions

**GPU Tests**
- `nvidia-smi`: Basic GPU health check (presence, visibility, and status).
- `dcgm`: Diagnostics using NVIDIA Data Center GPU Manager.
- `gemm`: Matrix multiplication performance test.
- `stream`: Sustainable memory bandwidth measurement.
- `nccl-single-node`: NCCL performance tests on a single node.
- `nccl-cluster`: NCCL performance tests across all nodes in the cluster.
- `nccl-pairwise`: NCCL performance tests between pairs of nodes.

**TPU Tests**
- `tpu-info`: Collects TPU system information and topology.
- `hbm`: High Bandwidth Memory performance tests.
- `host-device`: Host-to-device and device-to-host bandwidth tests.
- `collectives`: TPU collectives performance benchmarks across nodes.
- `collectives-single-node`: TPU collectives benchmarks on a single node.

##### Suite Composition
- **lite**: Designed for quick verification.
  - Included GPU tests: `nvidia-smi`, `dcgm`, `gemm`, `nccl-single-node`
  - Included TPU tests: `tpu-info`, `hbm`, `host-device`, `collectives-single-node`
- **standard**: Comprehensive health check.
  - Included GPU tests: `nvidia-smi`, `dcgm`, `stream`, `nccl-cluster`, `nccl-pairwise`
  - Included TPU tests: `tpu-info`, `collectives`
- **burn-in**: Focused on stress testing.
  - Included GPU tests: `dcgm` (with higher diagnostic levels)

#### Examples:

**Kubernetes Examples**
```bash
# Run the lite suite (default) on discovered nodes
chs run

# Run the lite suite on a specific node pool
chs run --nodepools my-node-pool

# Run the lite suite on specific nodes
chs run --nodes node-01,node-02

# Run specific tests on specific nodes
chs run --tests nccl-cluster,gemm --nodes node-01,node-02
```

**Slurm Examples**
```bash
# Run the lite suite (default) on discovered nodes
chs run

# Run the lite suite on a specific partition
chs run --partitions my-partition

# Run the lite suite on specific nodes
chs run --nodes node-01,node-02

# Run specific tests on specific nodes
chs run --tests nccl-cluster,gemm --nodes node-01,node-02
```

**Output:** Upon successful submission, CHS provides:
- **Request ID:** The name of the created workspace (timestamp-based).
- **Job IDs:** The Slurm job IDs for the submitted tests.

### 4. cancel
Safely terminates ALL running CHS jobs and removes stale resources on the active cluster. Because this action acts as a global abort and deletes all CHS resources, it requires an explicit `[y/N]` confirmation.

**Example:**

```bash
# Cancel all active runs
chs cancel

# Cancel without prompting (useful for scripts)
chs cancel --auto-approve
```

### 5. history
Shows the history of the last 5 runs.

**Example:**

```bash
chs history
```

**Output:** Displays a table with Workspace and Details.

**Example Output:**

```
Workspace                     Details
------------------------------------------------------------------------------------------
2026-05-05_16.23.13.228716    chs run kubernetes
2026-05-04_19.05.29.792632    chs run
```

### 6. results
Shows the generated "Bill of Health" results and metric details for a run. If validation is enabled, this will display two separate tables in the terminal: one for all raw metrics and one strictly for validated metrics. If no `RUN_ID` is provided, you will be prompted to select one.

**Arguments:**

- `RUN_ID` (Optional): The ID (Workspace Name) of the run to view.

**Options:**

- `--tests` / `-t`: Comma-separated list of test names to show details for.
- `--resources` / `-r`: Comma-separated list of resource names to show details for.

**Example:**

```bash
# View the Bill of Health for the most recent run
chs results

# View the Bill of Health for a specific run
chs results sim-a3mega-chs-custom-lplt

# View details for specific tests
chs results --tests nccl-cluster,gemm
```

**Output Directory Structure & Archiving:**

Workspaces are created locally in: `~/.local/share/chs/workspaces`
The workspace name (Run ID) follows the format: `yyyy-mm-dd_hh.mm.ss.ffffff`

Upon completion of analysis, CHS automatically archives the workspace and uploads it to the configured storage target (if configured):
- **Bill of Health:** Uploaded/copied to `<storage_target>/bills_of_health/<run_id>.json`.
- **Workspace Archive:** The workspace (configs, templates, and raw logs) is compressed into a tarball and uploaded/copied to `<storage_target>/workspaces/<run_id>.tar.gz`.

Local workspace directory structure:

```text
~/.local/share/chs/workspaces/
└── <WORKSPACE_NAME>/            # e.g., 2026-02-13_08.48.45.261141
    ├── bill_of_health.json      # The generated Bill of Health report
    └── experiments/             # Raw logs and experiment data (created by Ramble)
        ├── nccl-tests/
        │   └── ...
        └── ...
```

The `bill_of_health.json` contains a `log_file` field for every test, pointing to the path of the log file relative to the workspace root.

**Example Log File Path in BoH:** `log_file: "experiments/nccl-tests/all-gather/cluster_health_diagnostics-nccl-all-gather/cluster_health_diagnostics-nccl-all-gather.out"`

## Docker Execution (Kubernetes Only)
CHS can be run natively via Docker without needing to install any host dependencies (other than Docker and valid GCP/Kubernetes credentials). This containerized execution strictly targets Kubernetes environments.

The Docker execution model runs the internal processes as `root` and uses Docker Named Volumes (`chs-local-state`) for persistent state to avoid host permission errors while reading read-only host credentials (like `~/.kube/config`).

### 0. Building and Pushing the Docker Image
If you make code changes and need to build a new CHS Docker image for your cluster:

```bash
# Set your artifact registry repository path (matches helm/values.yaml)
export REPO="us-docker.pkg.dev/gce-ai-infra/chs/chs"

# Build the image locally
docker build -t ${REPO}:latest .

# Configure Docker authentication for Artifact Registry
gcloud auth configure-docker us-docker.pkg.dev

# Push the image to the registry
docker push ${REPO}:latest
```

### 1. Initial Storage Configuration (Run Once)
Before triggering a run, configure where the archives should be saved. This configuration is permanently saved into the `chs-local-state` volume.

**Option A: Archiving to GCS**

```bash
docker run -it --network host \
  -v ~/.kube:/workspace/.kube:ro \
  -v ~/.config/gcloud:/workspace/.config/gcloud:ro \
  -v chs-local-state:/workspace/.local/share \
  chs:latest configure storage gcs --bucket-name my-bucket
```

**Option B: Archiving to a Local Host Directory**
If you want to extract the archive tarballs directly to your machine, bind-mount a host directory to the container (e.g., `-v ~/chs-archives:/archive`):

```bash
docker run -it --network host \
  -v ~/.kube:/workspace/.kube:ro \
  -v ~/.config/gcloud:/workspace/.config/gcloud:ro \
  -v chs-local-state:/workspace/.local/share \
  -v ~/chs-archives:/archive \
  chs:latest configure storage path /archive
```

### 2. Running Diagnostics
Because configuration is saved in the `chs-local-state` volume, subsequent runs will automatically respect the chosen storage backend.

```bash
docker run -it --network host \
  -v ~/.kube:/workspace/.kube:ro \
  -v ~/.config/gcloud:/workspace/.config/gcloud:ro \
  -v chs-local-state:/workspace/.local/share \
  -v ~/chs-archives:/archive \
  chs:latest run --suite lite
```

*(Note: If using GCS, the `~/chs-archives:/archive` mount can be omitted).*

### 3. Native GKE Deployment
CHS can be deployed inside the cluster using the bundled Helm chart. The orchestrator pod prefers CPU nodes and uses a specialized `--attach` mechanism to gracefully recover state if the pod is preempted while jobs are running.

It is highly recommended to use the provided turnkey execution script (which automatically drives Helm and configures security bindings) rather than running Helm manually:

```bash
# 1. Configure your run preferences in helm/values.yaml
# 2. Run the deployment script:
./bin/run_chs_gke.sh
```

See the `helm/` directory for additional configuration values.

## Configuration

### Workspaces Directory
CHS stores all run data, configurations, and logs in a dedicated `workspaces` directory.

- **Linux:** `~/.local/share/chs/workspaces`
- **macOS:** `~/Library/Application Support/chs/workspaces`

### Ramble Integration
CHS expects a standard Ramble installation. It looks for the Ramble setup script (`setup-env.sh`) relative to the installed `chs` package location (typically inside `site-packages/share/ramble`).

## Validation Thresholds
CHS utilizes validation thresholds to automatically evaluate if a completed test run meets specific performance or health criteria. During run execution, CHS evaluates the extracted metrics against thresholds and records the outcome in the Bill of Health (`bill_of_health.json`).

You can configure these thresholds by providing a YAML file containing baseline criteria.

### Usage and CLI Flags
**Specify custom thresholds:** Use the `--validation <PATH>` flag during `chs run` to specify a custom thresholds YAML file:

```bash
chs run --tests nccl-cluster --validation /path/to/custom_thresholds.yaml
```

**Skip validation:** If you want to run tests without applying any threshold checks, use the `--skip-validation` flag.

```bash
chs run --skip-validation
```

### Configuration Schema & Hierarchy
The thresholds configuration YAML file follows a two-level hierarchical structure:

1. **Top Level (default or Machine Type):** Defined either under the `default` block or under a specific hardware platform name (e.g., `a3-ultra`, `a3-mega`, `a4`, `a4x`).
2. **Second Level (Test Configurations):** Contains individual test configurations mapped by their test names (e.g., `nccl-single-node`, `gemm`, `nvidia-smi`, `stream`).

Each test configuration must specify:
- `type`: The validation comparison type (`min`, `scalar`, or `relative`).
- `thresholds`: A dictionary of metric names (or user-friendly aliases) mapped to threshold values.

### Merge and Precedence Rules
- **Precedence:** Machine-specific configuration blocks (e.g., `a3-ultra`) take precedence over `default` configurations.
- **Merging:** When a machine type is active, its test-level configurations are merged with those under `default`. If a metric threshold is defined in both the machine block and default, the machine-specific threshold overrides the default.
- **No Configuration:** If a specific metric within a test has no thresholds defined under either the active machine type or default, that metric is not validated.

### Comparison Types

| Comparison Type | Evaluation Formula / Logic | Supported Types | Common Use Case Example |
|---|---|---|---|
| min | Observed Value >= Threshold | Numeric (int, float) | Bandwidth must be at least 80 GB/s. |
| scalar | Observed Value == Threshold | Numeric, Boolean, String (case-insensitive) | GPU Count must equal 8; Healthy must be true. |
| relative | (Achieved Metric / Peak Metric) >= Threshold | Numeric ratio (fractional value between 0 and 1) | Memory copy rate must reach at least 80% (0.8) of peak hardware bandwidth. |

#### 1. min (Minimum Value Validation)
Ensures that the metric achieves or exceeds a baseline numeric threshold.
- **Expected Value:** An integer or floating-point number.
- **Result Status:** `SUCCESS` if Observed Value >= Threshold; `FAILED` otherwise.

#### 2. scalar (Exact Match Validation)
Ensures that the metric matches the exact value specified.
- **Expected Value:** An integer, float, boolean, or string.
- **Result Status:**
  - **Numeric:** `SUCCESS` if Observed Value == Threshold.
  - **Boolean:** `SUCCESS` if Observed Value == Threshold.
  - **String:** `SUCCESS` if Observed Value (case-insensitive) == Threshold (case-insensitive).

#### 3. relative (Relative Performance Validation)
Calculates the ratio of a measured achieved metric against a theoretical hardware peak metric.
- **Expected Value:** A float (representing a fraction, e.g. 0.8 for 80%).
- **Evaluation:**
  - Ratio = Achieved Metric / Peak Metric
  - Pass Condition: Ratio >= Threshold
- **Ratio Rules & Edge Cases:**
  **Caution:**
  If the Peak Metric is missing from the results, or if its value is less than or equal to 0 (which would cause division by zero), the validation automatically fails with a `FAILED` status.

### Plugin-Specific Metric Aliases
To keep configurations simple and readable, CHS plugins translate complex raw benchmark metrics (Figures of Merit, or FOMs) into clean, human-readable aliases. You can use these aliases directly inside the `thresholds` section of your YAML file.

| Test Name | YAML Key (Alias) | Translated Raw FOM Name | Typical Comparison Type |
|---|---|---|---|
| `nccl-single-node` | `bandwidth` | "Avg. Bus Bandwidth" | `min` |
| `nccl-cluster` | `bandwidth` | "Avg. Bus Bandwidth" | `min` |
| `nccl-pairwise` | `bandwidth` | "Avg. Bus Bandwidth" | `min` |
| `gemm` | `performance` | "Value" | `min` |
| `nvidia-smi` | `gpus` | "GPU Count" | `scalar` |
| `stream` | `efficiency` | Achieved "Rate" vs. "PEAK Bandwidth" | `relative` |

**Note:**
This table lists the aliases currently supported by CHS for use in the validation YAML. Additional metrics may be available in the raw benchmark outputs but are not yet aliased for direct use in thresholds.

### Example validation.yaml Configuration

```yaml
# Default baselines for any machine type
default:
  nccl-single-node:
    type: min
    thresholds:
      bandwidth: 50 # Bandwidth must be >= 50 GB/s
  gemm:
    type: min
    thresholds:
      performance: 100 # Performance must be >= 100 TFLOPS

# Specific overrides and additional tests for a3-ultra
a3-ultra:
  nccl-single-node:
    type: min
    thresholds:
      bandwidth: 80 # Overrides default bandwidth to require >= 80 GB/s
  stream:
    type: relative
    thresholds:
      efficiency: 0.8 # Achieved Rate / Peak Bandwidth must be >= 80% (0.8)
  nvidia-smi:
    type: scalar
    thresholds:
      gpus: 8 # GPU Count must be exactly 8
```

## Error Handling & Troubleshooting
- **Missing Metric:** If a threshold metric listed in the YAML file is not found in the test's results, a warning is logged (`Metric <name> not found in results`), and the validator skips evaluating that metric without raising an error.
- **Invalid YAML Structure:** If the YAML file is malformed or contains incorrect field types (e.g., `type` is not `min`, `scalar`, or `relative`), CHS raises a `ChsValidationError` before execution or analysis begins.

## Known Limitations
- **GCS Storage Configuration:** For the `chs configure storage gcs` command, the specified GCS bucket must already exist; CHS does not create it automatically.

## Development & TODOs
The following features are currently in development or planned:

- **Scheduler Support:** Currently, only `slurm` and `kubernetes` are supported. Support for JBVMs and Bare Metals is planned.
- **Dynamic Listing:** The `list` command currently acts as a placeholder and needs to be connected to the Ramble registry to dynamically list available suites and tests.
- **Workspace Cleanup:** A cleanup command or logic to manage old workspaces is pending implementation.
- **Configurable Paths:** Support for user-defined workspace directories via configuration or flags.
