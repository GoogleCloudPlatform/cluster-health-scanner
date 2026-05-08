# Cluster Health Scanner (CHS)

CHS provides a comprehensive suite of tools for validating the health and performance of Accelerators. It acts as a wrapper around [Ramble](https://github.com/GoogleCloudPlatform/ramble), managing workspaces, configuration, and execution of benchmark suites.

## Overview

CHS (Cluster Health Scanner) is designed to help site reliability engineers and cluster administrators ensure their infrastructure is operating at peak performance. It includes tools for:

-   **Running Tests**: Execute validation suites on schedulers like Slurm.
-   **Analyzing Results**: Inspect and interpret test outcomes, generating a "Bill of Health".
-   **Managing Resources**: List and filter available tests, suites, and schedulers.
-   **Visualizing Data**: View detailed results and metrics directly in the terminal.

## Prerequisites

### Slurm Configuration

**Note:** Commands for Slurm should be run from the Slurm controller node.

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

### GKE Configuration

To run tests on GKE, you first need to get the cluster credentials. For example:

```bash
gcloud container clusters get-credentials {cluster_name} \
    --location {location} \
    --project {project_id}
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

## Usage

The general syntax for CHS commands is:
```bash
chs [OPTIONS] COMMAND [ARGS]...
```

### Global Options
-   `-v`, `--verbose`: Enable verbose logging (useful for debugging).
-   `-h`, `--help`: Show the help message and exit.

---

### Commands

#### 1. `list`
Lists available resources (suites, tests, schedulers).
*Note: Currently, this command logs information but does not yet output a dynamic list.*

**Options:**
-   `--suites`: List available test suites.
-   `--tests`: List available tests.
-   `--schedulers`: List available schedulers.
-   `--from-suite <SUITE_NAME>`: specialized filter to list tests belonging to a specific suite (requires `--tests`).

**Example:**
```bash
chs list --suites
chs list --tests --from-suite lite
```

#### 2. `configure`
Configure CHS settings.

*Note: Configuring a GCS bucket is optional. Setting a bucket allows CHS to upload reports and archives to GCS for persistence and sharing.*

**Commands:**
-   `bucket`: Sets the GCS bucket for storing results.

**Options:**
-   `--name`: GCS bucket name (default: `chs-results`).

**Example:**
```bash
# Set the GCS bucket name
chs configure bucket --name my-results-bucket
```

#### 3. `run`
Executes a test suite on a specified cluster.

**Arguments:**
-   `--scheduler` / `-sch` **(Required)**: The scheduler to use.
    *   *Supported values:* `slurm`, `gke`.
-   `--machine-type` **(Required)**: The machine type of the cluster nodes.
    *   *Supported values:* `a3-ultra`, `a3-mega`, `a4`.
-   `--suite` / `-s`: Name of the test suite to run.
    *   *Default:* `lite`.
    *   *Supported values:* `lite`, `standard`, `burn-in`.
-   `--tests` / `-t`: Comma-separated list of specific tests to run.
    *   *Default (if not provided):* Loads tests from the selected suite.
    *   *Supported values:* `nvidia-smi`, `dcgm`, `gemm`, `nccl-single-node`, `stream`, `nccl-cluster`, `nccl-pairwise`.
-   `--nodes`: Comma-separated list of individual nodes to run on (e.g., `node-01,node-02`).
-   `--nodelist`: Slurm node list pattern to run on (e.g., `node-[1-3]`).
-   `--nodepool`: GKE node pool to run on.
-   `--partition`: Slurm partition to run on.
-   `--dry-run`: Set up the workspace but do not execute experiments or analyze.
-   `--validation`: Path to a user-defined thresholds YAML file (optional).
-   `--skip-validation`: Skip threshold evaluation entirely.

### Supported Suites and Tests

Below are the details of the supported test suites and the tests included in each suite, along with a brief description of each test.

#### Test Descriptions

-   **`nvidia-smi`**: Basic GPU health check (presence, visibility, and status).
-   **`dcgm`**: Diagnostics using NVIDIA Data Center GPU Manager.
-   **`gemm`**: Matrix multiplication performance test.
-   **`stream`**: Sustainable memory bandwidth measurement.
-   **`nccl-single-node`**: NCCL performance tests on a single node.
-   **`nccl-cluster`**: NCCL performance tests across all nodes in the cluster.
-   **`nccl-pairwise`**: NCCL performance tests between pairs of nodes.

#### Suite Composition

-   **`lite`**: Designed for quick verification.
    -   Included tests: `nvidia-smi`, `dcgm`, `gemm`, `nccl-single-node`
-   **`standard`**: Comprehensive health check.
    -   Included tests: `nvidia-smi`, `dcgm`, `stream`, `nccl-cluster`, `nccl-pairwise`
-   **`burn-in`**: Focused on stress testing.
    -   Included tests: `dcgm` (with higher diagnostic levels)

**Examples:**

#### GKE Examples

```bash
# Run the lite suite (default) on a3-ultra machine
chs run --scheduler gke --machine-type a3-ultra

# Run the lite suite on a specific node pool
chs run --scheduler gke --machine-type a3-ultra --nodepool my-node-pool

# Run the lite suite on specific nodes
chs run --scheduler gke --machine-type a3-ultra --nodes node-01,node-02

# Run specific tests on specific nodes
chs run --scheduler gke --machine-type a3-ultra --tests nccl-cluster,gemm --nodes node-01,node-02
```

#### Slurm Examples

```bash
# Run the lite suite (default) on a3-ultra machine
chs run --scheduler slurm --machine-type a3-ultra

# Run the lite suite on a specific partition
chs run --scheduler slurm --machine-type a3-ultra --partition my-partition

# Run the lite suite on specific nodes
chs run --scheduler slurm --machine-type a3-ultra --nodes node-01,node-02

# Run specific tests on specific nodes
chs run --scheduler slurm --machine-type a3-ultra --tests nccl-cluster,gemm --nodes node-01,node-02
```

**Output:**
Upon successful submission, CHS provides:
-   **Request ID**: The name of the created workspace (timestamp-based).
-   **Job IDs**: The Slurm job IDs for the submitted tests.

#### 4. `analyze`
Analyzes the results of a test run and generates a Bill of Health.

**Arguments:**
-   `RUN_ID` (Optional): The ID (Workspace Name) of the run to analyze.
    *   *Default:* The most recent run found in the workspaces directory.

**Options:**
-   `--validation`: Path to a user-defined thresholds YAML file to override the workspace copy (optional).
-   `--skip-validation`: Skip threshold evaluation entirely.

**Example:**
```bash
# Analyze the latest run
chs analyze

# Analyze a specific run
chs analyze 2023-10-27_10.30.00.123456
```

**Output:**
-   Generates a `bill_of_health.json` file in the run's directory.
-   Copies experiment logs to a user-accessible directory.

**Output Directory Structure:**

Workspaces are created in: `~/.local/share/chs/workspaces`

The workspace name follows the format: `yyyy-mm-dd_hh.mm.ss.ffffff`

The Bill of Health and experiment logs are located in the `chs-results` directory within the workspace.

This folder is what will be uploaded to GCS.

```
~/.local/share/chs/workspaces/
└── <WORKSPACE_NAME>/            # e.g., 2026-02-13_08.48.45.261141
    └── chs-results/             # Contains Bill of Health and experiment logs
        ├── bill_of_health.json  # The summarized Bill of Health report
        └── experiments/         # Raw logs and experiment data
            ├── nccl-tests/
            │   └── ...
            └── ...
```

The `bill_of_health.json` contains a `log_file` field for every test, pointing to the absolute path of the log file in the `experiments` directory.

**Example Log File Path:**
`log_file: "/usr/local/google/home/user/.local/share/chs/workspaces/2026-02-13_08.48.45.261141/chs-results/experiments/nccl-tests/all-gather/cluster_health_diagnostics-nccl-all-gather/cluster_health_diagnostics-nccl-all-gather.out"`
#### 5. `cancel`
Cancels running jobs for a specific test run.

**Arguments:**
-   `RUN_ID` (Optional): The ID (Workspace Name) of the run to cancel.
    *   *Default:* The most recent run.

**Example:**
```bash
# Cancel the latest run
chs cancel

# Cancel a specific run
chs cancel 2026-03-30_10.30.00.123456
```

#### 6. `results`
Displays the Bill of Health results in the terminal.

**Arguments:**
-   `RUN_ID` (Optional): The ID of the run to show.
    *   *Default:* The most recent run.

**Options:**
-   `--tests` / `-t`: Comma-separated list of test names to show detailed metrics for.
-   `--resources` / `-r`: Comma-separated list of resource names to show details for.

**Example:**
```bash
# Show summary of the latest run
chs results

# Show details for specific tests
chs results --tests nccl-cluster
```


## Configuration

### Workspaces Directory
CHS stores all run data, configurations, and logs in a dedicated workspaces directory.
-   **Linux**: `~/.local/share/chs/workspaces`
-   **macOS**: `~/Library/Application Support/chs/workspaces`

### Ramble Integration
CHS expects a standard Ramble installation. It looks for the Ramble setup script (`setup-env.sh`) relative to the installed `chs` package location (typically inside `site-packages/share/ramble`).

### Validation Thresholds Configuration

When using the `--validation` flag, you can provide a YAML file to define custom success criteria. The file follows a hierarchical structure where you can define `default` thresholds and machine-specific overrides.

Example `validation.yaml`:
```yaml
default:
  nccl-single-node:
    type: min
    thresholds:
      bandwidth: 50
  gemm:
    type: min
    thresholds:
      performance: 100

a3-ultra:
  nccl-single-node:
    type: min
    thresholds:
      bandwidth: 80
```
For a complete reference of supported tests and default values, see `src/chs/configs/thresholds.yaml`.

## Known Limitations

-   **A4X on GKE**: NCCL tests are currently not working on A4X machine types within GKE.
-   **GCS Bucket Configuration**: For the `chs configure bucket` command, the specified GCS bucket must already exist; CHS does not create it automatically.

## Development & TODOs

The following features are currently in development or planned:

-   **Scheduler Support**: Currently, only `slurm` and `gke` are supported. Support for JBVMs and Bare Metals is planned.
-   **Dynamic Listing**: The `list` command currently acts as a placeholder and needs to be connected to the Ramble registry to dynamically list available suites and tests.
-   **Workspace Cleanup**: A `cleanup` command or logic to manage old workspaces is pending implementation.
-   **Configurable Paths**: Support for user-defined workspace directories via configuration or flags.
