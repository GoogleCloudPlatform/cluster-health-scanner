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

### Kubernetes Configuration

To run tests on Kubernetes, you first need to get the cluster credentials. For example:

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
-   `--machine-type <MACHINE_TYPE>`: Filter tests by machine type (requires `--tests`).

**Example:**
```bash
chs list --suites
chs list --tests --from-suite lite
chs list --tests --machine-type a3-ultra
chs list --tests --machine-type a3-ultra --from-suite lite
```

#### 2. `configure`
Configure CHS settings.

*Note: Configuring a storage target is optional. Setting a storage target allows CHS to upload reports and archives for persistence and sharing.*

**Commands:**
-   `storage gcs <BUCKET_NAME>`: Configures a Google Cloud Storage bucket.
-   `storage path <DIRECTORY_PATH>`: Configures a local or mounted directory path.

If no subcommand is provided (i.e., `chs configure storage`), CHS will start an interactive prompt to select and configure the storage type.

During configuration, CHS will validate write access to the target storage (via a canary write/delete test) and fail-fast if validation fails.

**Examples:**
```bash
# Configure GCS bucket
chs configure storage gcs my-results-bucket

# Configure Local Path storage
chs configure storage path /mnt/shared/chs-archive
```

#### 3. `run`
Executes a test suite on a specified cluster.

**Scheduler Auto-detection:**
CHS automatically detects the scheduler (`slurm` or `kubernetes`) of the active environment. Slurm detection is attempted first (by checking if `sinfo` and `squeue` are available and reachable). If Slurm is not detected, it falls back to Kubernetes (by checking for in-cluster credentials or `kubectl` connectivity).

**Arguments:**
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
-   `--nodepool`: Kubernetes node pool to run on.
-   `--partition`: Slurm partition to run on.
-   `--dry-run` `[slurm|kubernetes]`: Set up the workspace for the specified scheduler but do not execute (developer dry-run).
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

#### Kubernetes Examples

```bash
# Run the lite suite (default) on a3-ultra machine
chs run --machine-type a3-ultra

# Run the lite suite on a specific node pool
chs run --machine-type a3-ultra --nodepool my-node-pool

# Run the lite suite on specific nodes
chs run --machine-type a3-ultra --nodes node-01,node-02

# Run specific tests on specific nodes
chs run --machine-type a3-ultra --tests nccl-cluster,gemm --nodes node-01,node-02
```

#### Slurm Examples

```bash
# Run the lite suite (default) on a3-ultra machine
chs run --machine-type a3-ultra

# Run the lite suite on a specific partition
chs run --machine-type a3-ultra --partition my-partition

# Run the lite suite on specific nodes
chs run --machine-type a3-ultra --nodes node-01,node-02

# Run specific tests on specific nodes
chs run --machine-type a3-ultra --tests nccl-cluster,gemm --nodes node-01,node-02
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

**Output Directory Structure & Archiving:**

Workspaces are created locally in: `~/.local/share/chs/workspaces`
The workspace name (Run ID) follows the format: `yyyy-mm-dd_hh.mm.ss.ffffff`

Upon completion of analysis, CHS automatically archives the workspace and uploads it to the configured storage target (if configured):
1.  **Bill of Health**: Uploaded/copied to `<storage_target>/bills_of_health/<run_id>.json`.
2.  **Workspace Archive**: The workspace (configs, templates, and raw logs) is compressed into a tarball and uploaded/copied to `<storage_target>/workspaces/<run_id>.tar.gz`.

Local workspace directory structure:
```
~/.local/share/chs/workspaces/
└── <WORKSPACE_NAME>/            # e.g., 2026-02-13_08.48.45.261141
    ├── bill_of_health.json      # The generated Bill of Health report
    └── experiments/             # Raw logs and experiment data (created by Ramble)
        ├── nccl-tests/
        │   └── ...
        └── ...
```

The `bill_of_health.json` contains a `log_file` field for every test, pointing to the path of the log file relative to the workspace root.

**Example Log File Path in BoH:**
`log_file: "experiments/nccl-tests/all-gather/cluster_health_diagnostics-nccl-all-gather/cluster_health_diagnostics-nccl-all-gather.out"`
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

#### 7. `history`
Shows the history of the last 5 runs.

**Example:**
```bash
chs history
```

**Output:**
Displays a table with Workspace and Details.

**Example Output:**
```
Workspace                     Details
------------------------------------------------------------------------------------------
2026-05-05_16.23.13.228716    chs run --machine-type a3-ultra --dry-run kubernetes
2026-05-04_19.05.29.792632    chs run --machine-type a3-ultra
```


## Configuration

### Workspaces Directory
CHS stores all run data, configurations, and logs in a dedicated workspaces directory.
-   **Linux**: `~/.local/share/chs/workspaces`
-   **macOS**: `~/Library/Application Support/chs/workspaces`

### Ramble Integration
CHS expects a standard Ramble installation. It looks for the Ramble setup script (`setup-env.sh`) relative to the installed `chs` package location (typically inside `site-packages/share/ramble`).

## Validation Thresholds

CHS utilizes validation thresholds to automatically evaluate if a completed test run meets specific performance or health criteria. During analysis, CHS evaluates the extracted metrics against thresholds and records the outcome in the Bill of Health (`bill_of_health.json`).

You can configure these thresholds by providing a YAML file containing baseline criteria.

### Usage and CLI Flags

* **Specify custom thresholds:** Use the `--validation <PATH>` flag during `chs run` or `chs analyze` to specify a custom thresholds YAML file:
  ```bash
  chs run --machine-type a3-ultra --validation /path/to/custom_thresholds.yaml
  chs analyze --validation /path/to/custom_thresholds.yaml
  ```
* **Skip validation:** If you want to run tests or analyze results without applying any threshold checks, use the `--skip-validation` flag.
  ```bash
  chs run --machine-type a3-ultra --skip-validation
  ```

### Configuration Schema & Hierarchy

The thresholds configuration YAML file follows a two-level hierarchical structure:
1. **Top Level (`default` or Machine Type):** Defined either under the `default` block or under a specific hardware platform name (e.g., `a3-ultra`, `a3-mega`, `a4`, `a4x`).
2. **Second Level (Test Configurations):** Contains individual test configurations mapped by their test names (e.g., `nccl-single-node`, `gemm`, `nvidia-smi`, `stream`).

Each test configuration must specify:
* `type`: The validation comparison type (`min`, `scalar`, or `relative`).
* `thresholds`: A dictionary of metric names (or user-friendly aliases) mapped to threshold values.

#### Merge and Precedence Rules
* **Precedence:** Machine-specific configuration blocks (e.g., `a3-ultra`) take precedence over `default` configurations.
* **Merging:** When a machine type is active, its test-level configurations are merged with those under `default`. If a metric threshold is defined in both the machine block and `default`, the machine-specific threshold overrides the default.
* **No Configuration:** If a specific metric within a test has no thresholds defined under either the active machine type or `default`, that metric is not validated.

### Comparison Types

| Comparison Type | Evaluation Formula / Logic | Supported Types | Common Use Case Example |
| :--- | :--- | :--- | :--- |
| `min` | `Observed Value >= Threshold` | Numeric (`int`, `float`) | Bandwidth must be at least 80 GB/s. |
| `scalar` | `Observed Value == Threshold` | Numeric, Boolean, String (case-insensitive) | GPU Count must equal 8; Healthy must be `true`. |
| `relative` | `(Achieved Metric / Peak Metric) >= Threshold` | Numeric ratio (fractional value between 0 and 1) | Memory copy rate must reach at least 80% (0.8) of peak hardware bandwidth. |

#### 1. `min` (Minimum Value Validation)

Ensures that the metric achieves or exceeds a baseline numeric threshold.
* **Expected Value:** An integer or floating-point number.
* **Result Status:** `SUCCESS` if `Observed Value >= Threshold`; `FAILED` otherwise.

#### 2. `scalar` (Exact Match Validation)

Ensures that the metric matches the exact value specified.
* **Expected Value:** An integer, float, boolean, or string.
* **Result Status:**
  * **Numeric:** `SUCCESS` if `Observed Value == Threshold`.
  * **Boolean:** `SUCCESS` if `Observed Value == Threshold`.
  * **String:** `SUCCESS` if `Observed Value` (case-insensitive) `==` `Threshold` (case-insensitive).

#### 3. `relative` (Relative Performance Validation)

Calculates the ratio of a measured achieved metric against a theoretical hardware peak metric.
* **Expected Value:** A float (representing a fraction, e.g. `0.8` for 80%).
* **Evaluation:**
  * `Ratio = Achieved Metric / Peak Metric`
  * `Pass Condition: Ratio >= Threshold`
* **Ratio Rules & Edge Cases:**
  > [!CAUTION]
  > If the **Peak Metric** is missing from the results, or if its value is less than or equal to `0` (which would cause division by zero), the validation automatically **fails** with a `FAILED` status.

### Plugin-Specific Metric Aliases

To keep configurations simple and readable, CHS plugins translate complex raw benchmark metrics (Figures of Merit, or FOMs) into clean, human-readable aliases. You can use these aliases directly inside the `thresholds` section of your YAML file.

| Test Name | YAML Key (Alias) | Translated Raw FOM Name | Typical Comparison Type |
| :--- | :--- | :--- | :--- |
| `nccl-single-node` | `bandwidth` | `"Avg. Bus Bandwidth"` | `min` |
| `nccl-cluster` | `bandwidth` | `"Avg. Bus Bandwidth"` | `min` |
| `nccl-pairwise` | `bandwidth` | `"Avg. Bus Bandwidth"` | `min` |
| `gemm` | `performance` | `"Value"` | `min` |
| `nvidia-smi` | `gpus` | `"GPU Count"` | `scalar` |
| `stream` | `efficiency` | Achieved `"Rate"` vs. `"PEAK Bandwidth"` | `relative` |

> [!NOTE]
> This table lists the aliases currently supported by CHS for use in the validation YAML. Additional metrics may be available in the raw benchmark outputs but are not yet aliased for direct use in thresholds.

### Example `validation.yaml` Configuration
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

### Error Handling & Troubleshooting
* **Missing Metric:** If a threshold metric listed in the YAML file is not found in the test's results, a warning is logged (`Metric <name> not found in results`), and the validator skips evaluating that metric without raising an error.
* **Invalid YAML Structure:** If the YAML file is malformed or contains incorrect field types (e.g., `type` is not `min`, `scalar`, or `relative`), CHS raises a `ChsValidationError` before execution or analysis begins.


## Known Limitations

-   **GCS Storage Configuration**: For the `chs configure storage gcs` command, the specified GCS bucket must already exist; CHS does not create it automatically.

## Support and Feedback

To report bugs or request new features, please [create a new issue in Buganizer](https://github.com/GoogleCloudPlatform/cluster-health-scanner/issues).

## Development & TODOs

The following features are currently in development or planned:

-   **Scheduler Support**: Currently, only `slurm` and `kubernetes` are supported. Support for JBVMs and Bare Metals is planned.
-   **Dynamic Listing**: The `list` command currently acts as a placeholder and needs to be connected to the Ramble registry to dynamically list available suites and tests.
-   **Workspace Cleanup**: A `cleanup` command or logic to manage old workspaces is pending implementation.
-   **Configurable Paths**: Support for user-defined workspace directories via configuration or flags.
