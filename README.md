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

Clone CHS from Gerrit's develop branch

```bash
#Clone using https
git clone -b develop https://cluster-toolkit.googlesource.com/cluster-health-scanner

#Clone using sso
git clone -b develop https://github.com/GoogleCloudPlatform/cluster-health-scanner.git

cd cluster-health-scanner
```

*Note: All commands listed below must be run from this root cluster-health-scanner directory*

To install CHS, it is recommended to use a virtual environment:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install CHS
pip install .
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
chs list --tests --from-suite a3-ultra
```

#### 2. `run`
Executes a test suite on a specified cluster.

**Arguments:**
-   `--scheduler` / `-sch` **(Required)**: The scheduler to use.
    *   *Supported values:* `slurm`, `gke`.
-   `--machine-type` **(Required)**: The machine type of the cluster nodes (e.g., `a3-high`, `a3-edge`).
-   `--suite` / `-s`: Name of the Ramble application/suite to run.
    *   *Default:* `a3-ultra`
-   `--tests` / `-t`: Comma-separated list of specific tests to run.
    *   *Default (if not provided):* `nccl-cluster, nccl-pairwise, dcgm`
-   `--nodes`: Comma-separated list of individual nodes to run on (e.g., `node-01,node-02`).
-   `--nodelist`: Slurm node list pattern to run on (e.g., `node-[1-3]`).
-   `--nodepool`: GKE node pool to run on.
-   `--partition`: Slurm partition to run on.

**Example:**
```bash
# Run the default a3-ultra suite on Slurm with specific nodes
chs run --scheduler slurm --machine-type a3-high --nodes node-01,node-02

# Run specific tests
chs run --scheduler slurm --machine-type a3-high --tests nccl-cluster

# Run on GKE with a specific node pool
chs run --scheduler gke --machine-type a3-high --nodepool my-node-pool
```

**Output:**
Upon successful submission, CHS provides:
-   **Request ID**: The name of the created workspace (timestamp-based).
-   **Job IDs**: The Slurm job IDs for the submitted tests.

#### 3. `analyze`
Analyzes the results of a test run and generates a Bill of Health.

**Arguments:**
-   `RUN_ID` (Optional): The ID (Workspace Name) of the run to analyze.
    *   *Default:* The most recent run found in the workspaces directory.

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

Workspaces are created in: `~/.local/share/cluster-health-scanner/workspaces`

The workspace name follows the format: `yyyy-mm-dd_hh.mm.ss.ffffff`

The Bill of Health and experiment logs are located in the `chs-results` directory within the workspace.

This folder is what will be uploaded to GCS.

```
~/.local/share/cluster-health-scanner/workspaces/
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
`log_file: "/usr/local/google/home/user/.local/share/cluster-health-scanner/workspaces/2026-02-13_08.48.45.261141/chs-results/experiments/nccl-tests/all-gather/cluster_health_diagnostics-nccl-all-gather/cluster_health_diagnostics-nccl-all-gather.out"`

#### 4. `show results`
Displays the Bill of Health results in the terminal.

**Arguments:**
-   `RUN_ID` (Optional): The ID of the run to show.
    *   *Default:* The most recent run.

**Options:**
-   `--tests` / `-t`: Comma-separated list of test names to show detailed metrics for.

**Example:**
```bash
# Show summary of the latest run
chs show results

# Show details for specific tests
chs show results --tests nccl-cluster
```

## Configuration

### Workspaces Directory
CHS stores all run data, configurations, and logs in a dedicated workspaces directory.
-   **Linux**: `~/.local/share/cluster-health-scanner/workspaces`
-   **macOS**: `~/Library/Application Support/cluster-health-scanner/workspaces`

### Ramble Integration
CHS expects a standard Ramble installation. It looks for the Ramble setup script (`setup-env.sh`) relative to the installed `chs` package location (typically inside `site-packages/share/ramble`).

## Development & TODOs

The following features are currently in development or planned:

-   **Scheduler Support**: Currently, only `slurm` is fully supported. Support for `gke` (Google Kubernetes Engine) and others is planned.
-   **Dynamic Listing**: The `list` command currently acts as a placeholder and needs to be connected to the Ramble registry to dynamically list available suites and tests.
-   **Workspace Cleanup**: A `cleanup` command or logic to manage old workspaces is pending implementation.
-   **Configurable Paths**: Support for user-defined workspace directories via configuration or flags.
