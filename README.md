# Cluster Health Scanner (CHS)

CHS provides a comprehensive suite of tools for validating the health and performance of Accelerators.

## Overview

CHS (Cluster Health Scanner) is designed to help site reliability engineers and cluster administrators ensure their infrastructure is operating at peak performance. It includes tools for:

- **Running Tests**: Execute validation suites on schedulers like Slurm or GKE.
- **Analyzing Results**: Inspect and interpret test outcomes.
- **Managing Resources**: List and filter available tests, suites, and schedulers.

## Installation

```bash
pip install .
```

## Usage

```bash
# List available test suites
chs list --suites

# Run a specific test suite
chs run --suite <suite_name> --scheduler <scheduler_type>

# Analyze a test run
chs analyze <run_id>

# Get help
chs --help
```
