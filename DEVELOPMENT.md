# Developer Workflow

This document outlines the workflow for developing, testing, and contributing to the Cluster Health Scanner (CHS) project.
All development should be done on the `develop` branch. Main branch is strictly protected and should only be used for releases.

## 1. Getting Started

### Prerequisites

*   Python 3.12 or higher
*   `git`
*   `pip`

### Forking and Cloning the Repository

1. Fork the [GoogleCloudPlatform/cluster-health-scanner](https://github.com/GoogleCloudPlatform/cluster-health-scanner) repository on GitHub (make sure to uncheck *"Copy the `main` branch only"* so your fork includes the `develop` branch).
2. Clone your fork locally and add the upstream repository as a remote:

```bash
git clone -b develop https://github.com/<your-username>/cluster-health-scanner.git chs
cd chs
git remote add upstream https://github.com/GoogleCloudPlatform/cluster-health-scanner.git
```

### Environment Setup

It is strongly recommended to use a virtual environment for development.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the package in editable mode with development dependencies
# Note: CHS uses pyproject.toml and setup.py build system.
pip install -e .
```

If you encounter issues with `pip`, try clearing your cache:
```bash
pip cache purge
pip install -e .
```

**Note:** CHS depends on a specific branch of [Ramble](https://github.com/Ramble-Project/ramble). The `pyproject.toml` is configured to fetch it from the correct fork/branch.


### Protobuf Compilation

CHS uses Protobufs for defining the "Bill of Health" structure. `pip install -e .` compiles them automatically (via `setup.py`).
You can manually trigger compilation if needed:

```bash
python3 setup.py build_py
```
This will regenerate files in `src/chs/protos/`.

### Container Image Dependencies & Local Docker Build

To build and test the CHS container image locally from source:

```bash
docker build -t chs:dev .
```

The image supports both `linux/amd64` and `linux/arm64`. To cross-compile for a different target architecture using Docker Buildx:

```bash
docker buildx build --platform linux/arm64 -t chs:dev-arm64 --load .
```

To build and publish a single multi-architecture manifest list covering both platforms:

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t <registry>/chs:dev --push .
```

The Docker image installs third-party Python packages only from hash-pinned lockfiles. Every `pip install` in the `Dockerfile` must either use `--require-hashes` or install a local source tree with `--no-deps --no-index`.

| Lockfile | Generated from | Installed into |
| --- | --- | --- |
| `requirements.txt` | `pyproject.toml` + `requirements-build.in` | The CHS virtualenv (`/opt/venv`) |
| `requirements-system.txt` | `requirements-system.in` | The base image's system Python (`/usr/local/bin/python3`) |

There are two exceptions. pip cannot hash-check git dependencies, so the image builds Ramble from a checkout of the commit pinned in `pyproject.toml`; always pin it to a full commit SHA. The TPU runtime (amd64 only) installs the TPU benchmarks' requirements, which are not hash-pinned yet.

The `pip-compile` pre-commit hooks keep the lockfiles in sync with their inputs. After changing dependencies, regenerate the lockfiles and commit them:

```bash
pre-commit run pip-compile --all-files
```

Existing pins are kept until you explicitly upgrade them. To pick up a newer release (for example, a security fix), rerun the `uv pip compile` command from the lockfile's header with `--upgrade-package <name>` (or `--upgrade` for everything).

## 2. Pre-commit Hooks

We use `pre-commit` to catch low-hanging fruit (trailing whitespace, file endings) and enforce code style (black, ruff).

### Installation

Install the pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

### Usage

The hooks will run automatically on `git commit`. If a hook fails, the commit will be blocked. Fix the issues and try committing again.

**Skipping Hooks:** If you need to skip the hooks for a specific commit, use the `--no-verify` flag:
```bash
git commit --no-verify
```

**Important:** Skipping pre-commit hooks is strongly discouraged, as they are crucial for maintaining the health of our codebase.

You can also run hooks manually on all files:

```bash
pre-commit run --all-files
```

**What is checked?**

*   **YAML:** Syntax and formatting (`check-yaml`).
*   **Python:** Formatting (`black`) and linting (`ruff`).
*   **Whitespace:** Trailing whitespace and end-of-file validation.
*   **Lockfiles:** `requirements.txt` and `requirements-system.txt` match their inputs (`pip-compile`, see [Container Image Dependencies & Local Docker Build](#container-image-dependencies--local-docker-build)).

## 3. Running Tests

All code changes must be tested. We use the standard `unittest` module along with `coverage` to measure code coverage.

### Running Unit Tests with Coverage

You can execute tests and check coverage manually:

```bash
# Run all tests and check coverage
coverage run -m unittest discover tests
coverage report --fail-under=65

# Run a specific test file
python3 -m unittest tests/test_ramble_manager.py
```

Code coverage is also enforced locally via our `pre-commit` hooks. You can run them manually with `pre-commit run --all-files`.

Tests should be run from the root of the repository. Ensure your virtual environment is activated and dependencies are installed.

## 4. Submitting Changes (GitHub Pull Requests)

All external contributions are developed in your personal fork and submitted via GitHub Pull Requests targeting the **`develop`** branch of `GoogleCloudPlatform/cluster-health-scanner`. Direct pushes to the upstream repository are disabled. See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor license agreement (CLA) and review details.

### Setup (First Time Only)

Ensure your Git author name and email match the account associated with your Google CLA:

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

### Workflow

1.  **Sync with `upstream/develop` and create a feature branch in your fork:**
    ```bash
    git fetch upstream
    git checkout -b feature/my-new-feature upstream/develop
    ```

2.  **Make changes, validate locally, and commit:**
    ```bash
    pre-commit run --all-files
    coverage run -m unittest discover tests
    git add .
    git commit
    ```
    *   **Commit Message:** Follow standard Git commit conventions:
        *   First line: Short summary (imperative mood).
        *   Body: Detailed explanation of what changed and why.
        *   Footer: GitHub Issue reference if applicable (e.g., `Fixes #123`).

3.  **Push the branch to your fork (`origin`) and open a Pull Request:**
    ```bash
    git push -u origin feature/my-new-feature
    ```
    On GitHub, open a Pull Request from your fork's `feature/my-new-feature` branch targeting the **`develop`** branch of `GoogleCloudPlatform/cluster-health-scanner` (`main` is reserved for stable releases and does not accept PRs).

4.  **Automated Checks & Review:**
    *   GitHub Actions will verify `google-cla`, `pre-commit` checks, unit tests, and vulnerability scanning.
    *   Address review comments by pushing additional commits to your fork's branch or rebasing onto `upstream/develop`. Once approved by a maintainer, the PR will be imported, validated, and merged automatically.

## 5. Adding Plugins (Directory-Based)

For better scalability and organization, CHS uses a directory-based plugin architecture. Each plugin should be its own Python package.

### 1. Create the Plugin Directory

Create a new directory in `src/chs/plugins/` (e.g., `src/chs/plugins/my_test/`). Ensure it contains an `__init__.py` file.

### 2. Implement the Components

You can split your plugin into multiple files for better organization:

*   `src/chs/plugins/my_test/renderer.py`: Visualization logic (Rich renderables).
*   `src/chs/plugins/my_test/extractor.py`: Logic for extracting test metadata/metrics.
*   `src/chs/plugins/my_test/__init__.py`: Plugin registration and core class.

### 3. Implement the Plugin Class

In `src/chs/plugins/my_test/__init__.py`, define a class inheriting from `TestPlugin`. You can import logic from your sub-modules:

```python
from chs.core.plugin_base import TestPlugin
from .renderer import MyTestRenderer
from .extractor import MyTestExtractor

class MyTestPlugin(TestPlugin, MyTestRenderer, MyTestExtractor):
    """A plugin that combines renderer and extractor components."""
    pass
```

Alternative: Just implement everything in one class in `__init__.py`.

### 4. Register the Plugin

In `src/chs/plugins/my_test/__init__.py`, define the `__chs_plugin__` dictionary:

```python
from .plugin import MyTestPlugin

__chs_plugin__ = {
    "test_name": "my-test",
    "plugin": MyTestPlugin,
}
```

**Note:**
The `test_name` is used as a prefix to match against the raw Ramble experiment name. If multiple plugins match, the longest prefix wins.

### 5. Future Extensibility

By using a directory-based structure, you can easily add other abstractions (e.g., `data_processors`, `validators`) as separate files within the plugin package, keeping the codebase clean and modular.

## 6. Resources

*   [Contributing Guide](CONTRIBUTING.md): CLA, pull request lifecycle, and review guidelines.
*   [README](README.md): Project overview and usage instructions.
