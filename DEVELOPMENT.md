# Developer Workflow

This document outlines the workflow for developing, testing, and contributing to the Cluster Health Scanner (CHS) project.
All development should be done on the `develop` branch. Main branch is strictly protected and should only be used for releases.

## 1. Getting Started

### Prerequisites

*   Python 3.10 or higher
*   `git`
*   `pip`
*   Access to the Gerrit repository

### Cloning the Repository

Clone the repository from Gerrit:

```bash
# Using HTTPS
git clone -b develop https://cluster-toolkit.googlesource.com/cluster-health-scanner chs

# Using SSO
git clone -b develop https://github.com/GoogleCloudPlatform/cluster-health-scanner.git chs

cd chs
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

**Note:** CHS depends on a specific branch of [Ramble](https://github.com/GoogleCloudPlatform/ramble). The `pyproject.toml` is configured to fetch it from the correct fork/branch.


### Protobuf Compilation

CHS uses Protobufs for defining the "Bill of Health" structure. `pip install -e .` compiles them automatically (via `setup.py`).
You can manually trigger compilation if needed:

```bash
python3 setup.py build_py
```
This will regenerate files in `src/chs/protos/`.

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

## 3. Running Tests

All code changes must be tested. We use `unittest` (or `pytest` if configured) for testing.

### Running Unit Tests

We use the standard `unittest` module for unit testing.

```bash
# Run all tests
python3 -m unittest discover tests

# Run a specific test file
python3 -m unittest tests/test_ramble_manager.py
```

Tests should be run from the root of the repository. Ensure your virtual environment is activated and dependencies are installed.

## 4. Code Review Process (Gerrit)

We use Gerrit for code reviews.

### Setup (First Time Only)

Ensure your git user and email match your Gerrit account.

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

### Workflow

1.  **Create a local branch:**
    ```bash
    git checkout -b feature/my-new-feature
    ```

2.  **Make changes and commit:**
    ```bash
    git add .
    git commit
    ```
    *   **Commit Message:** Follow the guidelines in [styleguide.md](.gemini/styleguide.md).
        *   First line: Short summary (Imperative mood).
        *   Body: Detailed explanation.
        *   Footer: `Bug: b/12345`.

3.  **Upload for Review:**
    ```bash
    git push origin HEAD:refs/for/develop
    ```

4.  **Address Feedback:**
    *   make changes locally.
    *   Amend the previous commit:
        ```bash
        git add .
        git commit --amend
        ```
    *   Push the update:
        ```bash
        git push origin HEAD:refs/for/develop
        ```

## 5. AI-Assisted Development & Review

We leverage the **Gerrit Review Agent** to provide automated, AI-driven insights during the code review process. The agent automatically analyzes CLs and suggests improvements based on project-specific context.

### Coordination with `.gemini/`

The Review Agent (and other Gemini-based dev tools) uses the files in the `.gemini/` directory to understand the project's architecture, goals, and coding standards:

*   **`GEMINI.md`**: Provides the project context, architecture overview, and layer responsibilities.
*   **`styleguide.md`**: Outlines the project's coding style, and best practices.

Contributors are encouraged to keep these files updated as the project evolves, as the quality of the AI's feedback directly depends on the accuracy of this documentation.

## 7. Adding Plugins (Directory-Based)

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

> [!NOTE]
> The `test_name` is used as a prefix to match against the raw Ramble experiment name. If multiple plugins match, the longest prefix wins.

### 5. Future Extensibility

By using a directory-based structure, you can easily add other abstractions (e.g., `data_processors`, `validators`) as separate files within the plugin package, keeping the codebase clean and modular.

## 8. Resources

*   [Style Guide](.gemini/styleguide.md): Coding standards and conventions.
*   [README](README.md): Project overview and usage instructions.
