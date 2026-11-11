# Development guide

All development targets the `develop` branch (`main` is reserved for releases).

## 1. Setup

Requires Python 3.12+ and `git`.

1. Fork [GoogleCloudPlatform/cluster-health-scanner](https://github.com/GoogleCloudPlatform/cluster-health-scanner) on GitHub (uncheck **"Copy the `main` branch only"** so your fork includes `develop`), then clone it and add `upstream`:

   ```bash
   git clone -b develop https://github.com/<your-username>/cluster-health-scanner.git chs
   cd chs
   git remote add upstream https://github.com/GoogleCloudPlatform/cluster-health-scanner.git
   ```

2. Create a virtual environment and install CHS in editable mode:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -e .
   ```

3. Install the pre-commit hooks (runs formatting and lint checks on `git commit`):

   ```bash
   pip install pre-commit
   pre-commit install
   ```

## 2. Making changes

Create a branch from `upstream/develop`:

```bash
git fetch upstream
git checkout -b feature/my-change upstream/develop
```

To add or modify a benchmark test, see the [Test development guide](test-development.md).

## 3. Running checks and tests

Run the linter/formatter hooks and the unit test suite from the repository root before committing:

```bash
# Run formatting and lint checks
pre-commit run --all-files

# Run all unit tests with coverage
coverage run -m unittest discover tests
coverage report --fail-under=65

# Run a single test file
python3 -m unittest tests/test_ramble_manager.py
```

### Building the container image locally (optional)

```bash
# Build for the host architecture
docker build -t chs:dev .

# Cross-compile for ARM64
docker buildx build --platform linux/arm64 -t chs:dev-arm64 --load .
```

## 4. Submitting a pull request

1. Ensure your Git author email matches your [Google CLA](https://cla.developers.google.com/).
2. Commit and push your branch to your fork:

   ```bash
   git push -u origin feature/my-change
   ```

3. Open a pull request on GitHub against the **`develop`** branch of `GoogleCloudPlatform/cluster-health-scanner`.
4. GitHub Actions runs the CLA check, `pre-commit` hooks, and unit tests. Contributors will not see an option to merge the pull request themselves; once a maintainer approves it and all checks pass, the maintainers will merge it. See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.
