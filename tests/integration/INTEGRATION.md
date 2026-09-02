# Integration Tests Description

## Running Integration Tests on Specific Clusters

You can run integration tests for a particular set of clusters using pytest's marker (`-m`) or keyword filtering (`-k`) flags.

### By Accelerator Type (Using Markers)
The test suite utilizes custom markers for different accelerator types. You can run tests exclusively for one type:

- **TPU clusters only**:
  ```bash
  pytest -m tpu tests/integration/
  ```

- **GPU clusters only**:
  ```bash
  pytest -m gpu tests/integration/
  ```

### By Cluster Topology/Version (Using Keyword Filtering)
Because test parameters include the configuration file paths (e.g., `tpu/v6e/multi_pool_2x4_4x4.yaml`), you can filter by any substring in the path using the `-k` flag.

- **v6e clusters only**:
  ```bash
  pytest -k "v6e" tests/integration/
  ```

- **tpu7x clusters only**:
  ```bash
  pytest -k "tpu7x" tests/integration/
  ```

- **Specific configuration**:
  ```bash
  pytest -k "multi_pool_2x4_4x4" tests/integration/
  ```
